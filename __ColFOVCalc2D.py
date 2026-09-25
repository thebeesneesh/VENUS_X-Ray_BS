"""Square-aperture collimator geometry calculator.

Layout (left to right): Extraction -> Puller -> Cu collimator -> Cu/W
collimator -> X-ray detector. Distances are measured face-to-face between
the named items. Each collimator has a square aperture through its full length.
"""
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk


class CollimatorCalculator:
    COMPONENTS = ("Extraction", "Puller", "Cu collimator", "Cu/W collimator", "X-ray detector")
    DEFAULTS = {
        "extraction_to_detector": 2580.3,
        "extraction_to_puller": 20.0,
        "puller_to_cu": 1700.0,
        "cu_to_w": 150.0,
        "w_to_detector": 100.0,
        "cu_length": 25.0,
        "w_length": 25.0,
        "cu_aperture": 2.0,
        "w_aperture": 2.0,
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Square Collimator Calculator")
        self.root.minsize(900, 690)
        self.config_file = Path(__file__).with_name("collimator_configs.json")
        self.saved_configs = self._read_configs()
        self.values = self.DEFAULTS.copy()
        self.vars = {key: tk.StringVar(value=f"{value:g}") for key, value in self.values.items()}

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main = ttk.Frame(root, padding=14)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="X-ray Collimator Geometry", font=("Segoe UI", 16, "bold")).grid(sticky="w")
        ttk.Label(main, text="All components are axially aligned. Distances are face-to-face in mm.").grid(sticky="w", pady=(0, 10))
        self._build_config_controls(main)
        self._build_inputs(main)
        self._build_results(main)

        diagram = ttk.LabelFrame(main, text="Axial layout (not to scale)", padding=8)
        diagram.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        diagram.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(diagram, height=200, background="white", highlightthickness=0)
        self.canvas.grid(sticky="ew")
        self.canvas.bind("<Configure>", lambda _event: self.calculate())
        self.calculate()

    def _build_config_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Saved configurations", padding=8)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(frame, text="Configuration:").grid(row=0, column=0, padx=(0, 5))
        self.config_var = tk.StringVar()
        self.config_box = ttk.Combobox(frame, textvariable=self.config_var, state="readonly", width=28)
        self.config_box.grid(row=0, column=1, padx=(0, 8))
        for column, (label, command) in enumerate((("Load", self.load_config), ("Save as…", self.save_as), ("Update", self.update_config), ("Delete", self.delete_config)), start=2):
            ttk.Button(frame, text=label, command=command).grid(row=0, column=column, padx=3)
        self._refresh_config_box()

    def _build_inputs(self, parent):
        frame = ttk.LabelFrame(parent, text="Geometry", padding=10)
        frame.grid(row=2, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)
        rows = (
            ("extraction_to_detector", "Extraction → X-ray detector", "mm", "1 to 10,000"),
            ("extraction_to_puller", "Extraction → Puller", "mm", "1 to 10,000"),
            ("puller_to_cu", "Puller → Cu collimator", "mm", "1 to 10,000"),
            ("cu_to_w", "Cu collimator → W collimator", "mm", "1 to 10,000"),
            ("w_to_detector", "W detector-facing face to X-ray detector", "mm", "0.1 to 10,000"),
            ("cu_length", "Cu collimator length", "mm", "0.1 to 1,000"),
            ("w_length", "W collimator length", "mm", "0.1 to 1,000"),
            ("cu_aperture", "Cu square-aperture side length", "mm", "0.1 to 10"),
            ("w_aperture", "W square-aperture side length", "mm", "0.1 to 10"),
        )
        for row, (key, label, unit, hint) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(frame, textvariable=self.vars[key], width=14)
            entry.grid(row=row, column=1, sticky="w", padx=8, pady=3)
            entry.bind("<Return>", lambda _event: self.calculate())
            entry.bind("<FocusOut>", lambda _event: self.calculate())
            ttk.Label(frame, text=f"{unit}  ({hint})", foreground="#555555").grid(row=row, column=2, sticky="w")
        ttk.Button(frame, text="Calculate", command=self.calculate).grid(row=len(rows), column=1, sticky="w", pady=(8, 0))

    def _build_results(self, parent):
        frame = ttk.LabelFrame(parent, text="Calculated acceptance", padding=10)
        frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.fov_label = ttk.Label(frame, font=("Segoe UI", 12, "bold"))
        self.fov_label.grid(row=0, column=0, sticky="w")
        self.angle_label = ttk.Label(frame)
        self.angle_label.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.limit_label = ttk.Label(frame)
        self.limit_label.grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Label(frame, text="FOV is the square region at the extraction plane whose rays reach the detector centre without clipping.", foreground="#555555", wraplength=800).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _read_configs(self):
        try:
            data = json.loads(self.config_file.read_text(encoding="utf-8"))
            required = set(self.DEFAULTS)
            return {
                name: values for name, values in data.items()
                if isinstance(values, dict) and required.issubset(values)
            }
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_configs(self):
        try:
            self.config_file.write_text(json.dumps(self.saved_configs, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save error", f"Could not save configurations:\n{exc}")

    def _refresh_config_box(self):
        names = sorted(self.saved_configs)
        self.config_box["values"] = names
        if self.config_var.get() not in names:
            self.config_var.set(names[0] if names else "")

    def _get_values(self, show_error=True):
        result = {}
        for key, default in self.DEFAULTS.items():
            try:
                value = float(self.vars[key].get())
            except ValueError:
                if show_error:
                    messagebox.showerror("Invalid value", f"Enter a number for {key.replace('_', ' ')}.")
                return None
            maximum = 10.0 if key.endswith("aperture") else (1000.0 if key.endswith("length") else 10000.0)
            if not 0.1 <= value <= maximum:
                if show_error:
                    messagebox.showerror("Value out of range", f"{key.replace('_', ' ').title()} must be from 0.1 to {maximum:g} mm.")
                return None
            result[key] = value
        return result

    def calculate(self):
        values = self._get_values(show_error=False)
        if values is None:
            return
        self.values = values
        # Coordinates from detector (0) to extraction. A ray ends at detector centre.
        # Coordinate distances point from the detector toward extraction. The
        # far (extraction-facing) aperture faces impose the tightest limits.
        w_near = values["w_to_detector"]
        w_far = w_near + values["w_length"]
        cu_near = w_far + values["cu_to_w"]
        cu_far = cu_near + values["cu_length"]
        detector_to_extraction = cu_far + values["puller_to_cu"] + values["extraction_to_puller"]
        limits = {
            "Cu": values["cu_aperture"] * detector_to_extraction / cu_far,
            "W": values["w_aperture"] * detector_to_extraction / w_far,
        }
        limiting, fov_side = min(limits.items(), key=lambda item: item[1])
        half_angle = math.degrees(math.atan((fov_side / 2) / detector_to_extraction))
        self.fov_label.config(text=f"Usable square FOV at extraction: {fov_side:.3f} mm × {fov_side:.3f} mm")
        self.angle_label.config(text=f"Maximum half-angle to detector centre: {half_angle:.4f}°")
        self.limit_label.config(text=f"Limiting aperture: {limiting} collimator ({limits[limiting]:.3f} mm projected FOV side)")
        self._draw(values, detector_to_extraction, cu_near, cu_far, w_near, w_far, fov_side)

    def _draw(self, values, extraction, cu_near, cu_far, w_near, w_far, fov_side):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 760)
        left, right, centre = 42, width - 42, 98
        def x(position): return right - position / extraction * (right - left)
        self.canvas.create_line(left, centre, right, centre, fill="#888", dash=(5, 4))
        items = (("Extraction", extraction, "#c44"), ("Puller", extraction - values["extraction_to_puller"], "#666"))
        for name, position, color in items:
            xpos = x(position)
            self.canvas.create_line(xpos, 38, xpos, 158, fill=color, width=5)
            self.canvas.create_text(xpos, 177, text=name, font=("Segoe UI", 8, "bold"))
        detector_x = x(0)
        self.canvas.create_rectangle(detector_x - 8, 43, detector_x + 8, 153,
                                     fill="#3572a5", outline="#16466e", width=2)
        self.canvas.create_rectangle(detector_x - 4, 53, detector_x + 4, 143,
                                     fill="#8fc5e9", outline="")
        self.canvas.create_text(detector_x, 177, text="X-ray detector", font=("Segoe UI", 8, "bold"))
        for name, near, far, aperture, color in (("Cu", cu_near, cu_far, values["cu_aperture"], "#b87333"), ("W", w_near, w_far, values["w_aperture"], "#555")):
            xpos_near, xpos_far = x(near), x(far)
            self.canvas.create_rectangle(xpos_far, 48, xpos_near, 148, fill=color, outline=color)
            opening = max(8, min(36, aperture * 5))
            self.canvas.create_rectangle(xpos_far - 1, centre - opening / 2, xpos_near + 1, centre + opening / 2, fill="white", outline="white")
            self.canvas.create_text((xpos_near + xpos_far) / 2, 177, text=f"{name}\n{far-near:g} mm", font=("Segoe UI", 8, "bold"))
        yhalf = min(55, max(8, fov_side / 2 / extraction * (right - left)))
        self.canvas.create_line(x(extraction), centre-yhalf, left, centre, fill="#7b3fa1", width=2)
        self.canvas.create_line(x(extraction), centre+yhalf, left, centre, fill="#7b3fa1", width=2)

    def save_as(self):
        values = self._get_values()
        if values is None:
            return
        name = simpledialog.askstring("Save configuration", "Configuration name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.saved_configs and not messagebox.askyesno("Replace configuration", f"Replace '{name}'?"):
            return
        self.saved_configs[name] = values
        self._write_configs()
        self._refresh_config_box()
        self.config_var.set(name)

    def load_config(self):
        values = self.saved_configs.get(self.config_var.get())
        if values is None:
            messagebox.showwarning("No configuration", "Choose a saved configuration to load.")
            return
        for key, default in self.DEFAULTS.items():
            self.vars[key].set(f"{float(values.get(key, default)):g}")
        self.calculate()

    def update_config(self):
        name = self.config_var.get()
        values = self._get_values()
        if not name or values is None:
            messagebox.showwarning("No configuration", "Choose a saved configuration to update.")
            return
        self.saved_configs[name] = values
        self._write_configs()

    def delete_config(self):
        name = self.config_var.get()
        if name and messagebox.askyesno("Delete configuration", f"Delete '{name}'?"):
            del self.saved_configs[name]
            self._write_configs()
            self._refresh_config_box()


if __name__ == "__main__":
    root = tk.Tk()
    CollimatorCalculator(root)
    root.mainloop()

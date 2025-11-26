import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import random

# ============================================================
#   RAY TRACING CORE (2D)
# ============================================================

def ray_intersects_cylinder_wall(origin, direction, z_entry, z_exit, radius):
    """
    Checks full wall intersection: does the ray ever cross |x| > radius
    while traveling from z_entry to z_exit?
    """
    ox, oz = origin
    dx, dz = direction

    # Find z where ray reaches x = ±radius
    # x = ox + dx*t = ±radius → t = (±radius - ox)/dx
    if abs(dx) < 1e-12:
        # Ray vertical → only check entry/exit x positions
        return abs(ox) > radius

    t1 = ( radius - ox) / dx
    t2 = (-radius - ox) / dx

    for t in (t1, t2):
        if t > 0:
            z_hit = oz + dz * t
            if z_entry <= z_hit <= z_exit:
                return True

    return False


def aperture_only_test(origin, direction, z_entry, z_exit, radius):
    """
    Only checks apertures (entry and exit openings),
    ignores wall intersection between them.
    """
    ox, oz = origin
    dx, dz = direction
    if dz == 0:
        return False

    # Entry intersection
    t_entry = (z_entry - oz)/dz
    if t_entry <= 0:
        return False
    x_entry = ox + dx*t_entry
    if abs(x_entry) > radius:
        return False

    # Exit intersection
    t_exit = (z_exit - oz)/dz
    if t_exit <= 0:
        return False
    x_exit = ox + dx*t_exit
    if abs(x_exit) > radius:
        return False

    return True


def collimator_pass(origin, direction, z_entry, length, radius, aperture_only):
    """
    Determine if ray passes through collimator.
    """
    z_exit = z_entry + length

    if aperture_only:
        return aperture_only_test(origin, direction, z_entry, z_exit, radius)

    # Full wall physics
    # Check entry aperture
    ox, oz = origin
    dx, dz = direction
    if dz == 0:
        return False
    t_entry = (z_entry - oz)/dz
    if t_entry <= 0:
        return False
    x_entry = ox + dx*t_entry
    if abs(x_entry) > radius:
        return False

    # Check for wall hit
    if ray_intersects_cylinder_wall(origin, direction, z_entry, z_exit, radius):
        return False

    # Check exit aperture
    t_exit = (z_exit - oz)/dz
    if t_exit <= 0:
        return False
    x_exit = ox + dx*t_exit
    if abs(x_exit) > radius:
        return False

    return True


def run_simulation(N, source_radius, cone_angle_deg, 
                   c1_z, c1_len, c1_rad,
                   c2_z, c2_len, c2_rad,
                   det_z,
                   aperture_only):
    """
    Run 2D Monte-Carlo ray tracing.
    Returns list of detector hit x-positions.
    """
    hits = []
    cone = math.radians(cone_angle_deg)

    for _ in range(N):
        # Source point
        x0 = (random.random()*2 - 1) * source_radius
        origin = (x0, 0)

        # Direction
        angle = (random.random()*2 - 1) * cone
        direction = (math.sin(angle), math.cos(angle))

        # Pass collimator 1?
        if not collimator_pass(origin, direction, c1_z, c1_len, c1_rad, aperture_only):
            continue

        # Pass collimator 2?
        if not collimator_pass(origin, direction, c2_z, c2_len, c2_rad, aperture_only):
            continue

        # Hit detector plane
        ox, oz = origin
        dx, dz = direction
        t_det = (det_z - oz)/dz
        x_det = ox + dx*t_det
        hits.append(x_det)

    return hits


# ============================================================
#   GUI + ANIMATION
# ============================================================

class CollimatorGUI:
    def __init__(self, root):
        self.root = root
        root.title("2D Collimator Ray-Tracing Simulator")

        # Frame layout
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # -----------------------------
        # Sliders
        # -----------------------------
        def add_slider(label, from_, to_, init):
            lbl = ttk.Label(control_frame, text=label)
            lbl.pack(anchor="w")
            sld = ttk.Scale(control_frame, from_=from_, to=to_, orient="horizontal")
            sld.set(init)
            sld.pack(fill=tk.X)
            return sld

        self.s_N = add_slider("Number of Rays", 100, 5000, 1000)
        self.s_source_r = add_slider("Source Radius (mm)", 0.0, 2.0, 0.5)
        self.s_cone = add_slider("Cone Angle (deg)", 0.1, 10, 2)

        ttk.Label(control_frame, text="--- Collimator 1 ---").pack()
        self.s_c1_z   = add_slider("C1 Entry Z", 5, 200, 20)
        self.s_c1_len = add_slider("C1 Length", 5, 200, 50)
        self.s_c1_rad = add_slider("C1 Radius", 0.1, 5, 0.5)

        ttk.Label(control_frame, text="--- Collimator 2 ---").pack()
        self.s_c2_z   = add_slider("C2 Entry Z", 50, 350, 150)
        self.s_c2_len = add_slider("C2 Length", 5, 200, 50)
        self.s_c2_rad = add_slider("C2 Radius", 0.1, 5, 0.5)

        self.s_det_z = add_slider("Detector Z", 100, 500, 300)

        # Toggle for aperture-only mode
        self.aperture_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, 
                        text="Aperture Only (no wall physics)", 
                        variable=self.aperture_only_var).pack(pady=10)

        # Run button
        ttk.Button(control_frame, text="Start Animation", command=self.start_animation).pack(pady=20)

        # -----------------------------
        # Matplotlib figure
        # -----------------------------
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.anim = None

    def start_animation(self):
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None

        self.anim = FuncAnimation(self.fig, self.update_frame, 
                                  frames=200, interval=100, blit=False)
        self.canvas.draw()

    def update_frame(self, frame):
        # Read GUI values
        N = int(self.s_N.get())
        source_r = self.s_source_r.get()
        cone = self.s_cone.get()

        c1_z = self.s_c1_z.get()
        c1_len = self.s_c1_len.get()
        c1_rad = self.s_c1_rad.get()

        c2_z = self.s_c2_z.get()
        c2_len = self.s_c2_len.get()
        c2_rad = self.s_c2_rad.get()

        det_z = self.s_det_z.get()

        aperture_only = self.aperture_only_var.get()

        # Run ray simulation
        hits = run_simulation(N, source_r, cone,
                              c1_z, c1_len, c1_rad,
                              c2_z, c2_len, c2_rad,
                              det_z, aperture_only)

        # Update plot
        self.ax.clear()
        self.ax.set_title("2D Collimator Ray-Tracing")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Z (mm)")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, det_z + 20)

        # Draw collimators
        self.ax.plot([-c1_rad, c1_rad], [c1_z, c1_z], 'r')
        self.ax.plot([-c1_rad, c1_rad], [c1_z + c1_len, c1_z + c1_len], 'r')

        self.ax.plot([-c2_rad, c2_rad], [c2_z, c2_z], 'b')
        self.ax.plot([-c2_rad, c2_rad], [c2_z + c2_len, c2_z + c2_len], 'b')

        # Draw detector hits
        if len(hits) > 0:
            self.ax.scatter(hits, [det_z]*len(hits), s=10, color="green")

            # Beam-spot width
            width = max(hits) - min(hits)
            self.ax.text(0.05, 0.95, 
                         f"Beam Spot Width = {width:.3f} mm", 
                         transform=self.ax.transAxes)

        return []


# ============================================================
#   MAIN
# ============================================================

root = tk.Tk()
gui = CollimatorGUI(root)
root.mainloop()

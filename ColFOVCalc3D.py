import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import math
import json
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class XRayCollimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Beam Collimator Calculator - 3D View")
        self.root.geometry("1100x900")
        
        self.config_file = "collimator_configs.json"
        self.saved_configs = self.load_configs_from_file()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="X-Ray Detector Field of View Calculator - 3D", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Configuration management frame
        config_frame = ttk.LabelFrame(main_frame, text="Saved Configurations", padding="10")
        config_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Config dropdown
        ttk.Label(config_frame, text="Configuration:").grid(row=0, column=0, padx=5)
        self.config_var = tk.StringVar()
        self.config_dropdown = ttk.Combobox(config_frame, textvariable=self.config_var, 
                                            width=30, state="readonly")
        self.config_dropdown.grid(row=0, column=1, padx=5)
        self.update_config_dropdown()
        
        # Buttons
        ttk.Button(config_frame, text="Load", command=self.load_config).grid(row=0, column=2, padx=5)
        ttk.Button(config_frame, text="Save As...", command=self.save_config).grid(row=0, column=3, padx=5)
        ttk.Button(config_frame, text="Update", command=self.update_config).grid(row=0, column=4, padx=5)
        ttk.Button(config_frame, text="Delete", command=self.delete_config).grid(row=0, column=5, padx=5)
        ttk.Button(config_frame, text="Export All", command=self.export_configs).grid(row=0, column=6, padx=5)
        
        # Parameters
        self.params = {
            'source_to_col1': {'value': 300, 'min': 10, 'max': 2000, 'label': 'X-ray Source to Collimator 2 (mm)'},
            'col1_length': {'value': 50, 'min': 10, 'max': 200, 'label': 'Collimator 2 Length (mm)'},
            'col1_radius': {'value': 5, 'min': 1, 'max': 50, 'label': 'Collimator 2 Radius (mm)'},
            'col1_to_col2': {'value': 200, 'min': 10, 'max': 1000, 'label': 'Distance Col2 to Col1 (mm)'},
            'col2_length': {'value': 50, 'min': 10, 'max': 200, 'label': 'Collimator 1 Length (mm)'},
            'col2_radius': {'value': 5, 'min': 1, 'max': 50, 'label': 'Collimator 1 Radius (mm)'},
            'col2_to_target': {'value': 100, 'min': 10, 'max': 500, 'label': 'Distance Col1 to Detector (mm)'},
        }
        
        self.sliders = {}
        self.value_labels = {}
        self.entry_vars = {}
        self.entries = {}
        
        # Create sliders with entry fields
        row = 2
        for key, param in self.params.items():
            label = ttk.Label(main_frame, text=param['label'])
            label.grid(row=row, column=0, sticky=tk.W, pady=5)
            
            slider = ttk.Scale(main_frame, from_=param['min'], to=param['max'],
                             orient=tk.HORIZONTAL, length=300,
                             command=lambda v, k=key: self.update_from_slider(k, v))
            slider.set(param['value'])
            slider.grid(row=row, column=1, padx=10, pady=5)
            self.sliders[key] = slider
            
            value_label = ttk.Label(main_frame, text=f"{param['value']:.1f}")
            value_label.grid(row=row, column=2, sticky=tk.W, pady=5, padx=5)
            self.value_labels[key] = value_label
            
            entry_var = tk.StringVar(value=f"{param['value']:.1f}")
            entry = ttk.Entry(main_frame, textvariable=entry_var, width=10)
            entry.grid(row=row, column=3, padx=5, pady=5)
            entry.bind('<Return>', lambda e, k=key: self.update_from_entry(k))
            entry.bind('<FocusOut>', lambda e, k=key: self.update_from_entry(k))
            self.entry_vars[key] = entry_var
            self.entries[key] = entry
            
            row += 1
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=row, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        self.beam_radius_label = ttk.Label(results_frame, text="Field of View Radius at Source: ", 
                                          font=('Arial', 12, 'bold'))
        self.beam_radius_label.grid(row=0, column=0, pady=5, sticky=tk.W)
        
        self.limiting_edges_label = ttk.Label(results_frame, text="Limiting Edges: ")
        self.limiting_edges_label.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        self.angle_label = ttk.Label(results_frame, text="Maximum Acceptance Angle: ")
        self.angle_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # 3D Diagram frame
        diagram_frame = ttk.LabelFrame(main_frame, text="3D Ray Path Through Collimators", 
                                      padding="10")
        diagram_frame.grid(row=row+1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Create matplotlib figure for 3D
        self.fig = Figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=diagram_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enable mouse rotation
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initial calculation
        self.calculate()
    
    def on_scroll(self, event):
        """Handle mouse scroll for zoom"""
        if event.button == 'up':
            self.ax.dist *= 0.9
        elif event.button == 'down':
            self.ax.dist *= 1.1
        self.canvas.draw()
    
    def load_configs_from_file(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def save_configs_to_file(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.saved_configs, f, indent=2)
        except IOError as e:
            messagebox.showerror("Error", f"Could not save configurations: {e}")
    
    def update_config_dropdown(self):
        config_names = list(self.saved_configs.keys())
        self.config_dropdown['values'] = config_names
        if config_names:
            self.config_var.set(config_names[0])
        else:
            self.config_var.set('')
    
    def get_current_values(self):
        return {key: self.params[key]['value'] for key in self.params}
    
    def set_values(self, values):
        for key, value in values.items():
            if key in self.params:
                self.params[key]['value'] = value
                self.sliders[key].set(value)
                self.value_labels[key].config(text=f"{value:.1f}")
                self.entry_vars[key].set(f"{value:.1f}")
        self.calculate()
    
    def save_config(self):
        name = simpledialog.askstring("Save Configuration", 
                                      "Enter a name for this configuration:",
                                      parent=self.root)
        if name:
            if name in self.saved_configs:
                if not messagebox.askyesno("Confirm Overwrite", 
                                          f"Configuration '{name}' already exists. Overwrite?"):
                    return
            
            self.saved_configs[name] = self.get_current_values()
            self.save_configs_to_file()
            self.update_config_dropdown()
            self.config_var.set(name)
            messagebox.showinfo("Saved", f"Configuration '{name}' saved successfully!")
    
    def update_config(self):
        name = self.config_var.get()
        if not name:
            messagebox.showwarning("Warning", "No configuration selected!")
            return
        
        if messagebox.askyesno("Confirm Update", 
                              f"Update configuration '{name}' with current values?"):
            self.saved_configs[name] = self.get_current_values()
            self.save_configs_to_file()
            messagebox.showinfo("Updated", f"Configuration '{name}' updated!")
    
    def load_config(self):
        name = self.config_var.get()
        if not name:
            messagebox.showwarning("Warning", "No configuration selected!")
            return
        
        if name in self.saved_configs:
            self.set_values(self.saved_configs[name])
            messagebox.showinfo("Loaded", f"Configuration '{name}' loaded!")
        else:
            messagebox.showerror("Error", f"Configuration '{name}' not found!")
    
    def delete_config(self):
        name = self.config_var.get()
        if not name:
            messagebox.showwarning("Warning", "No configuration selected!")
            return
        
        if messagebox.askyesno("Confirm Delete", 
                              f"Delete configuration '{name}'?"):
            del self.saved_configs[name]
            self.save_configs_to_file()
            self.update_config_dropdown()
            messagebox.showinfo("Deleted", f"Configuration '{name}' deleted!")
    
    def export_configs(self):
        if not self.saved_configs:
            messagebox.showinfo("Export", "No configurations to export!")
            return
        
        export_text = "X-Ray Collimator Configurations\n"
        export_text += "=" * 50 + "\n\n"
        
        for name, values in self.saved_configs.items():
            export_text += f"Configuration: {name}\n"
            export_text += "-" * 30 + "\n"
            for key, value in values.items():
                label = self.params[key]['label']
                export_text += f"  {label}: {value:.1f}\n"
            export_text += "\n"
        
        export_file = "collimator_configs_export.txt"
        try:
            with open(export_file, 'w') as f:
                f.write(export_text)
            messagebox.showinfo("Export", f"Configurations exported to '{export_file}'!")
        except IOError as e:
            messagebox.showerror("Error", f"Could not export: {e}")
    
    def update_from_slider(self, key, value):
        val = float(value)
        self.params[key]['value'] = val
        self.value_labels[key].config(text=f"{val:.1f}")
        self.entry_vars[key].set(f"{val:.1f}")
        self.calculate()
    
    def update_from_entry(self, key):
        try:
            val = float(self.entry_vars[key].get())
            min_val = self.params[key]['min']
            max_val = self.params[key]['max']
            
            if val < min_val:
                val = min_val
            elif val > max_val:
                val = max_val
            
            self.params[key]['value'] = val
            self.sliders[key].set(val)
            self.value_labels[key].config(text=f"{val:.1f}")
            self.entry_vars[key].set(f"{val:.1f}")
            self.calculate()
            
        except ValueError:
            val = self.params[key]['value']
            self.entry_vars[key].set(f"{val:.1f}")
    
    def calculate_ray(self, pos_source, edge1_pos, edge1_r, edge2_pos, edge2_r):
        if edge1_pos == edge2_pos:
            return None, None, None
        
        slope = (edge1_r - edge2_r) / (edge1_pos - edge2_pos)
        r_source = edge1_r + slope * (pos_source - edge1_pos)
        r_detector = edge2_r - slope * edge2_pos
        
        return r_source, r_detector, slope
    
    def check_ray_clears_collimator(self, edge1_pos, edge1_r, edge2_pos, edge2_r,
                                     col_entrance_pos, col_exit_pos, col_radius):
        if edge1_pos == edge2_pos:
            return True
        
        slope = (edge1_r - edge2_r) / (edge1_pos - edge2_pos)
        
        r_at_entrance = edge2_r + slope * (col_entrance_pos - edge2_pos)
        if abs(r_at_entrance) > col_radius + 1e-9:
            return False
        
        r_at_exit = edge2_r + slope * (col_exit_pos - edge2_pos)
        if abs(r_at_exit) > col_radius + 1e-9:
            return False
        
        return True
    
    def calculate(self):
        d_detector_to_col1 = self.params['col2_to_target']['value']
        l_col1 = self.params['col2_length']['value']
        r_col1 = self.params['col2_radius']['value']
        d_col1_to_col2 = self.params['col1_to_col2']['value']
        l_col2 = self.params['col1_length']['value']
        r_col2 = self.params['col1_radius']['value']
        d_col2_to_source = self.params['source_to_col1']['value']
        
        pos_col1_exit = d_detector_to_col1
        pos_col1_entrance = d_detector_to_col1 + l_col1
        pos_col2_exit = d_detector_to_col1 + l_col1 + d_col1_to_col2
        pos_col2_entrance = d_detector_to_col1 + l_col1 + d_col1_to_col2 + l_col2
        pos_source = pos_col2_entrance + d_col2_to_source
        
        vertices = [
            ("Col2 entrance top", pos_col2_entrance, +r_col2),
            ("Col2 entrance bot", pos_col2_entrance, -r_col2),
            ("Col2 exit top", pos_col2_exit, +r_col2),
            ("Col2 exit bot", pos_col2_exit, -r_col2),
            ("Col1 entrance top", pos_col1_entrance, +r_col1),
            ("Col1 entrance bot", pos_col1_entrance, -r_col1),
            ("Col1 exit top", pos_col1_exit, +r_col1),
            ("Col1 exit bot", pos_col1_exit, -r_col1),
        ]
        
        max_fov = 0
        best_edge1 = None
        best_edge2 = None
        best_slope = None
        best_r_detector = None
        
        for i, (name1, pos1, r1) in enumerate(vertices):
            for j, (name2, pos2, r2) in enumerate(vertices):
                if pos1 <= pos2:
                    continue
                
                r_source, r_det, slope = self.calculate_ray(pos_source, pos1, r1, pos2, r2)
                
                if r_source is None:
                    continue
                
                passes_col2 = self.check_ray_clears_collimator(
                    pos1, r1, pos2, r2,
                    pos_col2_exit, pos_col2_entrance, r_col2
                )
                
                passes_col1 = self.check_ray_clears_collimator(
                    pos1, r1, pos2, r2,
                    pos_col1_exit, pos_col1_entrance, r_col1
                )
                
                if not passes_col2 or not passes_col1:
                    continue
                
                fov = abs(r_source)
                if fov > max_fov:
                    max_fov = fov
                    best_edge1 = (name1, pos1, r1)
                    best_edge2 = (name2, pos2, r2)
                    best_slope = slope
                    best_r_detector = r_det
        
        if best_edge1 is None:
            self.beam_radius_label.config(
                text="No valid ray path found!",
                foreground='red'
            )
            self.limiting_edges_label.config(text="N/A")
            self.angle_label.config(text="N/A")
            return
        
        field_of_view_radius = max_fov
        max_angle = math.atan(field_of_view_radius / pos_source)
        
        self.beam_radius_label.config(
            text=f"Field of View Radius at X-ray Source: {field_of_view_radius:.2f} mm",
            foreground='black'
        )
        self.limiting_edges_label.config(
            text=f"Limiting Edges: {best_edge1[0]} → {best_edge2[0]}"
        )
        self.angle_label.config(
            text=f"Maximum Acceptance Angle: {math.degrees(max_angle):.3f}°"
        )
        
        self.draw_3d_diagram(field_of_view_radius, 
                            pos_col1_exit, pos_col1_entrance,
                            pos_col2_exit, pos_col2_entrance, pos_source,
                            l_col1, r_col1, l_col2, r_col2, 
                            best_edge1, best_edge2, best_slope, best_r_detector)
    
    def draw_cylinder(self, ax, z_start, z_end, radius, color='blue', alpha=0.3):
        """Draw a hollow cylinder (collimator) in 3D"""
        # Create cylinder surface
        theta = np.linspace(0, 2*np.pi, 50)
        z = np.array([z_start, z_end])
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)
        
        # Draw entrance and exit circles
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        ax.plot(x_circle, y_circle, [z_start]*len(theta), color=color, linewidth=2)
        ax.plot(x_circle, y_circle, [z_end]*len(theta), color=color, linewidth=2)
    
    def draw_disk(self, ax, z_pos, radius, color='red', alpha=0.5):
        """Draw a filled disk (source/detector)"""
        theta = np.linspace(0, 2*np.pi, 50)
        r = np.linspace(0, radius, 10)
        theta_grid, r_grid = np.meshgrid(theta, r)
        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)
        z_grid = np.ones_like(x_grid) * z_pos
        
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)
    
    def draw_cone_of_rays(self, ax, z_source, r_source, z_detector, r_detector, 
                          edge1_pos, edge1_r, edge2_pos, edge2_r, color='green', num_rays=12):
        """Draw a cone of rays from source to detector"""
        theta = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        
        for t in theta:
            # Source point
            x_src = r_source * np.cos(t)
            y_src = r_source * np.sin(t)
            
            # Edge 1 point
            x_e1 = edge1_r * np.cos(t)
            y_e1 = edge1_r * np.sin(t)
            
            # Edge 2 point
            x_e2 = edge2_r * np.cos(t)
            y_e2 = edge2_r * np.sin(t)
            
            # Detector point
            x_det = r_detector * np.cos(t)
            y_det = r_detector * np.sin(t)
            
            # Draw ray segments
            ax.plot([x_src, x_e1], [y_src, y_e1], [z_source, edge1_pos], 
                   color=color, linewidth=1.5, alpha=0.8)
            ax.plot([x_e1, x_e2], [y_e1, y_e2], [edge1_pos, edge2_pos], 
                   color=color, linewidth=1.5, alpha=0.8)
            ax.plot([x_e2, x_det], [y_e2, y_det], [edge2_pos, z_detector], 
                   color=color, linewidth=1.5, alpha=0.8)
    
    def draw_3d_diagram(self, fov_radius, 
                        pos_c1_exit, pos_c1_entrance,
                        pos_c2_exit, pos_c2_entrance, pos_source,
                        l1, r1, l2, r2, 
                        edge1, edge2, slope, r_detector):
        self.ax.clear()
        
        # Draw detector at z=0
        self.draw_disk(self.ax, 0, max(r1, r2) * 1.5, color='darkblue', alpha=0.6)
        self.ax.text(0, 0, -20, "Detector", fontsize=10, ha='center')
        
        # Draw Collimator 1 (nearest detector)
        self.draw_cylinder(self.ax, pos_c1_exit, pos_c1_entrance, r1, color='steelblue', alpha=0.4)
        self.ax.text(0, r1 + 5, (pos_c1_exit + pos_c1_entrance)/2, f"Col1\nr={r1:.1f}", fontsize=8)
        
        # Draw Collimator 2 (nearest source)
        self.draw_cylinder(self.ax, pos_c2_exit, pos_c2_entrance, r2, color='lightblue', alpha=0.4)
        self.ax.text(0, r2 + 5, (pos_c2_exit + pos_c2_entrance)/2, f"Col2\nr={r2:.1f}", fontsize=8)
        
        # Draw source FOV disk
        self.draw_disk(self.ax, pos_source, fov_radius, color='orange', alpha=0.6)
        self.ax.text(0, 0, pos_source + 20, f"Source FOV\nr={fov_radius:.1f}", fontsize=10, ha='center')
        
        if edge1 is not None and edge2 is not None:
            name1, pos1, r1_signed = edge1
            name2, pos2, r2_signed = edge2
            
            # Determine if crossing rays
            is_crossing = (r1_signed * r2_signed) < 0
            ray_color = 'purple' if is_crossing else 'green'
            
            # Draw cone of rays
            self.draw_cone_of_rays(self.ax, pos_source, fov_radius, 0, r_detector,
                                   pos1, r1_signed, pos2, r2_signed, 
                                   color=ray_color, num_rays=16)
            
            # Mark limiting edges with circles
            theta = np.linspace(0, 2*np.pi, 50)
            x_e1 = abs(r1_signed) * np.cos(theta)
            y_e1 = abs(r1_signed) * np.sin(theta)
            self.ax.plot(x_e1, y_e1, [pos1]*len(theta), color='red', linewidth=3)
            
            x_e2 = abs(r2_signed) * np.cos(theta)
            y_e2 = abs(r2_signed) * np.sin(theta)
            self.ax.plot(x_e2, y_e2, [pos2]*len(theta), color='red', linewidth=3)
        
        # Draw central axis
        self.ax.plot([0, 0], [0, 0], [0, pos_source], 'k--', alpha=0.3, linewidth=1)
        
        # Set labels and view
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z - Optical Axis (mm)')
        
        # Set aspect ratio
        max_range = max(pos_source, fov_radius * 3)
        self.ax.set_xlim([-max_range/4, max_range/4])
        self.ax.set_ylim([-max_range/4, max_range/4])
        self.ax.set_zlim([0, pos_source * 1.1])
        
        # Set initial view angle
        self.ax.view_init(elev=15, azim=45)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = XRayCollimatorGUI(root)
    root.mainloop()
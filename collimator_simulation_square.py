import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import math
import json
import os
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

class XRayCollimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Beam Collimator Calculator - Square Apertures")
        self.root.geometry("620x750")
        
        self.config_file = "collimator_configs.json"
        self.saved_configs = self.load_configs_from_file()
        
        # Create PyVista plotter
        self.plotter = BackgroundPlotter(title="X-Ray Collimator 3D View - Square Apertures", 
                                         window_size=(900, 700))
        self.plotter.add_axes()
        self.plotter.show_grid()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="X-Ray Detector FOV Calculator (Square Apertures)", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Configuration management frame
        config_frame = ttk.LabelFrame(main_frame, text="Saved Configurations", padding="10")
        config_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(config_frame, text="Configuration:").grid(row=0, column=0, padx=5)
        self.config_var = tk.StringVar()
        self.config_dropdown = ttk.Combobox(config_frame, textvariable=self.config_var, 
                                            width=25, state="readonly")
        self.config_dropdown.grid(row=0, column=1, padx=5)
        self.update_config_dropdown()
        
        ttk.Button(config_frame, text="Load", command=self.load_config).grid(row=0, column=2, padx=3)
        ttk.Button(config_frame, text="Save As...", command=self.save_config).grid(row=0, column=3, padx=3)
        ttk.Button(config_frame, text="Update", command=self.update_config).grid(row=0, column=4, padx=3)
        ttk.Button(config_frame, text="Delete", command=self.delete_config).grid(row=0, column=5, padx=3)
        
        # Parameters - now with half-width instead of radius
        self.params = {
            'source_to_col2': {'value': 300, 'min': 10, 'max': 2000, 'label': 'X-ray Source to Collimator 2 (mm)'},
            'col2_length': {'value': 50, 'min': 10, 'max': 200, 'label': 'Collimator 2 Length (mm)'},
            'col2_half_width': {'value': 5, 'min': 1, 'max': 50, 'label': 'Collimator 2 Aperture Half-Width (mm)'},
            'col2_to_col1': {'value': 200, 'min': 10, 'max': 1000, 'label': 'Distance Col2 to Col1 (mm)'},
            'col1_length': {'value': 50, 'min': 10, 'max': 200, 'label': 'Collimator 1 Length (mm)'},
            'col1_half_width': {'value': 5, 'min': 1, 'max': 50, 'label': 'Collimator 1 Aperture Half-Width (mm)'},
            'col1_to_detector': {'value': 100, 'min': 10, 'max': 500, 'label': 'Distance Col1 to Detector (mm)'},
        }
        
        self.sliders = {}
        self.value_labels = {}
        self.entry_vars = {}
        self.entries = {}
        
        row = 2
        for key, param in self.params.items():
            label = ttk.Label(main_frame, text=param['label'])
            label.grid(row=row, column=0, sticky=tk.W, pady=5)
            
            slider = ttk.Scale(main_frame, from_=param['min'], to=param['max'],
                             orient=tk.HORIZONTAL, length=220,
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
        
        self.fov_label = ttk.Label(results_frame, text="Field of View at Source: ", 
                                   font=('Arial', 11, 'bold'))
        self.fov_label.grid(row=0, column=0, pady=5, sticky=tk.W)
        
        self.limiting_edges_label = ttk.Label(results_frame, text="Limiting Edges: ")
        self.limiting_edges_label.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        self.angle_label = ttk.Label(results_frame, text="Maximum Acceptance Half-Angle: ")
        self.angle_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        self.detector_fov_label = ttk.Label(results_frame, text="FOV at Detector: ")
        self.detector_fov_label.grid(row=3, column=0, pady=5, sticky=tk.W)
        
        # View controls
        view_frame = ttk.LabelFrame(main_frame, text="3D View Controls", padding="10")
        view_frame.grid(row=row+1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Button(view_frame, text="Reset View", command=self.reset_view).grid(row=0, column=0, padx=5)
        ttk.Button(view_frame, text="Top View", command=self.top_view).grid(row=0, column=1, padx=5)
        ttk.Button(view_frame, text="Side View", command=self.side_view).grid(row=0, column=2, padx=5)
        ttk.Button(view_frame, text="Front View", command=self.front_view).grid(row=0, column=3, padx=5)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initial calculation
        self.calculate()
    
    def on_closing(self):
        self.plotter.close()
        self.root.destroy()
    
    def reset_view(self):
        self.plotter.view_isometric()
    
    def top_view(self):
        self.plotter.view_xy()
    
    def side_view(self):
        self.plotter.view_xz()
    
    def front_view(self):
        self.plotter.view_yz()
    
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
            messagebox.showinfo("Saved", f"Configuration '{name}' saved!")
    
    def update_config(self):
        name = self.config_var.get()
        if not name:
            messagebox.showwarning("Warning", "No configuration selected!")
            return
        if messagebox.askyesno("Confirm Update", f"Update configuration '{name}'?"):
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
    
    def delete_config(self):
        name = self.config_var.get()
        if not name:
            messagebox.showwarning("Warning", "No configuration selected!")
            return
        if messagebox.askyesno("Confirm Delete", f"Delete configuration '{name}'?"):
            del self.saved_configs[name]
            self.save_configs_to_file()
            self.update_config_dropdown()
            messagebox.showinfo("Deleted", f"Configuration '{name}' deleted!")
    
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
            val = max(min_val, min(max_val, val))
            
            self.params[key]['value'] = val
            self.sliders[key].set(val)
            self.value_labels[key].config(text=f"{val:.1f}")
            self.entry_vars[key].set(f"{val:.1f}")
            self.calculate()
        except ValueError:
            self.entry_vars[key].set(f"{self.params[key]['value']:.1f}")
    
    def calculate_ray_1d(self, pos_source, edge1_pos, edge1_coord, edge2_pos, edge2_coord):
        """Calculate ray in one dimension (x or y)"""
        if edge1_pos == edge2_pos:
            return None, None, None
        slope = (edge1_coord - edge2_coord) / (edge1_pos - edge2_pos)
        coord_source = edge1_coord + slope * (pos_source - edge1_pos)
        coord_detector = edge2_coord - slope * edge2_pos
        return coord_source, coord_detector, slope
    
    def check_ray_clears_aperture_1d(self, edge1_pos, edge1_coord, edge2_pos, edge2_coord,
                                      aperture_entrance_pos, aperture_exit_pos, half_width):
        """Check if ray clears square aperture in one dimension"""
        if edge1_pos == edge2_pos:
            return True
        slope = (edge1_coord - edge2_coord) / (edge1_pos - edge2_pos)
        
        coord_at_entrance = edge2_coord + slope * (aperture_entrance_pos - edge2_pos)
        if abs(coord_at_entrance) > half_width + 1e-9:
            return False
        
        coord_at_exit = edge2_coord + slope * (aperture_exit_pos - edge2_pos)
        if abs(coord_at_exit) > half_width + 1e-9:
            return False
        
        return True
    
    def calculate(self):
        d_detector_to_col1 = self.params['col1_to_detector']['value']
        l_col1 = self.params['col1_length']['value']
        hw_col1 = self.params['col1_half_width']['value']
        d_col1_to_col2 = self.params['col2_to_col1']['value']
        l_col2 = self.params['col2_length']['value']
        hw_col2 = self.params['col2_half_width']['value']
        d_col2_to_source = self.params['source_to_col2']['value']
        
        pos_col1_exit = d_detector_to_col1
        pos_col1_entrance = d_detector_to_col1 + l_col1
        pos_col2_exit = d_detector_to_col1 + l_col1 + d_col1_to_col2
        pos_col2_entrance = d_detector_to_col1 + l_col1 + d_col1_to_col2 + l_col2
        pos_source = pos_col2_entrance + d_col2_to_source
        
        # For square apertures, we analyze one dimension (symmetry means x and y are identical)
        # Edges are now the sides of the square aperture
        vertices = [
            ("Col2 entrance +edge", pos_col2_entrance, +hw_col2),
            ("Col2 entrance -edge", pos_col2_entrance, -hw_col2),
            ("Col2 exit +edge", pos_col2_exit, +hw_col2),
            ("Col2 exit -edge", pos_col2_exit, -hw_col2),
            ("Col1 entrance +edge", pos_col1_entrance, +hw_col1),
            ("Col1 entrance -edge", pos_col1_entrance, -hw_col1),
            ("Col1 exit +edge", pos_col1_exit, +hw_col1),
            ("Col1 exit -edge", pos_col1_exit, -hw_col1),
        ]
        
        max_fov = 0
        best_edge1 = None
        best_edge2 = None
        best_coord_detector = None
        
        for name1, pos1, coord1 in vertices:
            for name2, pos2, coord2 in vertices:
                if pos1 <= pos2:
                    continue
                
                coord_source, coord_det, slope = self.calculate_ray_1d(
                    pos_source, pos1, coord1, pos2, coord2)
                
                if coord_source is None:
                    continue
                
                # Check if ray clears both apertures
                passes_col2 = self.check_ray_clears_aperture_1d(
                    pos1, coord1, pos2, coord2, pos_col2_exit, pos_col2_entrance, hw_col2)
                passes_col1 = self.check_ray_clears_aperture_1d(
                    pos1, coord1, pos2, coord2, pos_col1_exit, pos_col1_entrance, hw_col1)
                
                if not passes_col2 or not passes_col1:
                    continue
                
                fov = abs(coord_source)
                if fov > max_fov:
                    max_fov = fov
                    best_edge1 = (name1, pos1, coord1)
                    best_edge2 = (name2, pos2, coord2)
                    best_coord_detector = coord_det
        
        if best_edge1 is None:
            self.fov_label.config(text="No valid ray path found!", foreground='red')
            self.limiting_edges_label.config(text="N/A")
            self.angle_label.config(text="N/A")
            self.detector_fov_label.config(text="N/A")
            return
        
        fov_half_width = max_fov
        fov_full_width = 2 * fov_half_width
        max_angle = math.atan(fov_half_width / pos_source)
        detector_fov = abs(best_coord_detector) * 2
        
        self.fov_label.config(
            text=f"Field of View at Source: {fov_full_width:.2f} × {fov_full_width:.2f} mm (half-width: {fov_half_width:.2f} mm)",
            foreground='black')
        self.limiting_edges_label.config(text=f"Limiting Edges: {best_edge1[0]} → {best_edge2[0]}")
        self.angle_label.config(text=f"Maximum Acceptance Half-Angle: {math.degrees(max_angle):.3f}°")
        self.detector_fov_label.config(text=f"FOV at Detector: {detector_fov:.2f} × {detector_fov:.2f} mm")
        
        self.draw_3d(fov_half_width, pos_col1_exit, pos_col1_entrance,
                     pos_col2_exit, pos_col2_entrance, pos_source,
                     l_col1, hw_col1, l_col2, hw_col2,
                     best_edge1, best_edge2, best_coord_detector)
    
    def create_hollow_box_collimator(self, center_z, length, aperture_half_width, wall_thickness=None):
        """Create a hollow rectangular collimator with square aperture"""
        if wall_thickness is None:
            wall_thickness = aperture_half_width * 0.8
        
        outer_half = aperture_half_width + wall_thickness
        
        # Outer box
        outer_box = pv.Box(bounds=(
            -outer_half, outer_half,
            -outer_half, outer_half,
            center_z - length/2, center_z + length/2
        ))
        
        # Inner box (aperture) - slightly longer to ensure clean boolean
        inner_box = pv.Box(bounds=(
            -aperture_half_width, aperture_half_width,
            -aperture_half_width, aperture_half_width,
            center_z - length/2 - 1, center_z + length/2 + 1
        ))
        
        # Create hollow collimator
        hollow = outer_box.boolean_difference(inner_box)
        return hollow, outer_half
    
    def draw_3d(self, fov_half_width, pos_c1_exit, pos_c1_entrance, pos_c2_exit, pos_c2_entrance,
                pos_source, l1, hw1, l2, hw2, edge1, edge2, coord_detector):
        self.plotter.clear()
        
        # Create collimator 1 (hollow box with square aperture)
        col1_center_z = (pos_c1_exit + pos_c1_entrance) / 2
        col1, outer1 = self.create_hollow_box_collimator(col1_center_z, l1, hw1)
        self.plotter.add_mesh(col1, color='steelblue', opacity=0.7, label='Collimator 1')
        
        # Create collimator 2 (hollow box with square aperture)
        col2_center_z = (pos_c2_exit + pos_c2_entrance) / 2
        col2, outer2 = self.create_hollow_box_collimator(col2_center_z, l2, hw2)
        self.plotter.add_mesh(col2, color='lightblue', opacity=0.7, label='Collimator 2')
        
        # Create detector (square)
        detector_size = max(hw1, hw2) * 3
        detector = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), 
                           i_size=detector_size*2, j_size=detector_size*2)
        self.plotter.add_mesh(detector, color='darkblue', opacity=0.8, label='Detector')
        
        # Create source FOV (square)
        source_plane = pv.Plane(center=(0, 0, pos_source), direction=(0, 0, 1),
                                i_size=fov_half_width*2, j_size=fov_half_width*2)
        self.plotter.add_mesh(source_plane, color='orange', opacity=0.8, label='Source FOV')
        
        # Draw aperture outlines
        self.draw_square_outline(pos_c1_exit, hw1, 'steelblue', 3)
        self.draw_square_outline(pos_c1_entrance, hw1, 'steelblue', 3)
        self.draw_square_outline(pos_c2_exit, hw2, 'cyan', 3)
        self.draw_square_outline(pos_c2_entrance, hw2, 'cyan', 3)
        
        # Draw rays from corners and edges
        if edge1 and edge2:
            name1, pos1, coord1 = edge1
            name2, pos2, coord2 = edge2
            is_crossing = (coord1 * coord2) < 0
            ray_color = 'purple' if is_crossing else 'green'
            
            # Draw rays from the 4 corners of the FOV square
            corners = [
                (fov_half_width, fov_half_width),
                (fov_half_width, -fov_half_width),
                (-fov_half_width, -fov_half_width),
                (-fov_half_width, fov_half_width),
            ]
            
            # Calculate corresponding points at edges and detector
            for cx, cy in corners:
                # Scale factors for x and y based on the 1D calculation
                scale = coord1 / fov_half_width if fov_half_width != 0 else 0
                
                e1_x = cx * scale
                e1_y = cy * scale
                
                scale2 = coord2 / coord1 if coord1 != 0 else 0
                e2_x = e1_x * scale2
                e2_y = e1_y * scale2
                
                det_scale = coord_detector / fov_half_width if fov_half_width != 0 else 0
                det_x = cx * det_scale
                det_y = cy * det_scale
                
                # Draw ray segments
                ray_points = np.array([
                    [cx, cy, pos_source],
                    [e1_x, e1_y, pos1],
                    [e2_x, e2_y, pos2],
                    [det_x, det_y, 0]
                ])
                ray = pv.Spline(ray_points, 50)
                self.plotter.add_mesh(ray, color=ray_color, line_width=3, opacity=0.9)
            
            # Draw rays from edge midpoints
            edge_mids = [
                (fov_half_width, 0),
                (-fov_half_width, 0),
                (0, fov_half_width),
                (0, -fov_half_width),
            ]
            
            for ex, ey in edge_mids:
                scale = coord1 / fov_half_width if fov_half_width != 0 else 0
                e1_x = ex * scale
                e1_y = ey * scale
                
                scale2 = coord2 / coord1 if coord1 != 0 else 0
                e2_x = e1_x * scale2
                e2_y = e1_y * scale2
                
                det_scale = coord_detector / fov_half_width if fov_half_width != 0 else 0
                det_x = ex * det_scale
                det_y = ey * det_scale
                
                ray_points = np.array([
                    [ex, ey, pos_source],
                    [e1_x, e1_y, pos1],
                    [e2_x, e2_y, pos2],
                    [det_x, det_y, 0]
                ])
                ray = pv.Spline(ray_points, 50)
                self.plotter.add_mesh(ray, color=ray_color, line_width=2, opacity=0.7)
            
            # Highlight limiting edges with red squares
            self.draw_square_outline(pos1, abs(coord1), 'red', 5)
            self.draw_square_outline(pos2, abs(coord2), 'red', 5)
        
        # Draw FOV outline at source
        self.draw_square_outline(pos_source, fov_half_width, 'orange', 4)
        
        # Draw detector FOV outline
        self.draw_square_outline(0, abs(coord_detector), 'yellow', 3)
        
        # Draw central axis
        axis = pv.Line((0, 0, 0), (0, 0, pos_source))
        self.plotter.add_mesh(axis, color='gray', line_width=1, style='wireframe')
        
        # Add labels
        self.plotter.add_point_labels(
            [(detector_size, 0, 0)], ['Detector'], font_size=12, text_color='darkblue')
        self.plotter.add_point_labels(
            [(outer1 + 5, 0, col1_center_z)], 
            [f'Col1\n{hw1*2:.1f}×{hw1*2:.1f}mm'], font_size=10, text_color='steelblue')
        self.plotter.add_point_labels(
            [(outer2 + 5, 0, col2_center_z)], 
            [f'Col2\n{hw2*2:.1f}×{hw2*2:.1f}mm'], font_size=10, text_color='lightblue')
        self.plotter.add_point_labels(
            [(fov_half_width + 10, 0, pos_source)], 
            [f'Source FOV\n{fov_half_width*2:.1f}×{fov_half_width*2:.1f}mm'], 
            font_size=12, text_color='darkorange')
    
    def draw_square_outline(self, z_pos, half_width, color, line_width):
        """Draw a square outline at given z position"""
        hw = half_width
        corners = np.array([
            [hw, hw, z_pos],
            [hw, -hw, z_pos],
            [-hw, -hw, z_pos],
            [-hw, hw, z_pos],
            [hw, hw, z_pos]  # Close the square
        ])
        square = pv.lines_from_points(corners)
        self.plotter.add_mesh(square, color=color, line_width=line_width)


if __name__ == "__main__":
    root = tk.Tk()
    app = XRayCollimatorGUI(root)
    root.mainloop()
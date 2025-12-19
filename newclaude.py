import tkinter as tk
from tkinter import ttk
import math

class XRayCollimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Beam Collimator Calculator")
        self.root.geometry("900x750")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="X-Ray Detector Field of View Calculator", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Parameters
        self.params = {
            'source_to_col1': {'value': 2141, 'min': 10, 'max': 2500, 'label': 'X-ray Source to Collimator 2 (mm)'},
            'col1_length': {'value': 128.6, 'min': 10, 'max': 200, 'label': 'Collimator 2 Length (mm)'},
            'col1_radius': {'value': 1.5, 'min': 0.5, 'max': 50, 'label': 'Collimator 2 Radius (mm)'},
            'col1_to_col2': {'value': 130.4, 'min': 10, 'max': 1000, 'label': 'Distance Col2 to Col1 (mm)'},
            'col2_length': {'value': 103, 'min': 10, 'max': 200, 'label': 'Collimator 1 Length (mm)'},
            'col2_radius': {'value': 0.5, 'min': 0.5, 'max': 50, 'label': 'Collimator 1 Radius (mm)'},
            'col2_to_target': {'value': 67, 'min': 0.000000000000001, 'max': 500, 'label': 'Distance Col1 to Detector (mm)'},
        }
        
        self.sliders = {}
        self.value_labels = {}
        self.entry_vars = {}
        self.entries = {}
        
        # Create sliders with entry fields
        row = 1
        for key, param in self.params.items():
            # Label
            label = ttk.Label(main_frame, text=param['label'])
            label.grid(row=row, column=0, sticky=tk.W, pady=5)
            
            # Slider
            slider = ttk.Scale(main_frame, from_=param['min'], to=param['max'],
                             orient=tk.HORIZONTAL, length=300,
                             command=lambda v, k=key: self.update_from_slider(k, v))
            slider.set(param['value'])
            slider.grid(row=row, column=1, padx=10, pady=5)
            self.sliders[key] = slider
            
            # Value display label
            value_label = ttk.Label(main_frame, text=f"{param['value']:.1f}")
            value_label.grid(row=row, column=2, sticky=tk.W, pady=5, padx=5)
            self.value_labels[key] = value_label
            
            # Entry field for manual input
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
        results_frame.grid(row=row, column=0, columnspan=4, pady=20, sticky=(tk.W, tk.E))
        
        # Result labels
        self.beam_radius_label = ttk.Label(results_frame, text="Field of View Radius at Source: ", 
                                          font=('Arial', 12, 'bold'))
        self.beam_radius_label.grid(row=0, column=0, pady=5, sticky=tk.W)
        
        self.limiting_edges_label = ttk.Label(results_frame, text="Limiting Edges: ")
        self.limiting_edges_label.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        self.angle_label = ttk.Label(results_frame, text="Maximum Acceptance Angle: ")
        self.angle_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # Diagram frame
        diagram_frame = ttk.LabelFrame(main_frame, text="Ray Path Through Collimators (Not to Scale)", 
                                      padding="10")
        diagram_frame.grid(row=row+1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        self.canvas = tk.Canvas(diagram_frame, width=850, height=200, bg='white')
        self.canvas.grid(row=0, column=0)
        
        # Initial calculation
        self.calculate()
    
    def update_from_slider(self, key, value):
        """Update from slider movement"""
        val = float(value)
        self.params[key]['value'] = val
        self.value_labels[key].config(text=f"{val:.1f}")
        self.entry_vars[key].set(f"{val:.1f}")
        self.calculate()
    
    def update_from_entry(self, key):
        """Update from manual entry"""
        try:
            val = float(self.entry_vars[key].get())
            
            # Clamp value to min/max
            min_val = self.params[key]['min']
            max_val = self.params[key]['max']
            
            if val < min_val:
                val = min_val
            elif val > max_val:
                val = max_val
            
            # Update all displays
            self.params[key]['value'] = val
            self.sliders[key].set(val)
            self.value_labels[key].config(text=f"{val:.1f}")
            self.entry_vars[key].set(f"{val:.1f}")
            self.calculate()
            
        except ValueError:
            # If invalid input, reset to current value
            val = self.params[key]['value']
            self.entry_vars[key].set(f"{val:.1f}")
    
    def calculate_ray(self, pos_source, edge1_pos, edge1_r, edge2_pos, edge2_r):
        """
        Calculate FOV for a ray passing through edge1 and edge2
        Returns (fov_radius, r_at_detector, slope)
        """
        if edge1_pos == edge2_pos:
            return None, None, None
        
        slope = (edge1_r - edge2_r) / (edge1_pos - edge2_pos)
        
        # Extend to source
        r_source = edge1_r + slope * (pos_source - edge1_pos)
        
        # Extend to detector (at position 0)
        r_detector = edge2_r - slope * edge2_pos
        
        return r_source, r_detector, slope
    
    def check_ray_clears_collimator(self, edge1_pos, edge1_r, edge2_pos, edge2_r,
                                     col_entrance_pos, col_exit_pos, col_radius):
        """Check if ray from edge1 to edge2 passes through a collimator without hitting body"""
        if edge1_pos == edge2_pos:
            return True
        
        slope = (edge1_r - edge2_r) / (edge1_pos - edge2_pos)
        
        # Check at entrance
        r_at_entrance = edge2_r + slope * (col_entrance_pos - edge2_pos)
        if abs(r_at_entrance) > col_radius + 1e-9:
            return False
        
        # Check at exit
        r_at_exit = edge2_r + slope * (col_exit_pos - edge2_pos)
        if abs(r_at_exit) > col_radius + 1e-9:
            return False
        
        return True
    
    def calculate(self):
        # Get parameters
        d_detector_to_col1 = self.params['col2_to_target']['value']
        l_col1 = self.params['col2_length']['value']
        r_col1 = self.params['col2_radius']['value']
        d_col1_to_col2 = self.params['col1_to_col2']['value']
        l_col2 = self.params['col1_length']['value']
        r_col2 = self.params['col1_radius']['value']
        d_col2_to_source = self.params['source_to_col1']['value']
        
        # Calculate edge positions from detector (at position 0)
        pos_col1_exit = d_detector_to_col1
        pos_col1_entrance = d_detector_to_col1 + l_col1
        pos_col2_exit = d_detector_to_col1 + l_col1 + d_col1_to_col2
        pos_col2_entrance = d_detector_to_col1 + l_col1 + d_col1_to_col2 + l_col2
        pos_source = pos_col2_entrance + d_col2_to_source
        
        # Define all 8 vertices: (name, position, radius with sign)
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
        
        # Find the maximum valid FOV (crossing rays give maximum angle)
        max_fov = 0
        best_edge1 = None
        best_edge2 = None
        best_slope = None
        best_r_detector = None
        
        # Try all combinations of edges
        for i, (name1, pos1, r1) in enumerate(vertices):
            for j, (name2, pos2, r2) in enumerate(vertices):
                # Edge1 must be closer to source than edge2
                if pos1 <= pos2:
                    continue
                
                # Calculate ray
                r_source, r_det, slope = self.calculate_ray(pos_source, pos1, r1, pos2, r2)
                
                if r_source is None:
                    continue
                
                # Check if ray passes cleanly through Col2
                passes_col2 = self.check_ray_clears_collimator(
                    pos1, r1, pos2, r2,
                    pos_col2_exit, pos_col2_entrance, r_col2
                )
                
                # Check if ray passes cleanly through Col1
                passes_col1 = self.check_ray_clears_collimator(
                    pos1, r1, pos2, r2,
                    pos_col1_exit, pos_col1_entrance, r_col1
                )
                
                if not passes_col2 or not passes_col1:
                    continue
                
                # Valid ray - check if it gives larger FOV
                fov = abs(r_source)
                if fov > max_fov:
                    max_fov = fov
                    best_edge1 = (name1, pos1, r1)
                    best_edge2 = (name2, pos2, r2)
                    best_slope = slope
                    best_r_detector = r_det
        
        # Update results
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
        
        # Draw diagram
        self.draw_diagram(field_of_view_radius, 
                         pos_col1_exit, pos_col1_entrance,
                         pos_col2_exit, pos_col2_entrance, pos_source,
                         l_col1, r_col1, l_col2, r_col2, 
                         best_edge1, best_edge2, best_slope, best_r_detector)
    
    def draw_diagram(self, fov_radius, 
                    pos_c1_exit, pos_c1_entrance,
                    pos_c2_exit, pos_c2_entrance, pos_source,
                    l1, r1, l2, r2, 
                    edge1, edge2, slope, r_detector):
        self.canvas.delete("all")
        
        # Scale factor
        scale = 800 / pos_source
        y_center = 100
        x_offset = 25
        
        # Draw X-ray source (on the right)
        x_source = x_offset + pos_source * scale
        self.canvas.create_rectangle(x_source-2, y_center-60, x_source+2, y_center+60,
                                     fill='orange', outline='red', width=2)
        self.canvas.create_text(x_source+35, y_center-70, text="X-ray Source\n(uniform)", 
                               font=('Arial', 9, 'bold'), fill='red')
        
        # Draw field of view at source
        y_top_source = y_center - fov_radius * scale
        y_bottom_source = y_center + fov_radius * scale
        self.canvas.create_line(x_source-10, y_top_source, x_source-10, y_bottom_source, 
                               fill='red', width=4)
        self.canvas.create_text(x_source-50, (y_center+y_top_source)/2, 
                               text=f"FOV\n{fov_radius:.1f}mm", font=('Arial', 8, 'bold'), fill='red')
        
        # Draw detector (on the left)
        self.canvas.create_rectangle(x_offset-2, y_center-30, x_offset+2, y_center+30,
                                     fill='darkblue', outline='blue', width=2)
        self.canvas.create_text(x_offset-40, y_center, text="X-ray\nDetector", 
                               font=('Arial', 9, 'bold'), fill='blue')
        
        # Draw Collimator 1 (nearest detector)
        x_c1_exit = x_offset + pos_c1_exit * scale
        x_c1_entrance = x_offset + pos_c1_entrance * scale
        
        self.canvas.create_rectangle(x_c1_exit, y_center-r1*scale-20, 
                                     x_c1_entrance, y_center-r1*scale,
                                     fill='#ADD8E6', outline='blue', width=2)
        self.canvas.create_rectangle(x_c1_exit, y_center+r1*scale, 
                                     x_c1_entrance, y_center+r1*scale+20,
                                     fill='#ADD8E6', outline='blue', width=2)
        self.canvas.create_text((x_c1_exit+x_c1_entrance)/2, y_center+r1*scale+35, 
                               text=f"Col1\n(r={r1:.1f}mm)", font=('Arial', 8, 'bold'))
        
        # Draw Collimator 2 (nearest source)
        x_c2_exit = x_offset + pos_c2_exit * scale
        x_c2_entrance = x_offset + pos_c2_entrance * scale
        
        self.canvas.create_rectangle(x_c2_exit, y_center-r2*scale-20, 
                                     x_c2_entrance, y_center-r2*scale,
                                     fill='#ADD8E6', outline='blue', width=2)
        self.canvas.create_rectangle(x_c2_exit, y_center+r2*scale, 
                                     x_c2_entrance, y_center+r2*scale+20,
                                     fill='#ADD8E6', outline='blue', width=2)
        self.canvas.create_text((x_c2_exit+x_c2_entrance)/2, y_center+r2*scale+35, 
                               text=f"Col2\n(r={r2:.1f}mm)", font=('Arial', 8, 'bold'))
        
        if edge1 is None or edge2 is None:
            return
        
        # Extract edge information
        name1, pos1, r1_signed = edge1
        name2, pos2, r2_signed = edge2
        
        x_edge1 = x_offset + pos1 * scale
        x_edge2 = x_offset + pos2 * scale
        
        y_edge1 = y_center - r1_signed * scale
        y_edge2 = y_center - r2_signed * scale
        
        # Determine if rays cross (opposite signs)
        is_crossing = (r1_signed * r2_signed) < 0
        ray_color = 'purple' if is_crossing else 'green'
        
        # Draw top ray
        self.canvas.create_line(x_source, y_top_source, x_edge1, y_edge1,
                               fill=ray_color, width=2, arrow=tk.LAST)
        self.canvas.create_line(x_edge1, y_edge1, x_edge2, y_edge2,
                               fill=ray_color, width=2)
        y_det_top = y_center - r_detector * scale
        self.canvas.create_line(x_edge2, y_edge2, x_offset, y_det_top,
                               fill=ray_color, width=2, arrow=tk.LAST)
        
        # Draw bottom ray (mirror)
        y_edge1_mir = y_center + r1_signed * scale
        y_edge2_mir = y_center + r2_signed * scale
        y_det_bot = y_center + r_detector * scale
        
        self.canvas.create_line(x_source, y_bottom_source, x_edge1, y_edge1_mir,
                               fill=ray_color, width=2, arrow=tk.LAST)
        self.canvas.create_line(x_edge1, y_edge1_mir, x_edge2, y_edge2_mir,
                               fill=ray_color, width=2)
        self.canvas.create_line(x_edge2, y_edge2_mir, x_offset, y_det_bot,
                               fill=ray_color, width=2, arrow=tk.LAST)
        
        # Mark the critical edges with dots
        dot_color = 'purple' if is_crossing else 'red'
        
        self.canvas.create_oval(x_edge1-4, y_edge1-4, x_edge1+4, y_edge1+4,
                               fill=dot_color, outline='darkred', width=2)
        self.canvas.create_oval(x_edge2-4, y_edge2-4, x_edge2+4, y_edge2+4,
                               fill=dot_color, outline='darkred', width=2)
        self.canvas.create_oval(x_edge1-4, y_edge1_mir-4, x_edge1+4, y_edge1_mir+4,
                               fill=dot_color, outline='darkred', width=2)
        self.canvas.create_oval(x_edge2-4, y_edge2_mir-4, x_edge2+4, y_edge2_mir+4,
                               fill=dot_color, outline='darkred', width=2)
        
        # Draw centerline
        self.canvas.create_line(x_offset, y_center, x_source, y_center,
                               fill='gray', width=1, dash=(5, 5))

if __name__ == "__main__":
    root = tk.Tk()
    app = XRayCollimatorGUI(root)
    root.mainloop()
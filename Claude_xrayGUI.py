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
        
        # Ray mode selection
        self.ray_mode = tk.StringVar(value="col2_col1")
        mode_frame = ttk.LabelFrame(main_frame, text="Ray Path Mode", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(mode_frame, text="Col2 entrance → Col1 exit (cross-collimator)", 
                       variable=self.ray_mode, value="col2_col1",
                       command=self.calculate).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="Col1 entrance → Col1 exit (single collimator)", 
                       variable=self.ray_mode, value="col1_only",
                       command=self.calculate).grid(row=0, column=1, sticky=tk.W, padx=5)
        
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
        
        self.limiting_col_label = ttk.Label(results_frame, text="Limiting Collimator: ")
        self.limiting_col_label.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        self.geometry_label = ttk.Label(results_frame, text="Geometry: ")
        self.geometry_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
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
        # Col1 (nearest detector):
        pos_col1_exit_detector = d_detector_to_col1  # Exit edge facing detector
        pos_col1_entrance = d_detector_to_col1 + l_col1  # Entrance edge facing col2
        
        # Col2 (nearest source):
        pos_col2_exit = d_detector_to_col1 + l_col1 + d_col1_to_col2  # Exit edge facing col1
        pos_col2_entrance_source = d_detector_to_col1 + l_col1 + d_col1_to_col2 + l_col2  # Entrance edge facing source
        
        # Source position
        pos_source = pos_col2_entrance_source + d_col2_to_source
        
        mode = self.ray_mode.get()
        
        if mode == "col2_col1":
            # Ray passes through col2 entrance and col1 exit
            edge1_pos = pos_col2_entrance_source
            edge1_radius = r_col2
            edge1_name = "Col2 entrance"
            
            edge2_pos = pos_col1_exit_detector
            edge2_radius = r_col1
            edge2_name = "Col1 exit"
            
            limiting_col = "Collimator 2 (near source)" if abs(r_col2 / pos_col2_entrance_source) > abs(r_col1 / pos_col1_exit_detector) else "Collimator 1 (near detector)"
            
        else:  # col1_only
            # Ray passes through both edges of col1
            edge1_pos = pos_col1_entrance
            edge1_radius = r_col1
            edge1_name = "Col1 entrance"
            
            edge2_pos = pos_col1_exit_detector
            edge2_radius = r_col1
            edge2_name = "Col1 exit"
            
            limiting_col = "Collimator 1 (both edges)"
        
        # Distance between the two critical edges
        d_between_edges = edge1_pos - edge2_pos
        
        # Distance from edge1 to source
        d_edge1_to_source = pos_source - edge1_pos
        
        # Distance from detector to edge2
        d_detector_to_edge2 = edge2_pos
        
        # The slope of the extreme ray through both aperture edges
        slope = (edge1_radius - edge2_radius) / d_between_edges
        
        # Extend this ray backward from edge1 to source
        r_source = edge1_radius + slope * d_edge1_to_source
        
        # Extend ray forward from edge2 to detector  
        r_at_detector = edge2_radius - slope * d_detector_to_edge2
        
        # The field of view at the source
        field_of_view_radius = abs(r_source)
        
        # Update results
        self.beam_radius_label.config(
            text=f"Field of View Radius at X-ray Source: {field_of_view_radius:.2f} mm"
        )
        self.limiting_col_label.config(
            text=f"Limiting Aperture: {limiting_col}"
        )
        self.geometry_label.config(
            text=f"Ray passes through {edge1_name} (r={edge1_radius:.1f}mm) and {edge2_name} (r={edge2_radius:.1f}mm)"
        )
        
        # Draw diagram
        self.draw_diagram(field_of_view_radius, limiting_col, 
                         pos_col1_exit_detector, pos_col1_entrance,
                         pos_col2_exit, pos_col2_entrance_source, pos_source,
                         l_col1, r_col1, l_col2, r_col2, r_at_detector, slope,
                         edge1_pos, edge1_radius, edge2_pos, edge2_radius, mode)
    
    def draw_diagram(self, fov_radius, limiting_col, 
                    pos_c1_exit, pos_c1_entrance,
                    pos_c2_exit, pos_c2_entrance, pos_source,
                    l1, r1, l2, r2, r_at_detector, slope,
                    edge1_pos, edge1_radius, edge2_pos, edge2_radius, mode):
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
        col1_color = 'blue' if "Collimator 1" in limiting_col else 'gray'
        col1_fill = '#ADD8E6' if "Collimator 1" in limiting_col else '#D3D3D3'
        
        # Top block
        self.canvas.create_rectangle(x_c1_exit, y_center-r1*scale-20, 
                                     x_c1_entrance, y_center-r1*scale,
                                     fill=col1_fill, outline=col1_color, width=2)
        # Bottom block
        self.canvas.create_rectangle(x_c1_exit, y_center+r1*scale, 
                                     x_c1_entrance, y_center+r1*scale+20,
                                     fill=col1_fill, outline=col1_color, width=2)
        
        # Label
        self.canvas.create_text((x_c1_exit+x_c1_entrance)/2, y_center+r1*scale+35, 
                               text=f"Col1\n(r={r1:.1f}mm)", font=('Arial', 8, 'bold'))
        
        # Draw Collimator 2 (nearest source)
        x_c2_exit = x_offset + pos_c2_exit * scale
        x_c2_entrance = x_offset + pos_c2_entrance * scale
        col2_color = 'blue' if "Collimator 2" in limiting_col else 'gray'
        col2_fill = '#ADD8E6' if "Collimator 2" in limiting_col else '#D3D3D3'
        
        # Top block
        self.canvas.create_rectangle(x_c2_exit, y_center-r2*scale-20, 
                                     x_c2_entrance, y_center-r2*scale,
                                     fill=col2_fill, outline=col2_color, width=2)
        # Bottom block
        self.canvas.create_rectangle(x_c2_exit, y_center+r2*scale, 
                                     x_c2_entrance, y_center+r2*scale+20,
                                     fill=col2_fill, outline=col2_color, width=2)
        
        # Label
        self.canvas.create_text((x_c2_exit+x_c2_entrance)/2, y_center+r2*scale+35, 
                               text=f"Col2\n(r={r2:.1f}mm)", font=('Arial', 8, 'bold'))
        
        # Calculate ray positions at critical edges
        x_edge1 = x_offset + edge1_pos * scale
        x_edge2 = x_offset + edge2_pos * scale
        
        y_top_at_detector = y_center - abs(r_at_detector) * scale
        y_top_at_edge2 = y_center - edge2_radius * scale
        y_top_at_edge1 = y_center - edge1_radius * scale
        
        y_bottom_at_detector = y_center + abs(r_at_detector) * scale
        y_bottom_at_edge2 = y_center + edge2_radius * scale
        y_bottom_at_edge1 = y_center + edge1_radius * scale
        
        # Choose colors based on mode
        if mode == "col2_col1":
            ray_color = 'green'
        else:  # col1_only
            ray_color = 'purple'
        
        # Draw the extreme ray path (top edge)
        self.canvas.create_line(x_source, y_top_source, x_edge1, y_top_at_edge1,
                               fill=ray_color, width=2, arrow=tk.LAST)
        self.canvas.create_line(x_edge1, y_top_at_edge1, x_edge2, y_top_at_edge2,
                               fill=ray_color, width=2)
        self.canvas.create_line(x_edge2, y_top_at_edge2, x_offset, y_top_at_detector,
                               fill=ray_color, width=2, arrow=tk.LAST)
        
        # Draw the extreme ray path (bottom edge)
        self.canvas.create_line(x_source, y_bottom_source, x_edge1, y_bottom_at_edge1,
                               fill=ray_color, width=2, arrow=tk.LAST)
        self.canvas.create_line(x_edge1, y_bottom_at_edge1, x_edge2, y_bottom_at_edge2,
                               fill=ray_color, width=2)
        self.canvas.create_line(x_edge2, y_bottom_at_edge2, x_offset, y_bottom_at_detector,
                               fill=ray_color, width=2, arrow=tk.LAST)
        
        # Mark the critical edges with dots
        dot_color = 'red' if mode == "col2_col1" else 'purple'
        
        # Edge 1
        self.canvas.create_oval(x_edge1-3, y_top_at_edge1-3, 
                               x_edge1+3, y_top_at_edge1+3,
                               fill=dot_color, outline='darkred')
        self.canvas.create_oval(x_edge1-3, y_bottom_at_edge1-3, 
                               x_edge1+3, y_bottom_at_edge1+3,
                               fill=dot_color, outline='darkred')
        
        # Edge 2
        self.canvas.create_oval(x_edge2-3, y_top_at_edge2-3, 
                               x_edge2+3, y_top_at_edge2+3,
                               fill=dot_color, outline='darkred')
        self.canvas.create_oval(x_edge2-3, y_bottom_at_edge2-3, 
                               x_edge2+3, y_bottom_at_edge2+3,
                               fill=dot_color, outline='darkred')
        
        # Draw centerline
        self.canvas.create_line(x_offset, y_center, x_source, y_center,
                               fill='gray', width=1, dash=(5, 5))

if __name__ == "__main__":
    root = tk.Tk()
    app = XRayCollimatorGUI(root)
    root.mainloop()
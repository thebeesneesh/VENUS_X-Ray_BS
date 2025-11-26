#!/usr/bin/env python3
"""
Ray-tracing GUI with:
 - two finite cylindrical collimators (entry plane, exit plane, inner radius)
 - proper wall-intersection physics (finite cylinder side + endcaps)
 - sliders + numeric entry controls
 - embedded matplotlib visualization:
     * side-view (x vs z) showing ray paths, collimator apertures, blocked rays
     * aperture inset (circles at entry/exit)
     * detector 2D heatmap (beam spot distribution)
     * interactive 3D view (rotatable) showing cylinders, rays, detector plane
 - option to save passing rays to CSV
"""

import math, random, csv
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

# -------------------------
# Geometry & Ray Tracing
# -------------------------

class Collimator:
    def __init__(self, z_entry, length, radius):
        self.z_entry = float(z_entry)
        self.length = float(length)
        self.z_exit = float(z_entry) + float(length)
        self.radius = float(radius)

    def __repr__(self):
        return f"Collimator(z_entry={self.z_entry}, z_exit={self.z_exit}, radius={self.radius})"


def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def intersect_plane_z(p0, v, z_plane, eps=1e-12):
    p0 = np.array(p0, dtype=float)
    v = np.array(v, dtype=float)
    vz = v[2]
    if abs(vz) < eps:
        return None, None
    t = (z_plane - p0[2]) / vz
    if t <= 0:
        return None, None
    pt = p0 + t * v
    return t, pt


def intersect_cylinder_side(p0, v, radius, z1, z2, eps=1e-12):
    p0 = np.array(p0, dtype=float)
    v = np.array(v, dtype=float)
    a = v[0]**2 + v[1]**2
    b = 2*(p0[0]*v[0] + p0[1]*v[1])
    c = p0[0]**2 + p0[1]**2 - radius**2
    ts = []
    if abs(a) < eps:
        return []
    disc = b*b - 4*a*c
    if disc < 0:
        return []
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    for t in (t1, t2):
        if t <= eps:
            continue
        z = p0[2] + t * v[2]
        if (z >= min(z1, z2) - 1e-9) and (z <= max(z1, z2) + 1e-9):
            ts.append(t)
    ts.sort()
    return ts


def first_hit_type_and_point(p0, v, col: Collimator):
    hits = []
    t_e, pt_e = intersect_plane_z(p0, v, col.z_entry)
    if t_e is not None:
        hits.append(("entry", t_e, pt_e))
    t_x, pt_x = intersect_plane_z(p0, v, col.z_exit)
    if t_x is not None:
        hits.append(("exit", t_x, pt_x))
    side_ts = intersect_cylinder_side(p0, v, col.radius, col.z_entry, col.z_exit)
    for t in side_ts:
        pt = p0 + t * v
        hits.append(("side", t, pt))
    if not hits:
        return None, None, None
    hits = [h for h in hits if h[1] > 0]
    if not hits:
        return None, None, None
    hits.sort(key=lambda x: x[1])
    return hits[0]


def ray_passes_collimator(p0, v, col: Collimator):
    hit1_type, t1, p1 = first_hit_type_and_point(p0, v, col)
    if hit1_type is None:
        return False, "no_intersection", {"first": None}
    if hit1_type == "side":
        return False, "blocked_by_side_first", {"first": (hit1_type, t1, p1)}
    if hit1_type == "entry":
        if (p1[0]**2 + p1[1]**2) > col.radius**2 + 1e-9:
            return False, "miss_entry_aperture", {"first": (hit1_type, t1, p1)}
        cand = []
        t_exit, p_exit = intersect_plane_z(p0, v, col.z_exit)
        if t_exit is not None and t_exit > t1 + 1e-12:
            cand.append(("exit", t_exit, p_exit))
        side_ts = intersect_cylinder_side(p0, v, col.radius, col.z_entry, col.z_exit)
        for t in side_ts:
            if t > t1 + 1e-12:
                cand.append(("side", t, p0 + t*v))
        if not cand:
            return False, "no_exit_after_entry", {"first": (hit1_type, t1, p1)}
        cand.sort(key=lambda x: x[1])
        next_type, t2, p2 = cand[0]
        if next_type == "side":
            return False, "hit_side_after_entry", {"first": (hit1_type, t1, p1), "next": (next_type, t2, p2)}
        if (p2[0]**2 + p2[1]**2) > col.radius**2 + 1e-9:
            return False, "exit_outside_aperture", {"first": (hit1_type, t1, p1), "next": (next_type, t2, p2)}
        return True, "passed_entry_exit", {"entry": (t1, p1), "exit": (t2, p2)}
    if hit1_type == "exit":
        side_ts = intersect_cylinder_side(p0, v, col.radius, col.z_entry, col.z_exit)
        for t in side_ts:
            if t < t1 - 1e-12:
                return False, "side_before_exit", {"first": (hit1_type, t1, p1)}
        if (p1[0]**2 + p1[1]**2) > col.radius**2 + 1e-9:
            return False, "exit_outside_aperture", {"first": (hit1_type, t1, p1)}
        return True, "passed_exit_only", {"exit": (t1, p1)}
    return False, "unknown_case", {"first": (hit1_type, t1, p1)}


def sample_direction_in_cone(axis=np.array([0,0,1.0]), max_angle_rad=math.radians(1.0)):
    u = random.random()
    cos_max = math.cos(max_angle_rad)
    cos_theta = u * (1.0 - cos_max) + cos_max
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi = random.random() * 2.0 * math.pi
    dir_local = np.array([sin_theta*math.cos(phi), sin_theta*math.sin(phi), cos_theta])
    axis = normalize(axis)
    z = np.array([0.,0.,1.])
    if np.allclose(axis, z):
        return dir_local
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    c = np.dot(z, axis)
    if s < 1e-12:
        return dir_local if c > 0 else -dir_local
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R = np.eye(3) + vx + vx.dot(vx)*((1-c)/(s*s))
    return R.dot(dir_local)


def monte_carlo_trace(n_rays, source_pos, source_radius, cone_deg, col1, col2, detector_z):
    results = []
    axis = np.array([0.,0.,1.])
    cone_rad = math.radians(cone_deg)
    for i in range(n_rays):
        r = math.sqrt(random.random()) * source_radius
        theta = random.random() * 2.0 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        origin = np.array([x + source_pos[0], y + source_pos[1], source_pos[2]], dtype=float)
        direction = sample_direction_in_cone(axis, cone_rad)
        direction = normalize(direction)
        c1_pass, c1_reason, c1_det = ray_passes_collimator(origin, direction, col1)
        c2_pass, c2_reason, c2_det = ray_passes_collimator(origin, direction, col2)
        passed_both = c1_pass and c2_pass
        det_pt = None
        if abs(direction[2]) > 1e-12:
            t_det = (detector_z - origin[2]) / direction[2]
            if t_det > 0:
                det_pt = origin + t_det * direction
        results.append({
            "origin": origin,
            "dir": direction,
            "col1_pass": c1_pass, "col1_reason": c1_reason,
            "col2_pass": c2_pass, "col2_reason": c2_reason,
            "passed_both": passed_both,
            "det_pt": det_pt
        })
    pass_frac = sum(1 for r in results if r["passed_both"]) / max(1, n_rays)
    return results, {"pass_fraction": pass_frac, "n_rays": n_rays, "passed": int(pass_frac * n_rays)}


# -------------------------
# GUI
# -------------------------

class RayTraceGUIApp:
    def __init__(self, master):
        self.master = master
        master.title("Ray Tracing Through Two Collimators — 3D View Added")

        control_frame = ttk.Frame(master)
        control_frame.grid(row=0, column=0, sticky="nw", padx=6, pady=6)

        def make_param(parent, label, default, row, from_=None, to=None):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=default)
            ent = ttk.Entry(parent, width=10, textvariable=var)
            ent.grid(row=row, column=1, padx=4)
            if from_ is not None:
                slider = ttk.Scale(parent, from_=from_, to=to, value=default, command=lambda v, var=var: var.set(float(v)))
                slider.grid(row=row, column=2, padx=6)
                def on_entry_change(*args, var=var, slider=slider):
                    try:
                        slider.set(var.get())
                    except Exception:
                        pass
                var.trace_add("write", on_entry_change)
            return var

        self.n_rays_var = tk.IntVar(value=4000)
        ttk.Label(control_frame, text="Number of rays:").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, width=10, textvariable=self.n_rays_var).grid(row=0, column=1, padx=4)

        self.source_x_var = make_param(control_frame, "Source X (mm)", 0.0, 1, from_=-10, to=10)
        self.source_y_var = make_param(control_frame, "Source Y (mm)", 0.0, 2, from_=-10, to=10)
        self.source_z_var = make_param(control_frame, "Source Z (mm)", -2590.0, 3, from_=-5000, to=0)
        self.source_radius_var = make_param(control_frame, "Source radius (mm)", 0.5, 4, from_=0.0, to=5.0)
        self.cone_deg_var = make_param(control_frame, "Cone half-angle (deg)", 0.5, 5, from_=0.0, to=10.0)

        ttk.Label(control_frame, text="--- Collimator 1 ---").grid(row=6, column=0, sticky="w", pady=(6,0))
        self.c1_z_var = make_param(control_frame, "C1 entry z (mm)", 0.0, 7, from_=-1000, to=2000)
        self.c1_len_var = make_param(control_frame, "C1 length (mm)", 128.6, 8, from_=1, to=1000)
        self.c1_rad_var = make_param(control_frame, "C1 radius (mm)", 0.5, 9, from_=0.01, to=20.0)

        ttk.Label(control_frame, text="--- Collimator 2 ---").grid(row=10, column=0, sticky="w", pady=(6,0))
        self.c2_z_var = make_param(control_frame, "C2 entry z (mm)", 500.0, 11, from_=-1000, to=5000)
        self.c2_len_var = make_param(control_frame, "C2 length (mm)", 128.6, 12, from_=1, to=1000)
        self.c2_rad_var = make_param(control_frame, "C2 radius (mm)", 0.5, 13, from_=0.01, to=20.0)

        self.det_z_var = make_param(control_frame, "Detector z (mm)", 2590.0, 14, from_=-5000, to=5000)

        ttk.Button(control_frame, text="Run", command=self.run_simulation).grid(row=15, column=0, pady=8)
        ttk.Button(control_frame, text="Save Passing Rays CSV", command=self.save_csv).grid(row=15, column=1, pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue").grid(row=16, column=0, columnspan=3, sticky="w", pady=(4,0))

        plot_frame = ttk.Frame(master)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        master.columnconfigure(1, weight=1)
        master.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(12,8))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[1,1.2])
        self.ax_side = self.fig.add_subplot(gs[0,0])
        self.ax_heat = self.fig.add_subplot(gs[0,1])
        self.ax_3d = self.fig.add_subplot(gs[1,:], projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.last_results = None
        self.last_stats = None

    def run_simulation(self):
        try:
            n = int(self.n_rays_var.get())
            sx = float(self.source_x_var.get())
            sy = float(self.source_y_var.get())
            sz = float(self.source_z_var.get())
            srad = float(self.source_radius_var.get())
            cone = float(self.cone_deg_var.get())
            c1 = Collimator(float(self.c1_z_var.get()), float(self.c1_len_var.get()), float(self.c1_rad_var.get()))
            c2 = Collimator(float(self.c2_z_var.get()), float(self.c2_len_var.get()), float(self.c2_rad_var.get()))
            dz = float(self.det_z_var.get())
        except Exception as e:
            messagebox.showerror("Input error", f"Invalid input: {e}")
            return

        self.status_var.set("Running simulation...")
        self.master.update_idletasks()

        results, stats = monte_carlo_trace(n, (sx, sy, sz), srad, cone, c1, c2, dz)
        self.last_results = results
        self.last_stats = stats

        self.status_var.set(f"Done — {stats['passed']} passed / {stats['n_rays']} rays ({stats['pass_fraction']*100:.3f}%)")
        self._update_plots(results, c1, c2, dz)

    def _plot_cylinder_3d(self, ax, col: Collimator, color='C0', alpha=0.25, res_theta=40, res_z=10):
        theta = np.linspace(0, 2*np.pi, res_theta)
        z = np.linspace(col.z_entry, col.z_exit, res_z)
        Theta, Z = np.meshgrid(theta, z)
        X = col.radius * np.cos(Theta)
        Y = col.radius * np.sin(Theta)
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)

        # entry & exit rims
        rim_theta = np.linspace(0, 2*np.pi, 64)
        rim_x = col.radius * np.cos(rim_theta)
        rim_y = col.radius * np.sin(rim_theta)
        ax.plot(rim_x, rim_y, np.full_like(rim_x, col.z_entry), color=color, linewidth=1)
        ax.plot(rim_x, rim_y, np.full_like(rim_x, col.z_exit), color=color, linewidth=1)

    def _update_plots(self, results, c1: Collimator, c2: Collimator, detector_z):
        self.ax_side.clear()
        self.ax_heat.clear()
        self.ax_3d.clear()

        sample = results if len(results) <= 1500 else random.sample(results, 1500)
        for r in sample:
            origin = r["origin"]; dirv = r["dir"]
            z0 = origin[2]; x0 = origin[0]; y0 = origin[1]
            if r["det_pt"] is not None:
                xd, yd, zd = r["det_pt"]
                color = "tab:green" if r["passed_both"] else "tab:red"
                alpha = 0.6 if r["passed_both"] else 0.15
                # side view line (x vs z)
                self.ax_side.plot([x0, xd], [z0, zd], color=color, linewidth=0.5, alpha=alpha)
                # 3D line
                self.ax_3d.plot([x0, xd], [y0, yd], [z0, zd], color=color, linewidth=0.5, alpha=alpha)
            else:
                # draw short ray in 3D and side
                self.ax_side.plot([x0, x0 + dirv[0]*200], [z0, z0 + dirv[2]*200], color="gray", linewidth=0.3, alpha=0.3)
                self.ax_3d.plot([x0, x0 + dirv[0]*200], [y0, y0 + dirv[1]*200], [z0, z0 + dirv[2]*200], color="gray", linewidth=0.3, alpha=0.25)

        def draw_collimator_on_side(ax, col, color="C0"):
            ax.add_patch(plt.Rectangle((-col.radius, col.z_entry), 2*col.radius, col.length, fill=False, edgecolor=color, linewidth=2, alpha=0.8))
            ax.plot([-col.radius, col.radius], [col.z_entry, col.z_entry], color=color, lw=1)
            ax.plot([-col.radius, col.radius], [col.z_exit, col.z_exit], color=color, lw=1)
            ax.text(col.radius + 1.0, (col.z_entry + col.z_exit)/2.0, f"r={col.radius} mm", color=color, va="center")

        draw_collimator_on_side(self.ax_side, c1, color="C0")
        draw_collimator_on_side(self.ax_side, c2, color="C1")

        self.ax_side.set_xlabel("x (mm)")
        self.ax_side.set_ylabel("z (mm)")
        self.ax_side.set_title("Side view (x vs z) — green = pass both, red = blocked")
        zs = [r["origin"][2] for r in results] + [detector_z]
        self.ax_side.set_ylim(min(zs)-100, detector_z+100)
        xs_all = [r["origin"][0] for r in results] + [r["det_pt"][0] for r in results if r["det_pt"] is not None]
        if xs_all:
            xmax = max(1.0, max(abs(min(xs_all)), abs(max(xs_all))))
            self.ax_side.set_xlim(-max(xmax, c1.radius, c2.radius)-10, max(xmax, c1.radius, c2.radius)+10)

        inset_ax = self.fig.add_axes([0.12, 0.58, 0.12, 0.18])
        inset_ax.set_title("Apertures (top)")
        import matplotlib.patches as mpatches
        inset_ax.add_patch(mpatches.Circle((0,0), c1.radius, fill=False, edgecolor="C0", linewidth=2))
        inset_ax.text(0, -c1.radius-0.2, f"C1 r={c1.radius}", ha="center", va="top", fontsize=8)
        inset_ax.add_patch(mpatches.Circle((0,0), c2.radius, fill=False, edgecolor="C1", linewidth=2))
        inset_ax.text(0, -c2.radius-1.6, f"C2 r={c2.radius}", ha="center", va="top", fontsize=8)
        det_hits = np.array([r["det_pt"][:2] for r in results if r["det_pt"] is not None and r["passed_both"]])
        if det_hits.size:
            inset_ax.scatter(det_hits[:,0], det_hits[:,1], s=5, alpha=0.6)
            maxcoord = max(np.max(np.abs(det_hits)), c1.radius, c2.radius)
            inset_ax.set_xlim(-maxcoord*1.1, maxcoord*1.1)
            inset_ax.set_ylim(-maxcoord*1.1, maxcoord*1.1)
        else:
            inset_ax.set_xlim(-max(c1.radius,c2.radius)*1.5, max(c1.radius,c2.radius)*1.5)
            inset_ax.set_ylim(-max(c1.radius,c2.radius)*1.5, max(c1.radius,c2.radius)*1.5)
        inset_ax.set_aspect("equal", "box")
        inset_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        det_points = np.array([r["det_pt"] for r in results if r["det_pt"] is not None])
        if det_points.size:
            x = det_points[:,0]; y = det_points[:,1]
            std_x = np.std(x) if len(x)>1 else 1.0
            std_y = np.std(y) if len(y)>1 else 1.0
            span_x = max(6*std_x, 5.0)
            span_y = max(6*std_y, 5.0)
            nbins = 80
            xedges = np.linspace(np.mean(x)-span_x, np.mean(x)+span_x, nbins)
            yedges = np.linspace(np.mean(y)-span_y, np.mean(y)+span_y, nbins)
            H, xe, ye = np.histogram2d(x, y, bins=(xedges, yedges))
            extent = [xe[0], xe[-1], ye[0], ye[-1]]
            im = self.ax_heat.imshow(H.T, origin='lower', extent=extent, aspect='equal', interpolation='nearest')
            self.ax_heat.set_title("Detector heatmap (counts)")
            self.ax_heat.set_xlabel("x (mm)")
            self.ax_heat.set_ylabel("y (mm)")
            # avoid duplicate colorbars piling up
            if hasattr(self, "_heat_cb") and self._heat_cb:
                try:
                    self._heat_cb.remove()
                except Exception:
                    pass
            self._heat_cb = self.fig.colorbar(im, ax=self.ax_heat, fraction=0.046, pad=0.04)
        else:
            self.ax_heat.clear()
            self.ax_heat.text(0.5, 0.5, "No detector hits", ha="center", va="center")
            self.ax_heat.set_title("Detector heatmap")
            self.ax_heat.set_xlabel("x (mm)")
            self.ax_heat.set_ylabel("y (mm)")

        # 3D view: draw cylinders and rays and detector plane
        self._plot_cylinder_3d(self.ax_3d, c1, color='C0', alpha=0.3)
        self._plot_cylinder_3d(self.ax_3d, c2, color='C1', alpha=0.3)

        # draw detector plane as translucent square centered at origin
        # choose size based on det hits or aperture radius
        if det_points.size:
            maxcoord = max(np.max(np.abs(det_points)), c1.radius, c2.radius)
            half = maxcoord * 1.5
        else:
            half = max(c1.radius, c2.radius) + 10.0
        xx = np.array([ -half, half, half, -half ])
        yy = np.array([ -half, -half, half, half ])
        zz = np.full_like(xx, detector_z)
        self.ax_3d.plot_trisurf(xx, yy, zz, color='gray', alpha=0.12)

        # draw rays sample in 3D legend-style (do not re-plot all heavy)
        sample3d = results if len(results) <= 2000 else random.sample(results, 2000)
        for r in sample3d:
            o = r["origin"]; d = r["dir"]
            if r["det_pt"] is not None:
                p = r["det_pt"]
                c = 'g' if r["passed_both"] else 'r'
                a = 0.7 if r["passed_both"] else 0.12
                self.ax_3d.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], color=c, linewidth=0.6, alpha=a)
            else:
                self.ax_3d.plot([o[0], o[0]+d[0]*200], [o[1], o[1]+d[1]*200], [o[2], o[2]+d[2]*200], color='gray', linewidth=0.3, alpha=0.2)

        # mark source location(s)
        srcs = np.array([r["origin"] for r in results])
        if srcs.size:
            mean_src = np.mean(srcs, axis=0)
            self.ax_3d.scatter([mean_src[0]], [mean_src[1]], [mean_src[2]], color='k', s=30, label='source center')

        # 3D axes limits
        all_x = [r["origin"][0] for r in results] + ([p[0] for p in det_points] if det_points.size else [])
        all_y = [r["origin"][1] for r in results] + ([p[1] for p in det_points] if det_points.size else [])
        all_z = [r["origin"][2] for r in results] + [detector_z]
        if all_x and all_y and all_z:
            margin = 0.1
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            zmin, zmax = min(all_z), max(all_z)
            # expand a bit
            dx = (xmax - xmin) or 1.0
            dy = (ymax - ymin) or 1.0
            dz = (zmax - zmin) or 1.0
            self.ax_3d.set_xlim(xmin - 0.2*dx, xmax + 0.2*dx)
            self.ax_3d.set_ylim(ymin - 0.2*dy, ymax + 0.2*dy)
            self.ax_3d.set_zlim(zmin - 0.2*dz, zmax + 0.2*dz)

        self.ax_3d.set_xlabel("x (mm)")
        self.ax_3d.set_ylabel("y (mm)")
        self.ax_3d.set_zlabel("z (mm)")
        self.ax_3d.set_title("3D view (rotate with mouse)")

        self.canvas.draw()

    def save_csv(self):
        if self.last_results is None:
            messagebox.showinfo("No data", "No simulation results to save. Run a simulation first.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not fname:
            return
        passing = [r for r in self.last_results if r["passed_both"]]
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["origin_x","origin_y","origin_z","dir_x","dir_y","dir_z","det_x","det_y","det_z"])
            for r in passing:
                ox, oy, oz = r["origin"]
                dx, dy, dz = r["dir"]
                if r["det_pt"] is not None:
                    dpx, dpy, dpz = r["det_pt"]
                else:
                    dpx = dpy = dpz = ""
                writer.writerow([ox, oy, oz, dx, dy, dz, dpx, dpy, dpz])
        messagebox.showinfo("Saved", f"Saved {len(passing)} passing rays to {fname}")


def main():
    root = tk.Tk()
    app = RayTraceGUIApp(root)
    root.geometry("1300x820")
    root.mainloop()

if __name__ == "__main__":
    main()

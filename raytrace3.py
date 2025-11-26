#!/usr/bin/env python3
"""
Beam spot simulator with two adjustable cylindrical collimators (2D detector).
Tkinter GUI with sliders and animation.

Save as beamspot_collimators_gui.py and run:
    python beamspot_collimators_gui.py

Controls:
 - Number of rays per frame (batch)
 - Source position and radius
 - Cone half-angle (deg)
 - Collimator 1 & 2: entry z, length, radius
 - Detector z
Buttons:
 - Start animation (accumulates rays)
 - Stop
 - Reset (clears accumulation)

Visualization:
 - Left: live 2D heatmap (detector) showing accumulated hits
 - Right: side view (x vs z) showing example rays and collimators
 - Metrics printed below: total rays accumulated, pass fraction, RMS radius, 90% containment radius
"""
import math, random, sys, time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm

# -----------------------
# Vectorized ray-tracing helpers
# -----------------------

def sample_sources_disk(center_x, center_y, z, radius, n):
    """Return array shape (n,3) of origin points uniformly sampled in disk."""
    r = np.sqrt(np.random.random(n)) * radius
    theta = np.random.random(n) * 2.0 * np.pi
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    z = np.full(n, z, dtype=float)
    return np.column_stack((x, y, z))

def sample_directions_cone(n, cone_half_angle_deg, axis=np.array([0.,0.,1.])):
    """Vectorized sampling of directions uniformly within cone about +z axis (axis assumed z)."""
    cone = math.radians(float(cone_half_angle_deg))
    u = np.random.random(n)
    cos_max = math.cos(cone)
    cos_theta = u * (1.0 - cos_max) + cos_max
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
    phi = np.random.random(n) * 2.0 * np.pi
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = cos_theta
    dirs = np.column_stack((dx, dy, dz))
    # if axis is not z, rotate (not implemented since axis is z in default UI)
    return dirs / np.linalg.norm(dirs, axis=1)[:,None]

def intersect_plane_z_batch(origins, dirs, z_plane):
    """
    Vectorized: compute t = (z_plane - z0)/vz; return t and points.
    If vz == 0 or t <= 0, mark t = np.nan and point = nan.
    """
    z0 = origins[:,2]
    vz = dirs[:,2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (z_plane - z0) / vz
    mask = (np.abs(vz) > 1e-12) & (t > 1e-12)
    t_valid = np.where(mask, t, np.nan)
    pts = origins + dirs * t_valid[:,None]
    return t_valid, pts

def cylinder_side_intersections_batch(origins, dirs, radius, z1, z2):
    """
    Solve quadratic for intersection with infinite cylinder for arrays.
    Returns two t arrays (t1,t2) where nan indicates no intersection; only t>0 kept.
    Only intersections whose z is between z1 and z2 are considered valid (others set to nan).
    """
    px = origins[:,0]; py = origins[:,1]; pz = origins[:,2]
    vx = dirs[:,0]; vy = dirs[:,1]; vz = dirs[:,2]
    a = vx*vx + vy*vy
    b = 2.0*(px*vx + py*vy)
    c = px*px + py*py - radius*radius
    disc = b*b - 4*a*c
    t1 = np.full_like(a, np.nan, dtype=float)
    t2 = np.full_like(a, np.nan, dtype=float)
    mask = (disc >= 0) & (a > 1e-16)
    if np.any(mask):
        sqrt_d = np.sqrt(disc[mask])
        a_m = a[mask]; b_m = b[mask]
        t1_m = (-b_m - sqrt_d) / (2.0*a_m)
        t2_m = (-b_m + sqrt_d) / (2.0*a_m)
        # assign
        t1[mask] = t1_m
        t2[mask] = t2_m
        # validate z-range and t>0
        for t_arr in (t1, t2):
            valid = ~np.isnan(t_arr)
            if np.any(valid):
                z_at_t = pz[valid] + t_arr[valid]*vz[valid]
                ok = (z_at_t >= min(z1,z2)-1e-9) & (z_at_t <= max(z1,z2)+1e-9) & (t_arr[valid] > 1e-12)
                # set invalid to nan
                invalid_idx = np.where(valid)[0][~ok]
                t_arr[invalid_idx] = np.nan
    return t1, t2

def passes_collimator_batch(origins, dirs, col_z_entry, col_z_exit, radius):
    """
    Determine which rays pass a finite cylindrical collimator (vectorized).
    Logic:
      - compute first valid intersection among entry plane, exit plane, and side intersections
      - if first is side -> blocked
      - if first is entry: must be inside radius; then next hit must be exit (not side)
      - if first is exit: allow pass if exit inside radius and no previous side
    Returns boolean mask array of passes.
    """
    n = origins.shape[0]
    # plane intersections
    t_entry, pt_entry = intersect_plane_z_batch(origins, dirs, col_z_entry)
    t_exit, pt_exit = intersect_plane_z_batch(origins, dirs, col_z_exit)
    # side intersections
    t1_side, t2_side = cylinder_side_intersections_batch(origins, dirs, radius, col_z_entry, col_z_exit)
    # create arrays of first hits: we'll compare t values
    # stack possible hits: for each ray, consider (t, type)
    # create masked arrays for t with nan where invalid
    t_candidates = np.vstack([t_entry, t_exit, t1_side, t2_side]).T  # shape (n,4)
    # types mapping: 0=entry,1=exit,2=side1,3=side2
    # find index of minimum t (ignoring nan)
    t_min = np.nanmin(t_candidates, axis=1)
    # if all nan -> no intersection -> blocked
    no_intersect = np.isnan(t_min)
    # determine which type is first
    # we need to check in order for ties; get argmin where nan treated large
    argmin = np.argmin(np.where(np.isnan(t_candidates), np.inf, t_candidates), axis=1)
    first_type = argmin  # 0..3
    # Prepare pass mask default False
    passes = np.zeros(n, dtype=bool)
    # for rays with no intersection -> blocked
    # for each ray evaluate logic
    # We'll vectorize by cases
    # Case first_type == 2 or 3 => side first => blocked
    side_first_mask = (first_type == 2) | (first_type == 3)
    # Case first_type == 0 (entry)
    entry_first_mask = (first_type == 0) & (~no_intersect)
    if np.any(entry_first_mask):
        idx = np.where(entry_first_mask)[0]
        # entry points:
        pts_entry = pt_entry[idx]
        inside_entry = (pts_entry[:,0]**2 + pts_entry[:,1]**2) <= (radius*radius + 1e-9)
        # For those that entered inside, we must check next hit after entry is exit and not side
        valid_idx = idx[inside_entry]
        if valid_idx.size:
            # For each such ray, find smallest t > t_entry among (t_exit,t1_side,t2_side)
            te = t_entry[valid_idx]
            candidates_next = np.vstack([t_exit[valid_idx], t1_side[valid_idx], t2_side[valid_idx]]).T
            # pick smallest > te with margin
            # mask candidates <= te -> nan
            with np.errstate(invalid='ignore'):
                candidates_next_masked = np.where(candidates_next > (te[:,None] + 1e-12), candidates_next, np.nan)
            t_next = np.nanmin(candidates_next_masked, axis=1)
            # find which candidate it corresponds to; if the min corresponds to exit (col 0) then pass; if side then blocked
            argmin_next = np.argmin(np.where(np.isnan(candidates_next_masked), np.inf, candidates_next_masked), axis=1)
            # argmin_next == 0 => exit
            pass_idx = valid_idx[argmin_next == 0]
            passes[pass_idx] = True
    # Case first_type == 1 (exit first) - origin inside cylinder interior upstream
    exit_first_mask = (first_type == 1) & (~no_intersect)
    if np.any(exit_first_mask):
        idx2 = np.where(exit_first_mask)[0]
        pts_exit = pt_exit[idx2]
        inside_exit = (pts_exit[:,0]**2 + pts_exit[:,1]**2) <= (radius*radius + 1e-9)
        # ensure no side intersection before exit
        if np.any(inside_exit):
            idx_ok = idx2[inside_exit]
            # check if any side t < t_exit
            # for each ray check if t1_side or t2_side exists and < t_exit
            t_exit_vals = t_exit[idx_ok]
            t1_vals = t1_side[idx_ok]; t2_vals = t2_side[idx_ok]
            side_before = ((~np.isnan(t1_vals)) & (t1_vals < t_exit_vals - 1e-12)) | ((~np.isnan(t2_vals)) & (t2_vals < t_exit_vals - 1e-12))
            pass_idx2 = idx_ok[~side_before]
            passes[pass_idx2] = True
    # final mask: pass only those flagged True
    passes = passes & (~no_intersect)
    return passes

# -----------------------
# Spot statistics
# -----------------------

def rms_radius(xy):
    if xy.size == 0:
        return 0.0
    r2 = np.sum(xy**2, axis=1)
    return math.sqrt(np.mean(r2))

def containment_radius(xy, fraction=0.9):
    """Return radius that encloses `fraction` of points (approx by sorting radii)."""
    if xy.size == 0:
        return 0.0
    r = np.sqrt(np.sum(xy**2, axis=1))
    r_sorted = np.sort(r)
    idx = min(len(r_sorted)-1, max(0, int(math.floor(fraction * len(r_sorted))) - 1))
    return float(r_sorted[idx])

# -----------------------
# GUI and Animation
# -----------------------

class BeamSpotGUI:
    def __init__(self, root):
        self.root = root
        root.title("Beam Spot through Two Collimators — 2D Heatmap + Animation")

        # left control frame
        ctrl = ttk.Frame(root, padding=8)
        ctrl.grid(row=0, column=0, sticky="nw")

        def add_slider(label, default, from_, to, row, resolution=0.01):
            ttk.Label(ctrl, text=label).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=default)
            ent = ttk.Entry(ctrl, width=8, textvariable=var)
            ent.grid(row=row, column=1, padx=4)
            s = ttk.Scale(ctrl, from_=from_, to=to, orient="horizontal", command=lambda v,var=var: var.set(float(v)))
            s.set(default)
            s.grid(row=row, column=2, padx=6)
            return var

        # batch size per animation frame
        self.batch_var = tk.IntVar(value=5000)
        ttk.Label(ctrl, text="Rays / frame:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, width=8, textvariable=self.batch_var).grid(row=0, column=1)

        row = 1
        self.src_x = add_slider("Source X (mm)", 0.0, -10.0, 10.0, row); row+=1
        self.src_y = add_slider("Source Y (mm)", 0.0, -10.0, 10.0, row); row+=1
        self.src_z = add_slider("Source Z (mm)", -2590.0, -10000.0, 0.0, row); row+=1
        self.src_r = add_slider("Source radius (mm)", 0.5, 0.0, 10.0, row); row+=1
        self.cone_deg = add_slider("Cone half-angle (deg)", 0.5, 0.0, 10.0, row); row+=1

        ttk.Separator(ctrl, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        ttk.Label(ctrl, text="Collimator 1").grid(row=row, column=0, sticky="w"); row+=1
        self.c1_z = add_slider("C1 entry z (mm)", 0.0, -2000.0, 2000.0, row); row+=1
        self.c1_len = add_slider("C1 length (mm)", 128.6, 0.1, 2000.0, row); row+=1
        self.c1_r = add_slider("C1 radius (mm)", 0.5, 0.01, 20.0, row); row+=1

        ttk.Label(ctrl, text="Collimator 2").grid(row=row, column=0, sticky="w"); row+=1
        self.c2_z = add_slider("C2 entry z (mm)", 500.0, -2000.0, 5000.0, row); row+=1
        self.c2_len = add_slider("C2 length (mm)", 128.6, 0.1, 2000.0, row); row+=1
        self.c2_r = add_slider("C2 radius (mm)", 0.5, 0.01, 20.0, row); row+=1

        self.det_z = add_slider("Detector z (mm)", 2590.0, -5000.0, 10000.0, row); row+=1

        ttk.Separator(ctrl, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        # buttons
        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start_animation)
        self.start_btn.grid(row=row, column=0, pady=8)
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.stop_animation, state="disabled")
        self.stop_btn.grid(row=row, column=1, pady=8)
        self.reset_btn = ttk.Button(ctrl, text="Reset", command=self.reset_accumulation)
        self.reset_btn.grid(row=row, column=2, pady=8)
        row += 1

        # metrics
        self.metrics_var = tk.StringVar(value="Rays: 0 | Passed: 0 | Pass frac: 0.00 | RMS r: 0.00 | 90% r: 0.00")
        ttk.Label(ctrl, textvariable=self.metrics_var, foreground="blue").grid(row=row, column=0, columnspan=3, sticky="w", pady=(6,0))
        row += 1

        # Figure and plots
        fig = plt.Figure(figsize=(8,4))
        self.ax_heat = fig.add_subplot(1,2,0+1)  # left
        self.ax_side = fig.add_subplot(1,2,2)    # right

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # heatmap grid settings
        self.grid_bins = 160
        self.heatmap = np.zeros((self.grid_bins, self.grid_bins), dtype=np.int32)
        self.xedges = np.linspace(-10, 10, self.grid_bins+1)
        self.yedges = np.linspace(-10, 10, self.grid_bins+1)
        self.extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        # animation control
        self.anim_running = False
        self.total_rays = 0
        self.total_passed = 0

        # prepare initial plots
        self.im = self.ax_heat.imshow(self.heatmap.T, origin='lower', extent=self.extent, interpolation='nearest', cmap=cm.inferno)
        self.ax_heat.set_title("Detector heatmap (accumulated)")
        self.ax_heat.set_xlabel("x (mm)")
        self.ax_heat.set_ylabel("y (mm)")

        self.side_lines = []
        self.ax_side.set_title("Side view (sample rays)")
        self.ax_side.set_xlabel("x (mm)")
        self.ax_side.set_ylabel("z (mm)")

        self.canvas.draw()

        # For animation scheduling using Tk's after()
        self._after_id = None

    def reset_accumulation(self):
        self.heatmap[:] = 0
        self.total_rays = 0
        self.total_passed = 0
        # reset edges to reasonable default based on apertures/detector
        self._reset_edges()
        self.update_plots()
        self.metrics_var.set("Rays: 0 | Passed: 0 | Pass frac: 0.00 | RMS r: 0.00 | 90% r: 0.00")

    def _reset_edges(self):
        # set heatmap extents based on current collimator radii and expected spread
        maxrad = max(self.c1_r.get(), self.c2_r.get(), 1.0)
        span = maxrad * 6.0 + 2.0
        # center at 0 + detector offset if source offset exists? Keep centered on 0
        cx = float(self.src_x.get())
        cy = float(self.src_y.get())
        self.xedges = np.linspace(cx - span, cx + span, self.grid_bins+1)
        self.yedges = np.linspace(cy - span, cy + span, self.grid_bins+1)
        self.extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

    def start_animation(self):
        if self.anim_running:
            return
        # reset heatmap if none
        self._reset_edges()
        self.anim_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._schedule_frame()

    def stop_animation(self):
        if not self.anim_running:
            return
        self.anim_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _schedule_frame(self):
        # schedule next frame via after()
        self._after_id = self.root.after(1, self._run_frame)

    def _run_frame(self):
        # run one frame: sample batch rays, trace, update heatmap + metrics + plots
        n = int(self.batch_var.get())
        # read parameters
        sx = float(self.src_x.get()); sy = float(self.src_y.get()); sz = float(self.src_z.get())
        srad = float(self.src_r.get()); cone = float(self.cone_deg.get())
        c1_z = float(self.c1_z.get()); c1_len = float(self.c1_len.get()); c1_r = float(self.c1_r.get())
        c2_z = float(self.c2_z.get()); c2_len = float(self.c2_len.get()); c2_r = float(self.c2_r.get())
        det_z = float(self.det_z.get())

        # sample origins & directions
        origins = sample_sources_disk(sx, sy, sz, srad, n)
        dirs = sample_directions_cone(n, cone)

        # test collimators
        c1_pass = passes_collimator_batch(origins, dirs, c1_z, c1_z + c1_len, c1_r)
        # For rays blocked at c1, we can optionally skip c2 calc
        # But we'll compute for all and AND them
        c2_pass = passes_collimator_batch(origins, dirs, c2_z, c2_z + c2_len, c2_r)

        passed_both = c1_pass & c2_pass

        # detector hits for rays with positive t to detector
        vz = dirs[:,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_det = (det_z - origins[:,2]) / vz
        valid_det = (t_det > 1e-12) & (~np.isnan(t_det))
        det_x = origins[:,0] + dirs[:,0]*t_det
        det_y = origins[:,1] + dirs[:,1]*t_det

        # keep only rays that passed both and have valid detector intersection
        keep_mask = passed_both & valid_det
        kept_x = det_x[keep_mask]
        kept_y = det_y[keep_mask]
        # update heatmap bins
        if kept_x.size:
            # compute bin indices
            ix = np.searchsorted(self.xedges, kept_x, side='right') - 1
            iy = np.searchsorted(self.yedges, kept_y, side='right') - 1
            # clip
            valid_bins = (ix >= 0) & (ix < self.grid_bins) & (iy >= 0) & (iy < self.grid_bins)
            ix = ix[valid_bins]; iy = iy[valid_bins]
            # accumulate
            np.add.at(self.heatmap, (ix, iy), 1)

        # update counts
        self.total_rays += n
        self.total_passed += int(np.sum(keep_mask))

        # compute metrics on accumulated points
        # get coordinates of all accumulated hits (could be large; compute from heatmap)
        # For metrics, reconstruct sample points from heatmap centers weighted by counts
        counts = self.heatmap
        total_hits = int(counts.sum())
        if total_hits > 0:
            # compute bin centers
            xcenters = 0.5*(self.xedges[:-1] + self.xedges[1:])
            ycenters = 0.5*(self.yedges[:-1] + self.yedges[1:])
            Xc, Yc = np.meshgrid(xcenters, ycenters, indexing='xy')
            # counts shape (bins_x, bins_y) -> need flatten accordingly
            flat_counts = counts.T.ravel()  # transpose so x varies slow? keep consistent with extent
            flat_x = Xc.T.ravel()
            flat_y = Yc.T.ravel()
            # expand weighted sample approx: use weighted statistics
            mean_x = np.sum(flat_x * flat_counts) / total_hits
            mean_y = np.sum(flat_y * flat_counts) / total_hits
            # radii
            r2_weighted = (flat_x-mean_x)**2 + (flat_y-mean_y)**2
            rms_r = math.sqrt(np.sum(r2_weighted * flat_counts) / total_hits)
            # containment radius approx: reconstruct list of radii by repeating centers proportional to counts (but that may still be large)
            # Instead approximate by expanding arrays where counts > 0 into vector of radii indices but limited size
            # We'll construct an array of radii up to max 200k entries to compute containment approx
            max_expand = 200000
            if total_hits <= max_expand:
                rx = np.repeat(flat_x, flat_counts.astype(int))
                ry = np.repeat(flat_y, flat_counts.astype(int))
                radii = np.sqrt((rx-mean_x)**2 + (ry-mean_y)**2)
                r90 = float(np.percentile(radii, 90.0))
            else:
                # sample proportionally
                probs = flat_counts / flat_counts.sum()
                sample_n = min(max_expand, int(total_hits))
                choose_idx = np.random.choice(len(flat_counts), size=sample_n, p=probs)
                rx = flat_x[choose_idx] + (np.random.random(sample_n)-0.5)*(self.xedges[1]-self.xedges[0])
                ry = flat_y[choose_idx] + (np.random.random(sample_n)-0.5)*(self.yedges[1]-self.yedges[0])
                radii = np.sqrt((rx-mean_x)**2 + (ry-mean_y)**2)
                r90 = float(np.percentile(radii, 90.0))
        else:
            rms_r = 0.0; r90 = 0.0

        pass_frac = (self.total_passed / max(1, self.total_rays))

        # update plots and metrics
        self.metrics_var.set(f"Rays: {self.total_rays} | Passed: {self.total_passed} | Pass frac: {pass_frac:.5f} | RMS r: {rms_r:.3f} mm | 90% r: {r90:.3f} mm")

        self.update_plots(sample_origins=origins, sample_dirs=dirs, kept_mask=keep_mask, det_x=det_x, det_y=det_y,
                          c1=(c1_z, c1_z+c1_len, c1_r), c2=(c2_z, c2_z+c2_len, c2_r), detector_z=det_z)

        # schedule next frame if running
        if self.anim_running:
            self._schedule_frame()
        else:
            self.stop_animation()

    def update_plots(self, sample_origins=None, sample_dirs=None, kept_mask=None, det_x=None, det_y=None,
                     c1=None, c2=None, detector_z=None):
        # update heatmap image
        self.im.set_data(self.heatmap.T)
        self.im.set_extent(self.extent)
        self.im.set_clim(0, max(1, self.heatmap.max()))
        self.ax_heat.set_xlim(self.extent[0], self.extent[1])
        self.ax_heat.set_ylim(self.extent[2], self.extent[3])
        # side view: clear and draw a small sample of rays for clarity
        self.ax_side.cla()
        self.ax_side.set_xlabel("x (mm)"); self.ax_side.set_ylabel("z (mm)")
        self.ax_side.set_title("Side view (x vs z) sample rays")
        if c1 is not None:
            cz1, cz1e, cr1 = c1
            self.ax_side.add_patch(plt.Rectangle((-cr1, cz1), 2*cr1, cz1e-cz1, fill=False, edgecolor='C0'))
        if c2 is not None:
            cz2, cz2e, cr2 = c2
            self.ax_side.add_patch(plt.Rectangle((-cr2, cz2), 2*cr2, cz2e-cz2, fill=False, edgecolor='C1'))
        if sample_origins is not None and sample_dirs is not None:
            # choose small subset for plot
            N = sample_origins.shape[0]
            idx = np.random.choice(N, size=min(300, N), replace=False)
            for i in idx:
                o = sample_origins[i]; d = sample_dirs[i]
                # draw line from origin to detector intersection if exists within shown z range
                if det_x is not None and det_y is not None:
                    # compute det z from provided detector_z in args
                    # we were given det_x,det_y arrays computed earlier; but simpler draw short line
                    self.ax_side.plot([o[0], o[0] + d[0]*400], [o[2], o[2] + d[2]*400], color='gray', linewidth=0.4, alpha=0.5)
                else:
                    self.ax_side.plot([o[0], o[0] + d[0]*400], [o[2], o[2] + d[2]*400], color='gray', linewidth=0.4, alpha=0.5)
        # refresh canvas
        self.canvas.draw_idle()

# -----------------------
# Run app
# -----------------------

def main():
    root = tk.Tk()
    app = BeamSpotGUI(root)
    root.geometry("1100x640")
    root.mainloop()

if __name__ == "__main__":
    main()

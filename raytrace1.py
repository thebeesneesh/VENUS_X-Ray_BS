import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
#   Ray-Tracing Core
# -----------------------------

class Collimator:
    def __init__(self, z_entry, length, radius):
        self.z_entry = float(z_entry)
        self.length = float(length)
        self.z_exit = float(z_entry + length)
        self.radius = float(radius)

def intersect_ray_with_plane_z(p0, v, z_plane, eps=1e-12):
    p0 = np.array(p0, dtype=float)
    v = np.array(v, dtype=float)
    if abs(v[2]) < eps:
        return None, None
    t = (z_plane - p0[2]) / v[2]
    if t < 0:
        return None, None
    return t, p0 + t * v

def point_inside_aperture(pt, radius):
    return pt[0]**2 + pt[1]**2 <= radius**2

def trace_ray(p0, v, col):
    t1, p1 = intersect_ray_with_plane_z(p0, v, col.z_entry)
    t2, p2 = intersect_ray_with_plane_z(p0, v, col.z_exit)
    if p1 is None or p2 is None:
        return False
    return point_inside_aperture(p1, col.radius) and point_inside_aperture(p2, col.radius)

def sample_direction_in_cone(axis, max_angle):
    axis = axis / np.linalg.norm(axis)
    u = random.random()
    cos_max = math.cos(max_angle)
    cos_theta = u * (1 - cos_max) + cos_max
    sin_theta = math.sqrt(1 - cos_theta*cos_theta)
    phi = random.random() * 2 * math.pi
    dir_local = np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta
    ])
    z = np.array([0,0,1.0])
    if np.allclose(axis, z):
        return dir_local
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    c = np.dot(z, axis)
    if s < 1e-12:
        return dir_local if c > 0 else -dir_local
    vx = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])
    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)
    return R @ dir_local


def monte_carlo(n, source_z, source_radius, cone_deg, col1, col2, detector_z):
    axis = np.array([0,0,1.0])
    cone_rad = math.radians(cone_deg)

    results = []
    passed = 0

    for _ in range(n):
        # sample source point in disk
        r = math.sqrt(random.random()) * source_radius
        th = random.random() * 2 * math.pi
        x = r * math.cos(th)
        y = r * math.sin(th)
        origin = np.array([x, y, source_z])

        v = sample_direction_in_cone(axis, cone_rad)
        v = v / np.linalg.norm(v)

        p1 = trace_ray(origin, v, col1)
        p2 = trace_ray(origin, v, col2)
        pboth = p1 and p2

        hit = None
        if v[2] != 0:
            t = (detector_z - origin[2]) / v[2]
            if t > 0:
                hit = origin + t * v

        results.append({
            "origin_x": origin[0], "origin_y": origin[1],
            "dir_x": v[0], "dir_y": v[1], "dir_z": v[2],
            "passed": pboth,
            "det_x": hit[0] if hit is not None else None,
            "det_y": hit[1] if hit is not None else None
        })

        if pboth:
            passed += 1

    return results, passed / n


# -----------------------------
# GUI
# -----------------------------

class RayTraceGUI:

    def __init__(self, root):
        self.root = root
        root.title("Ray Tracing Through Two Collimators")

        frm = ttk.Frame(root, padding=10)
        frm.grid()

        def add(label, default):
            row = ttk.Frame(frm)
            row.grid(sticky="w", pady=2)
            ttk.Label(row, text=label, width=28).pack(side="left")
            ent = ttk.Entry(row, width=12)
            ent.pack(side="left")
            ent.insert(0, str(default))
            return ent

        # Inputs
        self.n_rays = add("Number of Rays:", 2000)
        self.source_z = add("Source Z position (mm):", -2590)
        self.source_radius = add("Source Radius (mm):", 0.5)
        self.cone_angle = add("Cone Half-Angle (deg):", 1.0)

        ttk.Label(frm, text="--- Collimator 1 ---", font=("Arial",10,"bold")).grid(pady=5)
        self.col1_z = add("Collimator 1 Entry Z (mm):", 0)
        self.col1_len = add("Collimator 1 Length (mm):", 128.6)
        self.col1_rad = add("Collimator 1 Radius (mm):", 0.5)

        ttk.Label(frm, text="--- Collimator 2 ---", font=("Arial",10,"bold")).grid(pady=5)
        self.col2_z = add("Collimator 2 Entry Z (mm):", 500)
        self.col2_len = add("Collimator 2 Length (mm):", 128.6)
        self.col2_rad = add("Collimator 2 Radius (mm):", 0.5)

        self.det_z = add("Detector Z (mm):", 2590)

        ttk.Button(frm, text="Run Simulation", command=self.run).grid(pady=10)

    def run(self):
        try:
            n = int(self.n_rays.get())
            sz = float(self.source_z.get())
            sr = float(self.source_radius.get())
            cone = float(self.cone_angle.get())

            c1 = Collimator(float(self.col1_z.get()),
                            float(self.col1_len.get()),
                            float(self.col1_rad.get()))
            c2 = Collimator(float(self.col2_z.get()),
                            float(self.col2_len.get()),
                            float(self.col2_rad.get()))

            dz = float(self.det_z.get())
        except:
            messagebox.showerror("Error", "Invalid input.")
            return

        results, frac = monte_carlo(n, sz, sr, cone, c1, c2, dz)

        df = pd.DataFrame(results)

        messagebox.showinfo("Results",
            f"Total Rays: {n}\n"
            f"Passed Both Collimators: {df['passed'].sum()}\n"
            f"Pass Fraction: {frac:.6f}"
        )

        # ---- Plot ----
        plt.figure(figsize=(7,6))
        for _, row in df.sample(min(200, len(df))).iterrows():
            x0, z0 = row["origin_x"], sz
            if row["det_x"] is not None:
                xd, zd = row["det_x"], dz
                plt.plot([x0, xd], [z0, zd], linewidth=0.5)
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("Ray Paths (Sample)")
        plt.grid(True)
        plt.show()

        print(df[df["passed"] == True].head())


# -----------------------------
# MAIN
# -----------------------------

root = tk.Tk()
gui = RayTraceGUI(root)
root.mainloop()
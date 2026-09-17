""" DetermineVolume.py — Python 3 version

    This program determines the volume of the resonance zone.

    Converted from Python 2 to Python 3 by ChatGPT (2025-11-04)

    
    This program will determine the volume of the resonance zone.
    The user supplies and external text file called "AxialField.txt"
    that has two columns: z-position and Bz with two additional header
    lines defining units.  This file is read to get the axial field and   
    the radial field due to solenoids is determined by simple integration
    of Maxwell's equation delB=0.  The radial field is set by defining
    a radius and a field at that radius.  The model uses a scaled 
    electrostatic sextupole to determine the vector field due to the 
    sextupole inside the source.  These are summed in order to get the total
    field at any point within the source.  The volume calculation is done
    by defining a mesh of points which are checked for being inside the
    resonance zone.  If they are inside, the volume surrounding that mesh
    point is added to total volume.  The volume calculation is improved by
    successively increasing the number of mesh points checked.
    			
    			Damon Todd
    			Thu Jun 19 12:01:58 PDT 2008 
"""

import argparse
from math import cos, pi, sin, sqrt

import numpy as np

from VENUSMagFieldCalcTest import VenusBFieldCalculator


def spline(x, y):
    n = len(x)
    d = np.zeros(n)
    b = np.zeros(n)
    y2 = np.zeros(n)
    dx = x[1] - x[0]

    d[0] = 1.0
    y2[1] = dx
    d[1] = 4.0 * dx
    b[1] = 6.0 * (y[2] - 2.0 * y[1] + y[0]) / dx

    for j in range(2, n - 1):
        b[j] = y2[j - 1] / d[j - 1]
        y2[j] = dx
        d[j] = 4.0 * dx - y2[j] * b[j]
        b[j] = (
            6.0 * (y[j + 1] - 2.0 * y[j] + y[j - 1]) / dx
            - b[j - 1] * b[j]
        )

    y2[-1] = 0.0
    y2[-2] = b[-2] / d[-2]

    for i in range(n - 3, 0, -1):
        y2[i] = (b[i] - y2[i] * y2[i + 1]) / d[i]

    y2[0] = 0.0
    return y2


def splint(x, y, y2, value):
    if value <= x[0]:
        return y[0]

    if value >= x[-1]:
        return y[-1]

    index = np.searchsorted(x, value)
    i = index - 1
    dx = x[i + 1] - x[i]

    left = value - x[i]
    right = x[i + 1] - value

    return (
        (y2[i + 1] * left**3 + y2[i] * right**3) / (6.0 * dx)
        + (y[i + 1] / dx - y2[i + 1] * dx / 6.0) * left
        + (y[i] / dx - y2[i] * dx / 6.0) * right
    )


def make_derivatives(z, bz):
    second_derivative = spline(z, bz)
    derivative = np.gradient(bz, z)
    return derivative, second_derivative


def get_sextupole_field(x, y, multiplier, bars):
    field = np.zeros(2)

    for i in range(6):
        sign = -1 if i % 2 == 0 else 1
        dx = x - bars[0, i]
        dy = y - bars[1, i]
        distance_squared = dx * dx + dy * dy

        field[0] += sign * multiplier * dx / distance_squared
        field[1] += sign * multiplier * dy / distance_squared

    return field


def get_solenoid_field(
    x,
    y,
    z_position,
    radius,
    z,
    bz,
    bz_second,
    dbz_dz,
    dbz_dz_second,
):
    field = np.zeros(3)
    field[2] = splint(z, bz, bz_second, z_position)

    if radius != 0.0:
        radial_field = (
            -radius
            / 2.0
            * splint(z, dbz_dz, dbz_dz_second, z_position)
        )
        field[0] = x * radial_field / radius
        field[1] = y * radial_field / radius

    return field


def calculate_volume(
    injection,
    center,
    extraction,
    sextupole,
    b_res,
    rmax_cm,
    zmin_cm,
    zmax_cm,
    nx,
    nz,
    write_points=False,
):
    calculator = VenusBFieldCalculator()

    axial = calculator.calculate_axial(
        injection,
        center,
        extraction,
        sextupole,
    )

    radial_field = calculator.calculate_radial(sextupole)

    z = calculator.z_cm / 100.0
    bz = axial["total_t"]

    dbz_dz, bz_second = make_derivatives(z, bz)
    dbz_dz_second = spline(z, dbz_dz)

    bars = np.zeros((2, 6))

    for i in range(6):
        angle = i * 2.0 * pi / 6.0
        bars[0, i] = 10.0 * cos(angle)
        bars[1, i] = 10.0 * sin(angle)

    # Calibrate the analytical sextupole model to the calculated
    # radial field at the chamber wall.
    chamber_radius_m = calculator.RADIAL_RADIUS_CM[-1] / 100.0
    chamber_field_t = radial_field[-1]

    model_field = get_sextupole_field(
        chamber_radius_m,
        0.0,
        1.0,
        bars,
    )[0]

    multiplier = chamber_field_t / model_field

    rmax = rmax_cm / 100.0
    zmin = zmin_cm / 100.0
    zmax = zmax_cm / 100.0

    dx = 2.0 * rmax / nx
    dy = dx
    dz = (zmax - zmin) / nz

    x_start = dx / 2.0 - rmax
    y_start = x_start
    z_start = zmin + dz / 2.0

    cell_volume = dx * dy * dz
    total_volume = 0.0

    minimum_z = float("inf")
    maximum_z = float("-inf")
    maximum_radius = 0.0

    output_file = open("pts.m", "w") if write_points else None

    try:
        for i in range(nx):
            for j in range(nx):
                for k in range(nz):
                    x = x_start + i * dx
                    y = y_start + j * dy
                    z_position = z_start + k * dz
                    radius = sqrt(x * x + y * y)

                    if radius >= rmax:
                        continue

                    sextupole_field = get_sextupole_field(
                        x,
                        y,
                        multiplier,
                        bars,
                    )

                    solenoid_field = get_solenoid_field(
                        x,
                        y,
                        z_position,
                        radius,
                        z,
                        bz,
                        bz_second,
                        dbz_dz,
                        dbz_dz_second,
                    )

                    bx = sextupole_field[0] + solenoid_field[0]
                    by = sextupole_field[1] + solenoid_field[1]
                    bz_value = solenoid_field[2]

                    field_magnitude = sqrt(
                        bx * bx
                        + by * by
                        + bz_value * bz_value
                    )

                    if field_magnitude <= b_res:
                        total_volume += cell_volume
                        minimum_z = min(minimum_z, z_position)
                        maximum_z = max(maximum_z, z_position)
                        maximum_radius = max(maximum_radius, radius)

                        if output_file is not None:
                            output_file.write(
                                f"{x} {y} {z_position}\n"
                            )
    finally:
        if output_file is not None:
            output_file.close()

    return {
        "volume_cm3": total_volume * 1.0e6,
        "z_min_cm": minimum_z * 100.0,
        "z_max_cm": maximum_z * 100.0,
        "radius_max_cm": maximum_radius * 100.0,
        "points_checked": nx * nx * nz,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the VENUS plasma resonance volume."
    )

    parser.add_argument("injection", type=float)
    parser.add_argument("center", type=float)
    parser.add_argument("extraction", type=float)
    parser.add_argument("sextupole", type=float)

    parser.add_argument("--bres", type=float, default=0.64)
    parser.add_argument("--rmax-cm", type=float, default=7.2)
    parser.add_argument("--zmin-cm", type=float, default=-100.0)
    parser.add_argument("--zmax-cm", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument("--write-points", action="store_true")

    args = parser.parse_args()

    result = calculate_volume(
        injection=args.injection,
        center=args.center,
        extraction=args.extraction,
        sextupole=args.sextupole,
        b_res=args.bres,
        rmax_cm=args.rmax_cm,
        zmin_cm=args.zmin_cm,
        zmax_cm=args.zmax_cm,
        nx=args.nx,
        nz=args.nz,
        write_points=args.write_points,
    )

    print(f"Points checked: {result['points_checked']}")
    print(f"Plasma volume: {result['volume_cm3']:.3f} cm^3")
    print(
        "Resonance region: "
        f"z = {result['z_min_cm']:.2f} to "
        f"{result['z_max_cm']:.2f} cm, "
        f"r = {result['radius_max_cm']:.2f} cm"
    )


if __name__ == "__main__":
    main()
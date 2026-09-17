from DetermineVolumeNew import calculate_volume


def read_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def main():
    print("VENUS Plasma Volume Calculator")

    injection = read_float("Injection current (A): ")
    center = read_float("Center current (A): ")
    extraction = read_float("Extraction current (A): ")
    sextupole = read_float("Sextupole current (A): ")
    becr = read_float("BECR/resonance field (T): ")

    result = calculate_volume(
        injection=injection,
        center=center,
        extraction=extraction,
        sextupole=sextupole,
        b_res=becr,
        rmax_cm=7.2,
        zmin_cm=-100.0,
        zmax_cm=100.0,
        nx=100,
        nz=200,
        write_points=False,
    )

    print(f"\nPlasma volume: {result['volume_cm3']:.3f} cm^3")
    print(
        f"Resonance region: z = {result['z_min_cm']:.2f} to "
        f"{result['z_max_cm']:.2f} cm, "
        f"r = {result['radius_max_cm']:.2f} cm"
    )


if __name__ == "__main__":
    main()
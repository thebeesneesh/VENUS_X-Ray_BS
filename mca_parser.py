import argparse
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np


def parse_mca_file(path):
    path = Path(path)
    with path.open('r', encoding='utf-8', errors='ignore') as fh:
        lines = [line.strip() for line in fh.readlines()]

    livetime = None
    start_time = None
    coarse_gain = None
    fine_gain = None
    gain = None
    fast_count = None
    slow_count = None
    raw_data = []
    gain_formula = None

    in_data = False
    for line in lines:
        if not line:
            continue

        if line == '<<DATA>>':
            in_data = True
            continue

        if line == '<<END>>':
            in_data = False
            continue

        if in_data:
            parts = line.split()
            if not parts:
                continue
            try:
                raw_data.extend(int(p) for p in parts)
            except ValueError:
                continue
            continue

        if livetime is None:
            m = re.match(r'^LIVE_TIME\s*-\s*([0-9.]+)$', line)
            if m:
                livetime = float(m.group(1))
                continue

        if start_time is None:
            m = re.match(r'^START_TIME\s*-\s*(.+)$', line)
            if m:
                start_time = m.group(1)
                continue

        if coarse_gain is None:
            m = re.match(r'^Coarse Gain:\s*([0-9.]+)', line, re.I)
            if m:
                coarse_gain = float(m.group(1))
                continue

        if fine_gain is None:
            m = re.match(r'^Fine Gain:\s*([0-9.]+)', line, re.I)
            if m:
                fine_gain = float(m.group(1))
                continue

        if gain is None:
            m = re.match(r'^GAIN\s*-\s*([0-9.]+)$', line)
            if m:
                gain = float(m.group(1))
                continue

        if fast_count is None:
            m = re.match(r'^Fast Count:\s*([0-9]+)$', line, re.I)
            if m:
                fast_count = int(m.group(1))
                continue

        if slow_count is None:
            m = re.match(r'^Slow Count:\s*([0-9]+)$', line, re.I)
            if m:
                slow_count = int(m.group(1))
                continue

    if coarse_gain is not None and fine_gain is not None:
        gain = coarse_gain * fine_gain
        gain_formula = f'{coarse_gain} x {fine_gain}'
    elif gain is None:
        gain = 1.0

    return {
        'file': path.name,
        'livetime': livetime,
        'start_time': start_time,
        'RawData': raw_data,
        'gain': gain,
        'gain_formula': gain_formula,
        'fast_count': fast_count,
        'slow_count': slow_count,
    }


def channel_to_energy(channels, gain=10.0):
    if gain == 10.0:
        calibration = {
            93.7: 46.5, #Pb-210
            27.7: 10.8, #Pb-210
            160.4: 80.998, #Ba-133
            63.5: 30.973, #Ba-133
            72.0: 34.987, #Ba-133
            81.8: 40.118, #Eu-152
            92.8: 45.414, #Eu-152
            87.5: 42.996, #Eu-154
            99.6: 48.695, #Eu-154
            66.0: 32.194, #Cs-137
        }
        x = np.array(list(channels), dtype=float)
        y = np.array(list(calibration.values()), dtype=float)
        x_ref = np.array(list(calibration.keys()), dtype=float)
        if len(x_ref) >= 2:
            fit = np.polyfit(x_ref, y, 1)
            slope, intercept = fit
            return x * slope + intercept
    return np.array(channels, dtype=float)


def plot_series(ax, values, title, ylabel, x_values=None, x_label='Channel'):
    if x_values is None:
        x_values = list(range(len(values)))
    ax.plot(x_values, values, linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def score_linear_fit(x_fit, y_fit, slope, intercept):
    predicted = slope * x_fit + intercept
    residuals = y_fit - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    if ss_tot == 0:
        return 0.0, 0.0
    r_squared = 1.0 - (ss_res / ss_tot)
    return r_squared, ss_res / len(x_fit)


def find_most_linear_log_range(x_values, y_values, peak_window=(60.0, 100.0), max_energy=200.0):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite_positive = np.isfinite(x) & np.isfinite(y) & (y > 0)

    peak_mask = finite_positive & (x >= peak_window[0]) & (x <= peak_window[1])
    if np.any(peak_mask):
        peak_energy = x[peak_mask][np.argmax(y[peak_mask])]
    else:
        peak_energy = 80.0

    min_energy = max(peak_energy, 80.0)
    fit_mask = finite_positive & (x > min_energy) & (x <= max_energy)
    x_candidates = x[fit_mask]
    y_candidates = np.log(y[fit_mask])
    n = len(x_candidates)

    if n < 2:
        return None

    min_points = min(max(12, n // 5), n)
    min_span = min(20.0, max(0.0, x_candidates[-1] - x_candidates[0]))
    full_span = max(x_candidates[-1] - x_candidates[0], 1.0)

    best = None
    for start in range(n - min_points + 1):
        for stop in range(start + min_points, n + 1):
            x_fit = x_candidates[start:stop]
            y_fit = y_candidates[start:stop]
            span = x_fit[-1] - x_fit[0]
            if span < min_span:
                continue

            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            r_squared, mean_squared_error = score_linear_fit(x_fit, y_fit, slope, intercept)
            length_bonus = 0.03 * ((stop - start) / n)
            span_bonus = 0.03 * (span / full_span)
            slope_penalty = 0.25 if slope >= 0 else 0.0
            score = r_squared + length_bonus + span_bonus - slope_penalty

            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'slope': slope,
                    'intercept': intercept,
                    'x_fit': x_fit,
                    'y_fit': y_fit,
                    'r_squared': r_squared,
                    'mean_squared_error': mean_squared_error,
                    'peak_energy': peak_energy,
                    'min_energy': x_fit[0],
                    'max_energy': x_fit[-1],
                }

    return best


def fit_log_linear(x_values, y_values):
    best_range = find_most_linear_log_range(x_values, y_values)
    if best_range is not None:
        return best_range

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x_fit = x[mask]
    y_fit = np.log(y[mask])
    if len(x_fit) < 2:
        return None
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    r_squared, mean_squared_error = score_linear_fit(x_fit, y_fit, slope, intercept)
    return {
        'score': r_squared,
        'slope': slope,
        'intercept': intercept,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'r_squared': r_squared,
        'mean_squared_error': mean_squared_error,
        'peak_energy': None,
        'min_energy': x_fit[0],
        'max_energy': x_fit[-1],
    }


def analyze_mca_file(input_file, output_dir):
    result = parse_mca_file(input_file)
    output_lines = [
        f"File: {result['file']}",
        f"Livetime: {result['livetime']}",
        f"Start time: {result['start_time']}",
    ]

    if result['gain_formula']:
        output_lines.append(f"Gain: {result['gain']}  (calculated from {result['gain_formula']})")
    else:
        output_lines.append(f"Gain: {result['gain']}")

    output_lines.extend([
        f"Fast count: {result['fast_count']}",
        f"Slow count: {result['slow_count']}",
        f"RawData length: {len(result['RawData'])}",
    ])

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    plot_series(axes[0], result['RawData'], 'Raw MCA Data', 'Counts')

    if result['livetime'] not in (None, 0):
        normalized = [value / result['livetime'] for value in result['RawData']]
        plot_series(axes[1], normalized, 'Normalized MCA Data (Counts/s)', 'Counts/s')
    else:
        normalized = result['RawData']
        axes[1].text(0.5, 0.5, 'Livetime is zero or missing', ha='center', va='center')
        axes[1].set_title('Normalized MCA Data')

    energy_axis = channel_to_energy(range(len(result['RawData'])), gain=10.0)
    plot_series(
        axes[2],
        normalized,
        'Normalized MCA Data vs Energy',
        'Counts/s',
        x_values=energy_axis,
        x_label='Energy (keV)',
    )

    if result['livetime'] not in (None, 0):
        fit_result = fit_log_linear(energy_axis, normalized)
        if fit_result is not None:
            fit_slope = fit_result['slope']
            fit_intercept = fit_result['intercept']
            fit_x = fit_result['x_fit']
            fit_y = fit_result['y_fit']
            fit_line = fit_slope * fit_x + fit_intercept
            log_mask = np.isfinite(energy_axis) & np.isfinite(normalized) & (np.asarray(normalized) > 0)
            axes[3].plot(
                np.asarray(energy_axis)[log_mask],
                np.log(np.asarray(normalized)[log_mask]),
                'o',
                color='0.75',
                markersize=2,
                label='log(data)',
            )
            axes[3].plot(fit_x, fit_y, 'o', markersize=3, label='selected fit range')
            axes[3].plot(fit_x, fit_line, '-', label=f'Fit: y = {fit_slope:.3g}x + {fit_intercept:.3g}')
            axes[3].set_xlabel('Energy (keV)')
            axes[3].set_ylabel('ln(counts/s)')
            axes[3].set_title('Log Spectrum with Linear Fit')
            axes[3].legend(loc='best')
            alpha = -fit_slope
            spectral_temperature = 1.0 / alpha
            output_lines.extend([
                f"Peak energy used to start search: {fit_result['peak_energy']}",
                f"Selected fit range: {fit_result['min_energy']} to {fit_result['max_energy']} keV",
                f"Selected fit points: {len(fit_x)}",
                f"Fit R^2: {fit_result['r_squared']}",
                f"Fit slope: {fit_slope}",
                f"Alpha (-slope): {alpha}",
                f"Fit intercept beta: {fit_intercept}",
                f"Spectral temperature Ts (1/alpha): {spectral_temperature}",
            ])
        else:
            axes[3].text(0.5, 0.5, 'Not enough positive data for fitting', ha='center', va='center')
            axes[3].set_title('Log Spectrum with Linear Fit')
            output_lines.append("Fit: Not enough positive data for fitting")
    else:
        axes[3].text(0.5, 0.5, 'Livetime is zero or missing', ha='center', va='center')
        axes[3].set_title('Log Spectrum with Linear Fit')
        output_lines.append("Fit: Livetime is zero or missing")

    plt.tight_layout()
    plot_path = output_dir / f"{Path(input_file).stem}_analysis.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    output_lines.append(f"Plot file: {plot_path.name}")
    return output_lines


def choose_input_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title='Select folder containing .mca files')
    root.destroy()
    return Path(folder) if folder else None


def main():
    parser = argparse.ArgumentParser(description='Parse all MCA files in a folder')
    parser.add_argument(
        'input_folder',
        nargs='?',
        help='Folder containing .mca files. If omitted, a folder picker opens.',
    )
    parser.add_argument(
        '--output',
        default='mca_analysis_results.txt',
        help='Name of the text report written to the selected folder',
    )
    args = parser.parse_args()

    if args.input_folder:
        input_folder = Path(args.input_folder)
    else:
        input_folder = choose_input_folder()

    if input_folder is None:
        return

    if not input_folder.is_dir():
        raise NotADirectoryError(f"{input_folder} is not a folder")

    mca_files = sorted(input_folder.glob('*.mca'))
    report_path = input_folder / args.output
    report_lines = [
        f"MCA analysis folder: {input_folder}",
        f"Number of .mca files: {len(mca_files)}",
        "",
    ]

    if not mca_files:
        report_lines.append("No .mca files found.")
    else:
        for mca_file in mca_files:
            report_lines.extend(analyze_mca_file(mca_file, input_folder))
            report_lines.append("")
            report_lines.append("-" * 72)
            report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding='utf-8')

    if not args.input_folder:
        messagebox.showinfo(
            'MCA analysis complete',
            f'Analyzed {len(mca_files)} .mca files.\n\nReport saved to:\n{report_path}',
        )


if __name__ == '__main__':
    main()

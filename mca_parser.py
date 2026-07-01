import argparse
import csv
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np


EFF_ENERGY_KEV = np.array([
    51.2, 54.7, 58.5, 62.5, 66.8, 71.5, 76.4, 81.7,
    87.3, 93.3, 99.8, 107, 114, 122, 130, 139, 149, 159,
    170, 182, 194, 208, 222, 237, 254, 271, 290, 310,
    332, 354, 379, 405,
], dtype=float)

EFF_ABS_PROBABILITY = np.array([
    0.994, 0.989, 0.98, 0.963, 0.938, 0.903, 0.859, 0.807,
    0.747, 0.684, 0.619, 0.556, 0.495, 0.439, 0.387, 0.34,
    0.299, 0.263, 0.231, 0.203, 0.18, 0.159, 0.142, 0.127,
    0.114, 0.103, 0.0937, 0.0856, 0.0787, 0.0726, 0.0674,
    0.0628,
], dtype=float)


def detector_absorption_efficiency(energy_values):
    energy = np.asarray(energy_values, dtype=float)
    return np.interp(
        energy,
        EFF_ENERGY_KEV,
        EFF_ABS_PROBABILITY,
        left=1.0,
        right=EFF_ABS_PROBABILITY[-1],
    )


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
    #if gain == 15.0:
        #calibration = {
        #    367: 30.973, #Ba-133
        #    381: 32.194, #Cs-137
        #    416: 34.987, #Ba-133
        #    430: 36.378, #Cs-137;
        #    960: 80.9971, #Ba-133;
        #}
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


def linear_fit_uncertainties(x_fit, y_fit, slope, intercept):
    predicted = slope * x_fit + intercept
    residuals = y_fit - predicted
    n = len(x_fit)
    degrees_of_freedom = n - 2
    sxx = np.sum((x_fit - np.mean(x_fit)) ** 2)

    if degrees_of_freedom <= 0 or sxx == 0:
        return None, None, None, None

    residual_variance = np.sum(residuals ** 2) / degrees_of_freedom
    residual_std_error = np.sqrt(residual_variance)
    slope_error = np.sqrt(residual_variance / sxx)
    intercept_error = np.sqrt(residual_variance * ((1.0 / n) + (np.mean(x_fit) ** 2 / sxx)))
    slope_intercept_covariance = -np.mean(x_fit) * residual_variance / sxx
    return slope_error, intercept_error, residual_std_error, slope_intercept_covariance


def fitted_line_error(x_values, fit_result):
    slope_error = fit_result['slope_error']
    intercept_error = fit_result['intercept_error']
    covariance = fit_result['slope_intercept_covariance']

    if slope_error is None or intercept_error is None or covariance is None:
        return None

    x = np.asarray(x_values, dtype=float)
    variance = (x ** 2 * slope_error ** 2) + (intercept_error ** 2) + (2.0 * x * covariance)
    return np.sqrt(np.maximum(variance, 0.0))


def find_most_linear_log_range(x_values, y_values, peak_window=(50.0, 120.0), max_energy=200.0):
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
            slope_error, intercept_error, residual_std_error, slope_intercept_covariance = linear_fit_uncertainties(
                x_fit,
                y_fit,
                slope,
                intercept,
            )
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
                    'slope_error': slope_error,
                    'intercept_error': intercept_error,
                    'residual_std_error': residual_std_error,
                    'slope_intercept_covariance': slope_intercept_covariance,
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
    slope_error, intercept_error, residual_std_error, slope_intercept_covariance = linear_fit_uncertainties(
        x_fit,
        y_fit,
        slope,
        intercept,
    )
    return {
        'score': r_squared,
        'slope': slope,
        'intercept': intercept,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'r_squared': r_squared,
        'mean_squared_error': mean_squared_error,
        'slope_error': slope_error,
        'intercept_error': intercept_error,
        'residual_std_error': residual_std_error,
        'slope_intercept_covariance': slope_intercept_covariance,
        'peak_energy': None,
        'min_energy': x_fit[0],
        'max_energy': x_fit[-1],
    }


CSV_COLUMNS = [
    'file',
    'livetime',
    'start_time',
    'gain',
    'gain_formula',
    'fast_count',
    'slow_count',
    'raw_data_length',
    'peak_energy_keV',
    'fit_range_min_keV',
    'fit_range_max_keV',
    'fit_points',
    'fit_r_squared',
    'fit_mean_squared_error',
    'fit_residual_std_error',
    'fit_slope',
    'fit_slope_error',
    'alpha',
    'alpha_error',
    'fit_intercept_beta',
    'fit_intercept_beta_error',
    'spectral_temperature',
    'spectral_temperature_error',
    'spectral_temperature_error_type',
    'plot_file',
    'fit_status',
]


def analyze_mca_file(input_file, output_dir):
    PLOT_MAX_ENERGY_KEV = 300.0

    result = parse_mca_file(input_file)
    row = {
        'file': result['file'],
        'livetime': result['livetime'],
        'start_time': result['start_time'],
        'gain': result['gain'],
        'gain_formula': result['gain_formula'],
        'fast_count': result['fast_count'],
        'slow_count': result['slow_count'],
        'raw_data_length': len(result['RawData']),
        'fit_status': '',
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # --- (0) Raw MCA Data plot: unchanged ---
    plot_series(axes[0], result['RawData'], 'Raw MCA Data', 'Counts')

    # --- normalize ---
    if result['livetime'] not in (None, 0):
        normalized = [value / result['livetime'] for value in result['RawData']]
        plot_title_note = f"(Livetime={result['livetime']})"
    else:
        normalized = result['RawData']
        axes[0].text(0.5, 0.02, 'Livetime is zero or missing', transform=axes[0].transAxes,
                     ha='center', va='bottom')
        plot_title_note = '(Livetime missing/zero)'

    # --- energy axis + efficiency-corrected ---
    energy_axis = channel_to_energy(range(len(result['RawData'])), gain=10.0)
    efficiency = detector_absorption_efficiency(energy_axis)

    normalized_array = np.asarray(normalized, dtype=float)
    corrected = np.divide(
        normalized_array,
        efficiency,
        out=np.full_like(normalized_array, np.nan, dtype=float),
        where=np.isfinite(efficiency) & (efficiency > 0),
    )

    energy_axis_array = np.asarray(energy_axis, dtype=float)
    in_plot_energy_mask = np.isfinite(energy_axis_array) & (energy_axis_array <= PLOT_MAX_ENERGY_KEV)

    # --- (1) NEW combined plot: normalized + corrected vs energy ---
    axes[1].plot(
        energy_axis_array[in_plot_energy_mask],
        normalized_array[in_plot_energy_mask],
        linewidth=1,
        label='Normalized (Counts/s)',
    )
    axes[1].plot(
        energy_axis_array[in_plot_energy_mask],
        corrected[in_plot_energy_mask],
        linewidth=1,
        label='Efficiency-corrected (Counts/s / absorption probability)',
    )
    axes[1].set_xlabel('Energy (keV)')
    axes[1].set_ylabel('Counts/s')
    axes[1].set_title(f'Normalized and Efficiency-Corrected Spectra vs Energy {plot_title_note}')
    axes[1].legend(loc='best')

    # Set x-range on energy plots to 300 keV max
    if np.any(in_plot_energy_mask):
        axes[1].set_xlim(float(np.nanmin(energy_axis_array[in_plot_energy_mask])), PLOT_MAX_ENERGY_KEV)
    else:
        axes[1].set_xlim(0, PLOT_MAX_ENERGY_KEV)

    # --- (2) log plot + fit: efficiency-corrected only, up to 300 keV ---
    if result['livetime'] not in (None, 0):
        fit_result = fit_log_linear(energy_axis, corrected)
        if fit_result is not None:
            fit_slope = fit_result['slope']
            fit_intercept = fit_result['intercept']
            fit_x = fit_result['x_fit']
            fit_y = fit_result['y_fit']
            fit_line = fit_slope * fit_x + fit_intercept
            fit_line_error = fitted_line_error(fit_x, fit_result)

            # Shade fit range on the combined plot
            fit_range_min = fit_x[0]
            fit_range_max = fit_x[-1]
            # Shade the raw-data (channel) plot corresponding to the same ENERGY fit range
            channels_axis = np.arange(len(energy_axis_array), dtype=float)
            raw_fit_mask = (energy_axis_array >= fit_range_min) & (energy_axis_array <= fit_range_max)

            if np.any(raw_fit_mask):
                raw_cmin = channels_axis[raw_fit_mask].min()
                raw_cmax = channels_axis[raw_fit_mask].max()
                axes[0].axvspan(
                    raw_cmin,
                    raw_cmax,
                    color='lightgreen',
                    alpha=0.25,
                    zorder=0,
            )
            axes[1].axvspan(
                fit_range_min,
                fit_range_max,
                color='0.85',
                alpha=0.6,
                zorder=0,
            )

            alpha = -fit_slope
            alpha_error = fit_result['slope_error']

            if alpha != 0:
                spectral_temperature = 1.0 / alpha
                if alpha_error is not None:
                    spectral_temperature_error = alpha_error / (alpha ** 2)
                else:
                    spectral_temperature_error = None
            else:
                spectral_temperature = None
                spectral_temperature_error = None

            corrected_mask = (
                np.isfinite(energy_axis_array)
                & np.isfinite(corrected)
                & (corrected > 0)
                & (energy_axis_array <= PLOT_MAX_ENERGY_KEV)
            )

            axes[2].plot(
                energy_axis_array[corrected_mask],
                np.log(corrected[corrected_mask]),
                'o',
                color='0.75',
                markersize=2,
                label='log(efficiency-corrected data)',
            )
            axes[2].plot(fit_x, fit_y, 'o', markersize=3, label='selected fit range')

            if fit_line_error is not None:
                axes[2].fill_between(
                    fit_x,
                    fit_line - fit_line_error,
                    fit_line + fit_line_error,
                    color='green',
                    alpha=0.3,
                    label='fit line 1-sigma band',
                )

            if spectral_temperature is not None and spectral_temperature_error is not None:
                fit_label = (
                    f'Fit: y = {fit_slope:.3g}x + {fit_intercept:.3g}; '
                    f'Ts = {spectral_temperature:.3g} +/- {spectral_temperature_error:.2g}'
                )
            else:
                fit_label = f'Fit: y = {fit_slope:.3g}x + {fit_intercept:.3g}'

            axes[2].plot(fit_x, fit_line, '-', label=fit_label)
            axes[2].text(
                0.02,
                0.95,
                f'Ts = {spectral_temperature:.4g} +/- {spectral_temperature_error:.2g}'
                if spectral_temperature is not None and spectral_temperature_error is not None
                else 'Ts error unavailable',
                transform=axes[2].transAxes,
                va='top',
            )

            axes[2].set_xlabel('Energy (keV)')
            axes[2].set_ylabel('ln(corrected counts/s)')
            axes[2].set_title('Log Efficiency-Corrected Spectrum with Linear Fit')
            axes[2].set_xlim(0, PLOT_MAX_ENERGY_KEV)
            axes[2].legend(loc='best')

            row.update({
                'peak_energy_keV': fit_result['peak_energy'],
                'fit_range_min_keV': fit_result['min_energy'],
                'fit_range_max_keV': fit_result['max_energy'],
                'fit_points': len(fit_x),
                'fit_r_squared': fit_result['r_squared'],
                'fit_mean_squared_error': fit_result['mean_squared_error'],
                'fit_residual_std_error': fit_result['residual_std_error'],
                'fit_slope': fit_slope,
                'fit_slope_error': fit_result['slope_error'],
                'alpha': alpha,
                'alpha_error': alpha_error,
                'fit_intercept_beta': fit_intercept,
                'fit_intercept_beta_error': fit_result['intercept_error'],
                'spectral_temperature': spectral_temperature,
                'spectral_temperature_error': spectral_temperature_error,
                'spectral_temperature_error_type': '1-sigma propagated from slope standard error',
                'fit_status': 'ok',
            })
        else:
            axes[2].text(0.5, 0.5, 'Not enough positive data for fitting', ha='center', va='center')
            axes[2].set_title('Log Efficiency-Corrected Spectrum with Linear Fit')
            axes[2].set_xlim(0, PLOT_MAX_ENERGY_KEV)
            row['fit_status'] = 'not enough positive data for fitting'
    else:
        axes[2].text(0.5, 0.5, 'Livetime is zero or missing', ha='center', va='center')
        axes[2].set_title('Log Efficiency-Corrected Spectrum with Linear Fit')
        axes[2].set_xlim(0, PLOT_MAX_ENERGY_KEV)
        row['fit_status'] = 'livetime is zero or missing'

    plt.tight_layout()
    plot_path = output_dir / f"{Path(input_file).stem}_analysis.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    row['plot_file'] = plot_path.name
    return row

def choose_input_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title='Select folder containing .mca files')
    root.destroy()
    return Path(folder) if folder else None


def main():
    parser = argparse.ArgumentParser(description='Parse all MCA files in a folder and write a CSV summary')
    parser.add_argument(
        'input_folder',
        nargs='?',
        help='Folder containing .mca files. If omitted, a folder picker opens.',
    )
    parser.add_argument(
        '--output',
        default='mca_analysis_results.csv',
        help='Name of the CSV report written to the selected folder',
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
    rows = []

    for mca_file in mca_files:
        rows.append(analyze_mca_file(mca_file, input_folder))

    with report_path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if not args.input_folder:
        messagebox.showinfo(
            'MCA analysis complete',
            f'Analyzed {len(mca_files)} .mca files.\n\nCSV saved to:\n{report_path}',
        )


if __name__ == '__main__':
    main()

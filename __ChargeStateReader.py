import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog


# Calibration anchors from the supplied table.
calibration_current = np.array([36.655, 52.018, 56.322, 60.168, 68.882, 73.568, 84.77, 103.293])
calibration_mq = np.array([1.0, 2.0, 2.29, 2.67, 3.20, 4.0, 5.33, 8.0])
oxygen_targets = [
    (8, 'O8+', 2.000),
    (7, 'O7+', 2.286),
    (6, 'O6+', 2.667),
    (5, 'O5+', 3.200),
    (4, 'O4+', 4.000),
    (3, 'O3+', 5.333),
    (2, 'O2+', 8.000),
]
BEAM_CURRENT_TO_MICROAMPS = 1e6


def calibrate_mq_from_batman(current_values):
    current_values = np.asarray(current_values, dtype=float)
    if np.any(current_values < calibration_current.min()) or np.any(current_values > calibration_current.max()):
        raise ValueError('BATMAN current contains values outside the calibration range')
    return np.interp(current_values, calibration_current, calibration_mq)


def load_numeric_data(filename):
    df = pd.read_csv(
        filename,
        sep=r'\s+',
        header=None,
        names=['timestamp', 'batman_current', 'batman_field', 'beam_current'],
        comment='#',
        engine='python',
    )

    for column in ['timestamp', 'batman_current', 'batman_field', 'beam_current']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=['timestamp', 'batman_current', 'batman_field', 'beam_current']).reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No numeric data rows found in {filename}')
    return df


def analyze_file(filename, plot=True):
    df = load_numeric_data(filename)

    calibration_mask = (
        df['batman_current'] >= calibration_current.min()
    ) & (
        df['batman_current'] <= calibration_current.max()
    )
    df = df.loc[calibration_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError('No data points fall within the calibration range')

    y = df['beam_current'].values * BEAM_CURRENT_TO_MICROAMPS

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['datetime_formatted'] = df['datetime'].dt.strftime('%d %b %Y %H:%M')

    first_timestamp = df['timestamp'].iloc[0]
    title_date = datetime.fromtimestamp(first_timestamp).strftime('%d %b %Y %H:%M')

    peak_prominence_uA = 1.0
    peaks, _ = find_peaks(y, prominence=peak_prominence_uA)
    df['mq_from_batman'] = calibrate_mq_from_batman(df['batman_current'].values)
    initial_x = df['mq_from_batman'].values

    # A broad physical peak can produce several nearby numerical maxima.
    # Keep only the tallest maximum within an M/Q separation that grows with M/Q.
    raw_peaks = peaks.copy()
    peak_separation = lambda mq: 0.14 + 0.02 * mq
    kept_peak_indices = []
    for peak_idx in sorted(raw_peaks, key=lambda index: y[index], reverse=True):
        if all(
            abs(initial_x[peak_idx] - initial_x[kept_idx])
            >= max(peak_separation(initial_x[peak_idx]), peak_separation(initial_x[kept_idx]))
            for kept_idx in kept_peak_indices
        ):
            kept_peak_indices.append(peak_idx)
    peaks = np.array(kept_peak_indices, dtype=int)

    selected_peak_indices = []
    selected_peak_data = {}
    selected_peak_labels = {}

    candidate_peak_indices = [peak_idx for peak_idx in peaks if 1.0 <= initial_x[peak_idx] <= 8.0]
    if not candidate_peak_indices:
        raise ValueError('No detected peaks fall within the physical M/Q range')

    target_labels = [label for _, label, _ in oxygen_targets]
    target_mq = np.array([target for _, _, target in oxygen_targets])
    strongest_peak_indices = sorted(
        sorted(candidate_peak_indices, key=lambda peak_idx: y[peak_idx], reverse=True)[:len(target_labels) - 1],
        key=lambda peak_idx: initial_x[peak_idx],
    )

    # The supplied calibration determines the provisional left-to-right M/Q
    # order; the six strongest peaks provide the O7+ through O2+ rescale.
    initial_assignments = dict(zip(target_labels[1:], strongest_peak_indices))
    if len(initial_assignments) >= 2:
        observed_mq = np.array([initial_x[peak_idx] for peak_idx in strongest_peak_indices])
        observed_targets = target_mq[1:len(strongest_peak_indices) + 1]
        rescale_slope, rescale_intercept = np.polyfit(
            observed_mq,
            observed_targets,
            1,
        )
        x = rescale_slope * initial_x + rescale_intercept
    else:
        rescale_slope = 1.0
        rescale_intercept = 0.0
        x = initial_x.copy()

    assigned_peaks = dict(zip(target_labels[1:], strongest_peak_indices))

    # O8+ can be weaker than background peaks, so select it by position rather
    # than strength: it must be left of the O7+ anchor after rescaling.
    o7_peak_idx = assigned_peaks['O7+']
    remaining_peak_indices = [
        peak_idx for peak_idx in candidate_peak_indices
        if peak_idx not in strongest_peak_indices
        and x[peak_idx] < x[o7_peak_idx]
    ]
    if remaining_peak_indices:
        assigned_peaks['O8+'] = min(
            remaining_peak_indices,
            key=lambda peak_idx: abs(x[peak_idx] - target_mq[0]),
        )

    # Emit selected peaks in the requested O8+ through O2+ order.
    for charge_state, label, target_mq in oxygen_targets:
        if label not in assigned_peaks:
            continue
        peak_idx = assigned_peaks[label]

        selected_peak_indices.append(peak_idx)
        selected_peak_data[label] = {
            'beam_at_peak': y[peak_idx],
            'batman_at_peak': df['batman_current'].iloc[peak_idx],
            'mq_at_peak': x[peak_idx],
            'charge_state': charge_state,
        }
        selected_peak_labels[peak_idx] = label

    result_row = {
        'file': Path(filename).name,
        'date': df['datetime'].dt.strftime('%Y-%m-%d').iloc[0],
        'time': df['datetime'].dt.strftime('%H:%M:%S').iloc[0],
    }
    for _, label, _ in oxygen_targets:
        peak_data = selected_peak_data.get(label, {})
        result_row[f'{label}_beam_current'] = peak_data.get('beam_at_peak', np.nan)
        result_row[f'{label}_batman_current'] = peak_data.get('batman_at_peak', np.nan)
        result_row[f'{label}_mq'] = peak_data.get('mq_at_peak', np.nan)

    selected_table = pd.DataFrame([
        {
            'charge_state': label,
            'M/Q': selected_peak_data[label]['mq_at_peak'],
            'BATMAN_current': selected_peak_data[label]['batman_at_peak'],
            'beam_current_uA': selected_peak_data[label]['beam_at_peak'],
        }
        for _, label, _ in oxygen_targets
        if label in selected_peak_data
    ])
    print(f'\nSelected peaks for {Path(filename).name}:')
    print(selected_table.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print(f'Oxygen M/Q rescale: M/Q_corrected = {rescale_slope:.6f} * M/Q_initial + {rescale_intercept:.6f}')

    if plot:
        fig, (ax, calibration_ax) = plt.subplots(1, 2, figsize=(16, 6))
        ax.plot(x, y, 'g-', label='Data')
        ax.scatter(
            x[peaks],
            y[peaks],
            color='0.55',
            s=35,
            zorder=2,
            label=f'All detected peaks (n={len(peaks)} of {len(raw_peaks)})',
        )
        ax.plot(
            x[selected_peak_indices],
            y[selected_peak_indices],
            'r*',
            markersize=11,
            zorder=3,
            label=f'Selected oxygen peaks (n={len(selected_peak_indices)})',
        )
        for peak_idx in selected_peak_indices:
            label = selected_peak_labels.get(peak_idx, '')
            ax.annotate(
                f'{label}\nM/Q={x[peak_idx]:.3f}\n{y[peak_idx]:.1f} uA',
                (x[peak_idx], y[peak_idx]),
                textcoords='offset points',
                xytext=(0, 8),
                ha='center',
                fontsize=8,
            )
        ax.set_xlabel('M/Q')
        ax.set_ylabel('Beam Current (µA)')
        ax.set_xlim(1.0, 8.2)
        ax.set_title(f'{Path(filename).name} - {title_date}')
        ax.legend()

        calibration_current_range = np.linspace(
            calibration_current.min(),
            calibration_current.max(),
            300,
        )
        calibration_ax.plot(
            calibration_current_range,
            calibrate_mq_from_batman(calibration_current_range),
            color='steelblue',
            linestyle='--',
            label='Supplied calibration',
        )
        calibration_ax.plot(
            calibration_current_range,
            rescale_slope * calibrate_mq_from_batman(calibration_current_range) + rescale_intercept,
            color='black',
            linewidth=1.5,
            label='Oxygen-rescaled calibration',
        )
        calibration_ax.scatter(
            calibration_current,
            calibration_mq,
            color='darkorange',
            edgecolors='black',
            zorder=3,
            label='Calibration points',
        )
        calibration_ax.scatter(
            df['batman_current'].iloc[peaks],
            x[peaks],
            color='0.55',
            s=30,
            label='Detected peaks',
        )
        calibration_ax.scatter(
            df['batman_current'].iloc[selected_peak_indices],
            x[selected_peak_indices],
            color='red',
            marker='*',
            s=100,
            zorder=4,
            label='Selected peaks',
        )
        for peak_idx in selected_peak_indices:
            calibration_ax.annotate(
                selected_peak_labels[peak_idx],
                (df['batman_current'].iloc[peak_idx], x[peak_idx]),
                textcoords='offset points',
                xytext=(4, 5),
                fontsize=8,
            )
        calibration_ax.set_xlabel('BATMAN Current')
        calibration_ax.set_ylabel('M/Q')
        calibration_ax.set_title('Calibration and selected peak locations')
        calibration_ax.grid(True, linestyle='--', alpha=0.6)
        calibration_ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    return result_row, selected_peak_data


def resolve_input_files(input_paths):
    ignored_names = {'peaks.txt', 'AxialField.txt'}
    files = []

    def is_supported_file(path):
        if not path.is_file():
            return False
        if path.name in ignored_names:
            return False
        if path.name.startswith('combined_oxygen_peak_results'):
            return False
        if path.name.startswith('csd_'):
            return True
        return path.suffix.lower() in {'.txt', '.dat', '.csv'}

    if input_paths:
        for raw_path in input_paths:
            path = Path(raw_path)
            if path.exists() and path.is_dir():
                matched = sorted(
                    [p for p in path.iterdir() if p.is_file() and is_supported_file(p) and p.name.startswith('csd_')]
                )
                if not matched:
                    matched = sorted([p for p in path.iterdir() if p.is_file() and is_supported_file(p)])
                if matched:
                    files.extend(matched)
                else:
                    files.append(path / 'data.txt')
            elif path.exists() and path.is_file():
                files.append(path)
            else:
                matches = sorted(Path('.').glob(raw_path))
                if matches:
                    files.extend([match for match in matches if is_supported_file(match)])
                elif '*' in raw_path or '?' in raw_path or '[' in raw_path:
                    continue
                else:
                    files.append(path)
    else:
        csd_files = sorted([path for path in Path('.').iterdir() if is_supported_file(path) and path.name.startswith('csd_')])
        if csd_files:
            files = csd_files
        else:
            files = sorted([path for path in Path('.').iterdir() if is_supported_file(path)])
        if not files:
            files = [Path('data.txt')]

    return files

def build_output_suffix(results_df, input_paths):
    """Suffix for output filenames, taken from the analyzed data's date."""
    if 'date' in results_df.columns:
        dates = pd.to_datetime(results_df['date'], errors='coerce').dropna()
        if not dates.empty:
            first, last = dates.min(), dates.max()
            if first.date() == last.date():
                return first.strftime('%m-%d-%y')
            return f"{first.strftime('%m-%d-%y')}_to_{last.strftime('%m-%d-%y')}"

    # Fall back to the folder name, then to today's date.
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_dir():
            return path.name

    return datetime.now().strftime('%m-%d-%y')

def main():
    if len(sys.argv) > 1:
        input_paths = sys.argv[1:]
    else:
        input_paths = []
        root = tk.Tk()
        root.withdraw()
        selected_folder = filedialog.askdirectory(title='Select folder to analyze')
        root.destroy()
        if selected_folder:
            input_paths = [selected_folder]

    input_files = resolve_input_files(input_paths)
    print('Using oxygen-charge-state-based M/Q calibration')

    summaries = []
    for file_path in input_files:
        try:
            row, _ = analyze_file(file_path, plot=True)
            summaries.append(row)
        except Exception as exc:
            print(f'Skipping {file_path}: {exc}')

    results_df = pd.DataFrame(summaries)
    if 'file' in results_df.columns:
        results_df = results_df.set_index('file')

    ordered_columns = ['date', 'time']
    for _, label, _ in oxygen_targets:
        ordered_columns.extend([f'{label}_beam_current', f'{label}_batman_current', f'{label}_mq'])
    existing_columns = [col for col in ordered_columns if col in results_df.columns]
    results_df = results_df[existing_columns]

    output_suffix = build_output_suffix(results_df, input_paths)
    output_csv = Path(f'combined_oxygen_peak_results_{output_suffix}.csv')
    output_txt = Path(f'combined_oxygen_peak_results_{output_suffix}.txt')

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        if 'file' in existing_df.columns:
            existing_df = existing_df.set_index('file')
        if not results_df.index.name:
            results_df.index.name = 'file'
        combined_df = pd.concat([existing_df, results_df], axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    else:
        combined_df = results_df

    combined_df.to_csv(output_csv)
    output_txt.write_text(combined_df.to_string())

    print(f'Processed {len(input_files)} file(s).')
    print(results_df)


if __name__ == '__main__':
    main()

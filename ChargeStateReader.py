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


def calibrate_mq_from_batman(current_values):
    current_values = np.asarray(current_values, dtype=float)
    out = np.empty_like(current_values)

    low_mask = current_values <= calibration_current[1]
    if np.any(low_mask):
        low_slope = (calibration_mq[1] - calibration_mq[0]) / (calibration_current[1] - calibration_current[0])
        low_intercept = calibration_mq[0] - low_slope * calibration_current[0]
        out[low_mask] = low_slope * current_values[low_mask] + low_intercept

    high_mask = ~low_mask
    if np.any(high_mask):
        out[high_mask] = np.interp(current_values[high_mask], calibration_current[1:], calibration_mq[1:])

    return out


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


def analyze_file(filename, plot=False):
    df = load_numeric_data(filename)
    df['mq_from_batman'] = calibrate_mq_from_batman(df['batman_current'].values)

    x = df['mq_from_batman'].values
    y = df['beam_current'].values

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['datetime_formatted'] = df['datetime'].dt.strftime('%d %b %Y %H:%M')

    first_timestamp = df['timestamp'].iloc[0]
    title_date = datetime.fromtimestamp(first_timestamp).strftime('%d %b %Y %H:%M')

    peaks, _ = find_peaks(y, prominence=0.000001)

    selected_peak_indices = []
    selected_peak_data = {}
    selected_peak_labels = {}
    remaining_peak_indices = list(peaks)

    for charge_state, label, target_mq in oxygen_targets:
        best_peak_idx = None
        best_distance = None

        for peak_idx in remaining_peak_indices:
            distance = abs(x[peak_idx] - target_mq)
            if best_peak_idx is None or distance < best_distance:
                best_peak_idx = peak_idx
                best_distance = distance

        if best_peak_idx is not None:
            selected_peak_indices.append(best_peak_idx)
            selected_peak_data[label] = {
                'beam_at_peak': y[best_peak_idx],
                'batman_at_peak': df['batman_current'].iloc[best_peak_idx],
                'mq_at_peak': x[best_peak_idx],
                'charge_state': charge_state,
            }
            selected_peak_labels[best_peak_idx] = f'O{charge_state}+'
            remaining_peak_indices.remove(best_peak_idx)

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

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'g-', label='Data')
        ax.plot(x[selected_peak_indices], y[selected_peak_indices], 'r*', markersize=8, label='Oxygen charge state')
        for peak_idx in selected_peak_indices:
            label = selected_peak_labels.get(peak_idx, '')
            ax.annotate(label, (x[peak_idx], y[peak_idx]), textcoords='offset points', xytext=(0, 8), ha='center')
        ax.set_xlabel('M/Q')
        ax.set_ylabel('Beam Current (µA)')
        ax.set_title(title_date)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return result_row, selected_peak_data


def resolve_input_files(input_paths):
    ignored_names = {'peaks.txt', 'AxialField.txt', 'combined_oxygen_peak_results.csv', 'combined_oxygen_peak_results.txt'}
    files = []

    def is_supported_file(path):
        if not path.is_file():
            return False
        if path.name in ignored_names:
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
            row, _ = analyze_file(file_path, plot=len(input_files) == 1)
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

    output_csv = Path('combined_oxygen_peak_results.csv')
    output_txt = Path('combined_oxygen_peak_results.txt')

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

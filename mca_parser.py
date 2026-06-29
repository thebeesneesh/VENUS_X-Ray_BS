import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


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
        gain_formula = f'{coarse_gain} × {fine_gain}'
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


def plot_series(ax, values, title, ylabel):
    x = list(range(len(values)))
    ax.plot(x, values, linewidth=1)
    ax.set_xlabel('Channel')
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description='Parse an MCA file and plot its data')
    parser.add_argument('input_file', help='Path to the .mca file')
    args = parser.parse_args()

    result = parse_mca_file(args.input_file)

    print(f"File: {result['file']}")
    print(f"Livetime: {result['livetime']}")
    print(f"Start time: {result['start_time']}")
    if result['gain_formula']:
        print(f"Gain: {result['gain']}  (calculated from {result['gain_formula']})")
    else:
        print(f"Gain: {result['gain']}")
    print(f"Fast count: {result['fast_count']}")
    print(f"Slow count: {result['slow_count']}")
    print(f"RawData length: {len(result['RawData'])}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_series(axes[0], result['RawData'], 'Raw MCA Data', 'Counts')

    if result['livetime'] not in (None, 0):
        normalized = [value / result['livetime'] for value in result['RawData']]
        plot_series(axes[1], normalized, 'Normalized MCA Data (Counts/s)', 'Counts/s')
    else:
        axes[1].text(0.5, 0.5, 'Livetime is zero or missing', ha='center', va='center')
        axes[1].set_title('Normalized MCA Data')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

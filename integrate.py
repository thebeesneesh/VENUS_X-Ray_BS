import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks

# Read the text file
filename = 'data.txt'

df = pd.read_csv(filename, sep='\s+', header=None, 
                 names=['timestamp', 'batman_current', 'col3', 'beam_current'])

# Get the first timestamp and convert to readable format
first_timestamp = df['timestamp'].iloc[0]
title_date = datetime.fromtimestamp(first_timestamp).strftime('%d %b %Y %H:%M')

# Extract the data
x = df['beam_current'].values
y = df['batman_current'].values

# Find peaks in batman_current (adjust parameters as needed)
peaks, _ = find_peaks(y, prominence=0.01)

# Find valleys (local minima) for peak boundaries
valleys, _ = find_peaks(-y)
valleys = np.concatenate([[0], valleys, [len(y)-1]])

# Calculate area under each peak
peak_data = []

for peak_idx in peaks:
    # Find valleys on either side of peak
    left_valleys = valleys[valleys < peak_idx]
    right_valleys = valleys[valleys > peak_idx]
    
    left_bound = left_valleys[-1] if len(left_valleys) > 0 else 0
    right_bound = right_valleys[0] if len(right_valleys) > 0 else len(y) - 1
    
    # Calculate area using trapezoidal integration
    peak_x = x[left_bound:right_bound+1]
    peak_y = y[left_bound:right_bound+1]
    area = np.trapz(peak_y, peak_x)
    
    peak_data.append({
        'batman_at_peak': y[peak_idx],
        'beam_at_peak': x[peak_idx],
        'area': area
    })

# Save to text file
with open('peak_areas.txt', 'w') as f:
    f.write('BATMAN_Current_at_Peak(A)\tArea\n')
    for p in peak_data:
        f.write(f"{p['batman_at_peak']}\t{p['area']}\n")

print(f"Found {len(peaks)} peaks. Results saved to peak_areas.txt")

# Create the plot with peaks marked
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b.-', label='Data')
plt.plot(x[peaks], y[peaks], 'ro', markersize=8, label='Peaks')

plt.xlabel('Beam Current (µA)')
plt.ylabel('BATMAN Current (A)')
plt.title(title_date)
plt.legend()

plt.tight_layout()
plt.savefig('current_plot.png', dpi=150)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks


# Read the text file (adjust separator if needed - common ones are '\t', ',', or ' ')
# Change 'data.txt' to your filename
filename = 'data.txt'

# Read the file - adjust sep parameter based on your file's delimiter
df = pd.read_csv(filename, sep='\s+', header=None, names=['timestamp', 'batman_current', 'batman_field', 'beam_current'])

x = df['batman_current'].values
y = df['beam_current'].values

# Convert Unix timestamp to readable datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Format as "day month year hour" for display
df['datetime_formatted'] = df['datetime'].dt.strftime('%d %b %Y %H:%M')

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot BATMAN current on x-axis
color1 = 'tab:blue'
color2 = 'tab:red'
ax1.set_xlabel('BATMAN Current (A)', color=color1)
ax1.set_ylabel('Beam Current (µA)', color=color2)
ax1.plot(df['batman_current'], df['beam_current'], color=color1, label='Beam Current')
ax1.tick_params(axis='y', labelcolor=color1)

# Get the first timestamp and convert to readable format
first_timestamp = df['timestamp'].iloc[0]
title_date = datetime.fromtimestamp(first_timestamp).strftime('%d %b %Y %H:%M')

# Add title and legend
#plt.title(title_date)
#fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

#plt.tight_layout()
#plt.savefig('current_plot.png', dpi=150)
#plt.show()

# Find peaks in beam_current (adjust prominence as needed (increase if too many peaks, decrease if not enough))
peaks, _ = find_peaks(y, prominence=0.000001)

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
    area = np.trapezoid(peak_y, peak_x)
    
    peak_data.append({
        'beam_at_peak': y[peak_idx],
        'batman_at_peak': x[peak_idx],
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
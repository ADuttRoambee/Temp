import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Read the original data
df = pd.read_csv('processed_data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by linear_time to ensure proper order
df = df.sort_values('linear_time')

# Create normalized time (0, 1, 2, ...) for original data
df['normalized_time'] = range(len(df))

# Initialize lists for the interpolated data
interpolated_times = []
interpolated_temps = []
interpolated_timestamps = []
interpolated_norm_times = []  # For normalized time

# Loop through the original data points
for i in range(len(df) - 1):
    # Get current and next point
    t1, t2 = df['linear_time'].iloc[i], df['linear_time'].iloc[i + 1]
    temp1, temp2 = df['temperature'].iloc[i], df['temperature'].iloc[i + 1]
    ts1, ts2 = df['timestamp'].iloc[i], df['timestamp'].iloc[i + 1]
    norm_t1, norm_t2 = i, i + 1
    
    # Add the current point
    interpolated_times.append(t1)
    interpolated_temps.append(temp1)
    interpolated_timestamps.append(ts1)
    interpolated_norm_times.append(norm_t1)
    
    # Add 5 points between current and next point
    for j in range(1, 6):
        # Calculate fraction (1/6, 2/6, 3/6, 4/6, 5/6)
        frac = j / 6
        
        # Linear interpolation
        new_time = t1 + (t2 - t1) * frac
        new_temp = temp1 + (temp2 - temp1) * frac
        time_diff = (ts2 - ts1).total_seconds() * frac
        new_timestamp = ts1 + timedelta(seconds=time_diff)
        new_norm_time = norm_t1 + frac
        
        interpolated_times.append(new_time)
        interpolated_temps.append(round(new_temp, 1))  # Round to 1 decimal
        interpolated_timestamps.append(new_timestamp)
        interpolated_norm_times.append(new_norm_time)

# Add the last point
interpolated_times.append(df['linear_time'].iloc[-1])
interpolated_temps.append(df['temperature'].iloc[-1])
interpolated_timestamps.append(df['timestamp'].iloc[-1])
interpolated_norm_times.append(len(df) - 1)

# Create new dataframe with interpolated data
dense_df = pd.DataFrame({
    'timestamp': interpolated_timestamps,
    'temperature': interpolated_temps,
    'linear_time': interpolated_times,
    'normalized_time': interpolated_norm_times
})

# Create the plot
plt.figure(figsize=(15, 8))

# Plot interpolated data with larger blue dots
plt.plot(dense_df['normalized_time'], dense_df['temperature'], 
         'b.', alpha=0.8, markersize=8, label='Interpolated Points')

# Plot original data points with original size red dots
plt.plot(df['normalized_time'], df['temperature'], 
         'ro', markersize=3, alpha=0.7, label='Original Data')

plt.title('Temperature vs Normalized Time')
plt.xlabel('Time Units')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend(markerscale=1)

# Save the plot with higher DPI for better quality
plt.savefig('temperature_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Save to CSV with only temperature and normalized time
output_df = pd.DataFrame({
    'normalized_time': dense_df['normalized_time'],
    'temperature': dense_df['temperature']
})
output_df.to_csv('temperature_vs_time.csv', index=False)

print(f"Original data points: {len(df)}")
print(f"Interpolated data points: {len(dense_df)}")
print("Data saved to 'temperature_vs_time.csv'")
print("Plot saved as 'temperature_vs_time.png'") 
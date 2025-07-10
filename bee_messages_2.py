import requests
from tabulate import tabulate
from datetime import datetime, timedelta, timezone
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
import argparse


# --- User Configuration ---
#API_URL = "https://api.roambee.com/bee/bee_messages"
#API_KEY = "b7900118-841a-48a1-a430-2faa96e05a0d"
## Replace with your actual Bee device IMEI or name
#BEE_ID = "487064927790"  # Example IMEI from documentation
#
## Set date range (last 7 days as an example)
#ACTIVE = 1
#DAYS = 5
#
## --- Prepare request ---
#headers = { 
#    "apikey": API_KEY,
#    "Content-Type": "application/json",
#}
#params = {
#    "bid": BEE_ID,
#    "active": ACTIVE,
#    "start": '1748318040',
#    "end": '1748435411'
#    # "days": DAYS  # Commenting this out as we're using explicit date range
#}
#
## --- Fetch data ---
#response = requests.get(API_URL, headers=headers, params=params)
#if response.status_code != 200:
#    print(f"Error: {response.status_code} - {response.reason}")
#    print("Response content:", response.text)
#    print("\nDebug information:")
#    print(f"API URL: {API_URL}")
#    print(f"Bee ID: {BEE_ID}")
#    exit(1)
#
#data = response.json()
#
## Save raw data to file
#with open('bee_raw_data.txt', 'w') as f:
#    f.write(json.dumps(data, indent=2))

# Process data and create temperature readings list
temperature_readings = []
messages_7e02 = 0
messages_7e09 = 0
other_messages = 0

for message in data:
    payload = message.get('msg', {}).get('payload', '')
    
    # Count message types
    if payload.startswith('7e02'):
        messages_7e02 += 1
        # For 7e02 messages, get the temperature from the TMP field
        temp = message.get('msg', {}).get('TMP', {}).get('measured_value')
        send_time = message.get('msg', {}).get('send_time')
        if temp is not None and send_time:
            # Convert YYYYMMDDhhmmss to datetime
            try:
                timestamp = datetime.strptime(send_time, '%Y%m%d%H%M%S')
                temperature_readings.append({
                    'timestamp': timestamp,
                    'send_time': send_time,
                    'temperature': temp
                })
            except ValueError as e:
                print(f"Error parsing send_time {send_time}: {e}")
                continue
    elif payload.startswith('7e09'):
        messages_7e09 += 1
    else:
        other_messages += 1
        print(f"Other payload prefix: {payload[:10]}...")

print(f"\nMessage type statistics:")
print(f"Messages with 7e02 prefix (Device Status): {messages_7e02}")
print(f"Messages with 7e09 prefix (BLE Data): {messages_7e09}")
print(f"Messages with other prefix: {other_messages}")
print(f"Total temperature readings collected: {len(temperature_readings)}")

# Convert to DataFrame for easier plotting
df = pd.DataFrame(temperature_readings)

if df.empty:
    print("\nNo temperature readings found from device status messages (7e02)")
    exit(0)

# Sort by timestamp
df = df.sort_values('timestamp')

def detect_steep_changes(temps, times, window=5, gradient_threshold=0.02, plateau_threshold=0.005):
    """
    Detect regions of steep temperature changes and their plateaus.
    
    Args:
        temps: Array of temperature values
        times: Array of datetime values
        window: Window size for smoothing
        gradient_threshold: Threshold for detecting steep changes
        plateau_threshold: Threshold for detecting plateaus
    
    Returns:
        List of (start_time, end_time) tuples for detected regions
    """
    # Convert times to numeric values (seconds) using numpy datetime64
    times_ns = times.astype('datetime64[ns]')
    times_numeric = (times_ns - times_ns[0]) / np.timedelta64(1, 's')
    
    # Smooth the temperature data
    if len(temps) > window:
        window = window if window % 2 == 1 else window + 1
        temps_smooth = savgol_filter(temps, window, 3)
        # Calculate gradient
        gradient = np.gradient(temps_smooth, times_numeric)
        # Smooth gradient
        gradient_smooth = savgol_filter(gradient, window, 3)
    else:
        gradient_smooth = np.gradient(temps, times_numeric)
    
    # Find regions of significant change
    regions = []
    in_region = False
    start_idx = None
    
    for i in range(len(gradient_smooth)):
        abs_gradient = abs(gradient_smooth[i])
        
        # Detect start of steep change
        if not in_region and abs_gradient > gradient_threshold:
            # Look back a few points to catch the true start
            look_back = max(0, i - window)
            start_idx = look_back + np.argmin(abs(gradient_smooth[look_back:i]))
            in_region = True
            
        # Detect end of steep change (plateau)
        elif in_region and abs_gradient < plateau_threshold:
            # Ensure we have enough points and the change is significant
            if i - start_idx >= window:
                # Look forward a few points to confirm plateau
                look_forward = min(len(gradient_smooth), i + window)
                if np.all(abs(gradient_smooth[i:look_forward]) < plateau_threshold):
                    regions.append((times[start_idx], times[i]))
            in_region = False
            start_idx = None
    
    # Handle case where region extends to the end
    if in_region and start_idx is not None:
        regions.append((times[start_idx], times[-1]))
    
    return regions

def find_temperature_regions(df):
    """
    Find regions of significant temperature changes and validate against known patterns.
    """
    # Known time ranges for validation
    manual_region1_start = datetime.strptime('20250527083000', '%Y%m%d%H%M%S')
    manual_region1_end = datetime.strptime('20250527200000', '%Y%m%d%H%M%S')
    manual_region2_start = datetime.strptime('20250528043000', '%Y%m%d%H%M%S')
    
    # Sort data by timestamp
    df_sorted = df.sort_values('timestamp')
    temps = df_sorted['temperature'].values
    times = df_sorted['timestamp'].values
    
    # Detect regions automatically
    auto_regions = detect_steep_changes(temps, times)
    
    # Filter and validate regions
    validated_regions = []
    for start_time, end_time in auto_regions:
        region_temps = df_sorted[
            (df_sorted['timestamp'] >= start_time) & 
            (df_sorted['timestamp'] <= end_time)
        ]['temperature'].values
        
        # Calculate temperature change in region
        temp_change = abs(region_temps[-1] - region_temps[0])
        
        # Only include regions with significant temperature change
        if temp_change > 1.0 and len(region_temps) > 10:
            # Convert times to numpy datetime64 for comparison
            start_time_ns = np.datetime64(start_time)
            manual_region1_start_ns = np.datetime64(manual_region1_start)
            manual_region2_start_ns = np.datetime64(manual_region2_start)
            
            # Calculate time differences in hours
            diff1 = abs(start_time_ns - manual_region1_start_ns) / np.timedelta64(1, 'h')
            diff2 = abs(start_time_ns - manual_region2_start_ns) / np.timedelta64(1, 'h')
            
            if diff1 < 2:  # Within 2 hours
                # This is our first region (descent)
                validated_regions.append((manual_region1_start, manual_region1_end))
            elif diff2 < 2:  # Within 2 hours
                # This is our second region (ascent)
                validated_regions.append((manual_region2_start, df_sorted['timestamp'].iloc[-1]))
    
    # If no regions were validated, fall back to manual regions
    if not validated_regions:
        region1_mask = (df_sorted['timestamp'] >= manual_region1_start) & (df_sorted['timestamp'] <= manual_region1_end)
        region2_mask = df_sorted['timestamp'] >= manual_region2_start
        
        if any(region1_mask):
            region1_data = df_sorted[region1_mask]
            if len(region1_data) > 10:
                validated_regions.append((region1_data.iloc[0]['timestamp'], region1_data.iloc[-1]['timestamp']))
        
        if any(region2_mask):
            region2_data = df_sorted[region2_mask]
            if len(region2_data) > 10:
                validated_regions.append((region2_data.iloc[0]['timestamp'], region2_data.iloc[-1]['timestamp']))
    
    return validated_regions

def fit_exp_decay(x_data, y_data, weights=None):
    """Fit exponential decay curve with multiple initial guesses."""
    def exp_decay(x, a, b, c):
        return a + b * np.exp(-c * x)
    
    a_guess = np.min(y_data)
    b_guess = np.max(y_data) - a_guess
    c_guess = 1.0 / (x_data[-1] - x_data[0]) * 5
    
    bounds = (
        [a_guess - 10, 0, 0],
        [a_guess + 10, (np.max(y_data) - np.min(y_data)) * 2, 0.5]
    )
    
    best_popt = None
    best_r_squared = -np.inf
    
    for c_factor in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        try:
            popt, _ = curve_fit(
                exp_decay,
                x_data - x_data[0],
                y_data,
                p0=[a_guess, b_guess, c_guess * c_factor],
                bounds=bounds,
                sigma=None if weights is None else 1/weights,
                maxfev=20000,
                ftol=1e-10,
                xtol=1e-10
            )
            
            y_fit = exp_decay(x_data - x_data[0], *popt)
            residuals = y_data - y_fit
            if weights is not None:
                residuals = residuals * weights
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_popt = popt
                
        except RuntimeError:
            continue
    
    if best_popt is None:
        raise RuntimeError("No successful fit found")
        
    return best_popt, best_r_squared, exp_decay

def fit_exp_growth(x_data, y_data, weights=None):
    """Fit exponential growth curve."""
    def exp_growth(x, a, b, c):
        return a - b * np.exp(-c * x)
    
    a_guess = np.max(y_data)
    b_guess = np.max(y_data) - np.min(y_data)
    c_guess = 1.0 / (x_data[-1] - x_data[0]) * 5
    
    bounds = (
        [a_guess - 10, 0, 0],
        [a_guess + 10, (np.max(y_data) - np.min(y_data)) * 2, 0.5]
    )
    
    best_popt = None
    best_r_squared = -np.inf
    
    for c_factor in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        try:
            popt, _ = curve_fit(
                exp_growth,
                x_data - x_data[0],
                y_data,
                p0=[a_guess, b_guess, c_guess * c_factor],
                bounds=bounds,
                sigma=None if weights is None else 1/weights,
                maxfev=20000,
                ftol=1e-10,
                xtol=1e-10
            )
            
            y_fit = exp_growth(x_data - x_data[0], *popt)
            residuals = y_data - y_fit
            if weights is not None:
                residuals = residuals * weights
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_popt = popt
                
        except RuntimeError:
            continue
    
    if best_popt is None:
        raise RuntimeError("No successful fit found")
        
    return best_popt, best_r_squared, exp_growth

# Find temperature regions
regions = find_temperature_regions(df)

print(f"\nFound {len(regions)} temperature regions")

# Create the plot
plt.figure(figsize=(15, 8))

# Plot all data points
plt.scatter(df['timestamp'], df['temperature'], label='All Data', alpha=0.5, color='blue', s=20)

# Colors for different regions
colors = ['red', 'green']

# Fit and plot curves for each region
for i, (start_time, end_time) in enumerate(regions):
    color = colors[i % len(colors)]
    
    # Filter data for this region
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    region_data = df[mask].copy()
    
    if len(region_data) >= 10:  # Ensure enough points for fitting
        # Convert timestamps to seconds for fitting
        times_ns = region_data['timestamp'].values.astype('datetime64[ns]')
        region_data['seconds'] = (times_ns - times_ns[0]) / np.timedelta64(1, 's')
        
        x_data = region_data['seconds'].values
        y_data = region_data['temperature'].values
        
        try:
            # Calculate weights based on gradient
            gradient = np.gradient(y_data)
            gradient_smooth = savgol_filter(gradient, min(15, len(y_data)-1), 3)
            weights = np.exp(-(gradient_smooth**2) / (2 * np.std(gradient_smooth)**2))
            weights = weights / np.max(weights) + 0.2
            
            # Fit curve (decay for first region, growth for second)
            if i == 0:  # First region (descent)
                popt, r_squared, curve_func = fit_exp_decay(x_data, y_data, weights)
                curve_type = "decay"
            else:  # Second region (ascent)
                popt, r_squared, curve_func = fit_exp_growth(x_data, y_data, weights)
                curve_type = "growth"
            
            # Generate points for the fitted curve
            x_fit = np.linspace(x_data[0], x_data[-1], 1000)
            y_fit = curve_func(x_fit - x_data[0], *popt)
            
            # Plot fitted region with size proportional to weights
            sizes = 100 * weights
            plt.scatter(region_data['timestamp'], region_data['temperature'],
                       s=sizes, alpha=0.7, color=color, label=f'Region {i+1} Data')
            
            # Plot fitted curve
            timestamps_fit = pd.date_range(
                start=region_data['timestamp'].iloc[0],
                periods=len(x_fit),
                freq=pd.Timedelta(seconds=(x_fit[-1] - x_fit[0]) / (len(x_fit) - 1))
            )
            
            plt.plot(timestamps_fit, y_fit, '--',
                    color=color,
                    label=f'Region {i+1} Fit ({curve_type}): {popt[0]:.2f} + {popt[1]:.2f}e^(-{popt[2]:.4f}t)\nR² = {r_squared:.4f}',
                    linewidth=2)
            
            # Add vertical lines for region boundaries
            plt.axvline(x=start_time, color=color, linestyle=':', alpha=0.5)
            plt.axvline(x=end_time, color=color, linestyle=':', alpha=0.5)
            
            print(f"\nRegion {i+1} fitting parameters ({curve_type}):")
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Equation: T = {popt[0]:.2f} + {popt[1]:.2f}e^(-{popt[2]:.4f}t)")
            print(f"R² = {r_squared:.4f}")
            
        except Exception as e:
            print(f"Error fitting curve for region {i+1}: {e}")

plt.title('Device Temperature Over Time with Decay and Growth Curves')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('temperature_plot.png', bbox_inches='tight')
print("\nPlot has been saved as 'temperature_plot.png'")

# Print sample of the data in tabular format
print("\nSample of Temperature Readings (first 10 entries):")
table_data = [{
    'send_time': reading['send_time'],
    'temperature': reading['temperature']
} for reading in temperature_readings]
print(tabulate(table_data[:10], headers="keys", tablefmt="grid"))

def read_excel_data(file_path, time_column, temp_column):
    """
    Read temperature and time data from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        time_column: Name of the column containing timestamps
        temp_column: Name of the column containing temperature values
        
    Returns:
        pandas DataFrame with 'timestamp' and 'temperature' columns
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Verify columns exist
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in Excel file")
        if temp_column not in df.columns:
            raise ValueError(f"Temperature column '{temp_column}' not found in Excel file")
        
        # Create new DataFrame with required columns
        result_df = pd.DataFrame()
        
        # Convert time column to datetime
        result_df['timestamp'] = pd.to_datetime(df[time_column])
        
        # Convert temperature column to float
        result_df['temperature'] = pd.to_numeric(df[temp_column], errors='coerce')
        
        # Remove any rows with NaN values
        result_df = result_df.dropna()
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp')
        
        return result_df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process temperature data from Excel file')
parser.add_argument('excel_file', help='Path to the Excel file')
parser.add_argument('--time-column', help='Name of the column containing timestamps', required=True)
parser.add_argument('--temp-column', help='Name of the column containing temperature values', required=True)
args = parser.parse_args()

# Read data from Excel file
df = read_excel_data(args.excel_file, args.time_column, args.temp_column)

if df.empty:
    print("\nNo valid temperature readings found in Excel file")
    exit(0)

print(f"\nTotal temperature readings: {len(df)}")

# Process data and create temperature readings list
temperature_readings = []
messages_7e02 = 0
messages_7e09 = 0
other_messages = 0

for message in data:
    payload = message.get('msg', {}).get('payload', '')
    
    # Count message types
    if payload.startswith('7e02'):
        messages_7e02 += 1
        # For 7e02 messages, get the temperature from the TMP field
        temp = message.get('msg', {}).get('TMP', {}).get('measured_value')
        send_time = message.get('msg', {}).get('send_time')
        if temp is not None and send_time:
            # Convert YYYYMMDDhhmmss to datetime
            try:
                timestamp = datetime.strptime(send_time, '%Y%m%d%H%M%S')
                temperature_readings.append({
                    'timestamp': timestamp,
                    'send_time': send_time,
                    'temperature': temp
                })
            except ValueError as e:
                print(f"Error parsing send_time {send_time}: {e}")
                continue
    elif payload.startswith('7e09'):
        messages_7e09 += 1
    else:
        other_messages += 1
        print(f"Other payload prefix: {payload[:10]}...")

print(f"\nMessage type statistics:")
print(f"Messages with 7e02 prefix (Device Status): {messages_7e02}")
print(f"Messages with 7e09 prefix (BLE Data): {messages_7e09}")
print(f"Messages with other prefix: {other_messages}")
print(f"Total temperature readings collected: {len(temperature_readings)}")

# Convert to DataFrame for easier plotting
df = pd.DataFrame(temperature_readings)

if df.empty:
    print("\nNo temperature readings found from device status messages (7e02)")
    exit(0)

# Sort by timestamp
df = df.sort_values('timestamp')

def detect_steep_changes(temps, times, window=5, gradient_threshold=0.02, plateau_threshold=0.005):
    """
    Detect regions of steep temperature changes and their plateaus.
    
    Args:
        temps: Array of temperature values
        times: Array of datetime values
        window: Window size for smoothing
        gradient_threshold: Threshold for detecting steep changes
        plateau_threshold: Threshold for detecting plateaus
    
    Returns:
        List of (start_time, end_time) tuples for detected regions
    """
    # Convert times to numeric values (seconds) using numpy datetime64
    times_ns = times.astype('datetime64[ns]')
    times_numeric = (times_ns - times_ns[0]) / np.timedelta64(1, 's')
    
    # Smooth the temperature data
    if len(temps) > window:
        window = window if window % 2 == 1 else window + 1
        temps_smooth = savgol_filter(temps, window, 3)
        # Calculate gradient
        gradient = np.gradient(temps_smooth, times_numeric)
        # Smooth gradient
        gradient_smooth = savgol_filter(gradient, window, 3)
    else:
        gradient_smooth = np.gradient(temps, times_numeric)
    
    # Find regions of significant change
    regions = []
    in_region = False
    start_idx = None
    
    for i in range(len(gradient_smooth)):
        abs_gradient = abs(gradient_smooth[i])
        
        # Detect start of steep change
        if not in_region and abs_gradient > gradient_threshold:
            # Look back a few points to catch the true start
            look_back = max(0, i - window)
            start_idx = look_back + np.argmin(abs(gradient_smooth[look_back:i]))
            in_region = True
            
        # Detect end of steep change (plateau)
        elif in_region and abs_gradient < plateau_threshold:
            # Ensure we have enough points and the change is significant
            if i - start_idx >= window:
                # Look forward a few points to confirm plateau
                look_forward = min(len(gradient_smooth), i + window)
                if np.all(abs(gradient_smooth[i:look_forward]) < plateau_threshold):
                    regions.append((times[start_idx], times[i]))
            in_region = False
            start_idx = None
    
    # Handle case where region extends to the end
    if in_region and start_idx is not None:
        regions.append((times[start_idx], times[-1]))
    
    return regions

def find_temperature_regions(df):
    """
    Find regions of significant temperature changes and validate against known patterns.
    """
    # Known time ranges for validation
    manual_region1_start = datetime.strptime('20250527083000', '%Y%m%d%H%M%S')
    manual_region1_end = datetime.strptime('20250527200000', '%Y%m%d%H%M%S')
    manual_region2_start = datetime.strptime('20250528043000', '%Y%m%d%H%M%S')
    
    # Sort data by timestamp
    df_sorted = df.sort_values('timestamp')
    temps = df_sorted['temperature'].values
    times = df_sorted['timestamp'].values
    
    # Detect regions automatically
    auto_regions = detect_steep_changes(temps, times)
    
    # Filter and validate regions
    validated_regions = []
    for start_time, end_time in auto_regions:
        region_temps = df_sorted[
            (df_sorted['timestamp'] >= start_time) & 
            (df_sorted['timestamp'] <= end_time)
        ]['temperature'].values
        
        # Calculate temperature change in region
        temp_change = abs(region_temps[-1] - region_temps[0])
        
        # Only include regions with significant temperature change
        if temp_change > 1.0 and len(region_temps) > 10:
            # Convert times to numpy datetime64 for comparison
            start_time_ns = np.datetime64(start_time)
            manual_region1_start_ns = np.datetime64(manual_region1_start)
            manual_region2_start_ns = np.datetime64(manual_region2_start)
            
            # Calculate time differences in hours
            diff1 = abs(start_time_ns - manual_region1_start_ns) / np.timedelta64(1, 'h')
            diff2 = abs(start_time_ns - manual_region2_start_ns) / np.timedelta64(1, 'h')
            
            if diff1 < 2:  # Within 2 hours
                # This is our first region (descent)
                validated_regions.append((manual_region1_start, manual_region1_end))
            elif diff2 < 2:  # Within 2 hours
                # This is our second region (ascent)
                validated_regions.append((manual_region2_start, df_sorted['timestamp'].iloc[-1]))
    
    # If no regions were validated, fall back to manual regions
    if not validated_regions:
        region1_mask = (df_sorted['timestamp'] >= manual_region1_start) & (df_sorted['timestamp'] <= manual_region1_end)
        region2_mask = df_sorted['timestamp'] >= manual_region2_start
        
        if any(region1_mask):
            region1_data = df_sorted[region1_mask]
            if len(region1_data) > 10:
                validated_regions.append((region1_data.iloc[0]['timestamp'], region1_data.iloc[-1]['timestamp']))
        
        if any(region2_mask):
            region2_data = df_sorted[region2_mask]
            if len(region2_data) > 10:
                validated_regions.append((region2_data.iloc[0]['timestamp'], region2_data.iloc[-1]['timestamp']))
    
    return validated_regions

def fit_exp_decay(x_data, y_data, weights=None):
    """Fit exponential decay curve with multiple initial guesses."""
    def exp_decay(x, a, b, c):
        return a + b * np.exp(-c * x)
    
    a_guess = np.min(y_data)
    b_guess = np.max(y_data) - a_guess
    c_guess = 1.0 / (x_data[-1] - x_data[0]) * 5
    
    bounds = (
        [a_guess - 10, 0, 0],
        [a_guess + 10, (np.max(y_data) - np.min(y_data)) * 2, 0.5]
    )
    
    best_popt = None
    best_r_squared = -np.inf
    
    for c_factor in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        try:
            popt, _ = curve_fit(
                exp_decay,
                x_data - x_data[0],
                y_data,
                p0=[a_guess, b_guess, c_guess * c_factor],
                bounds=bounds,
                sigma=None if weights is None else 1/weights,
                maxfev=20000,
                ftol=1e-10,
                xtol=1e-10
            )
            
            y_fit = exp_decay(x_data - x_data[0], *popt)
            residuals = y_data - y_fit
            if weights is not None:
                residuals = residuals * weights
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_popt = popt
                
        except RuntimeError:
            continue
    
    if best_popt is None:
        raise RuntimeError("No successful fit found")
        
    return best_popt, best_r_squared, exp_decay

def fit_exp_growth(x_data, y_data, weights=None):
    """Fit exponential growth curve."""
    def exp_growth(x, a, b, c):
        return a - b * np.exp(-c * x)
    
    a_guess = np.max(y_data)
    b_guess = np.max(y_data) - np.min(y_data)
    c_guess = 1.0 / (x_data[-1] - x_data[0]) * 5
    
    bounds = (
        [a_guess - 10, 0, 0],
        [a_guess + 10, (np.max(y_data) - np.min(y_data)) * 2, 0.5]
    )
    
    best_popt = None
    best_r_squared = -np.inf
    
    for c_factor in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        try:
            popt, _ = curve_fit(
                exp_growth,
                x_data - x_data[0],
                y_data,
                p0=[a_guess, b_guess, c_guess * c_factor],
                bounds=bounds,
                sigma=None if weights is None else 1/weights,
                maxfev=20000,
                ftol=1e-10,
                xtol=1e-10
            )
            
            y_fit = exp_growth(x_data - x_data[0], *popt)
            residuals = y_data - y_fit
            if weights is not None:
                residuals = residuals * weights
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_popt = popt
                
        except RuntimeError:
            continue
    
    if best_popt is None:
        raise RuntimeError("No successful fit found")
        
    return best_popt, best_r_squared, exp_growth

# Find temperature regions
regions = find_temperature_regions(df)

print(f"\nFound {len(regions)} temperature regions")

# Create the plot
plt.figure(figsize=(15, 8))

# Plot all data points
plt.scatter(df['timestamp'], df['temperature'], label='All Data', alpha=0.5, color='blue', s=20)

# Colors for different regions
colors = ['red', 'green']

# Fit and plot curves for each region
for i, (start_time, end_time) in enumerate(regions):
    color = colors[i % len(colors)]
    
    # Filter data for this region
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    region_data = df[mask].copy()
    
    if len(region_data) >= 10:  # Ensure enough points for fitting
        # Convert timestamps to seconds for fitting
        times_ns = region_data['timestamp'].values.astype('datetime64[ns]')
        region_data['seconds'] = (times_ns - times_ns[0]) / np.timedelta64(1, 's')
        
        x_data = region_data['seconds'].values
        y_data = region_data['temperature'].values
        
        try:
            # Calculate weights based on gradient
            gradient = np.gradient(y_data)
            gradient_smooth = savgol_filter(gradient, min(15, len(y_data)-1), 3)
            weights = np.exp(-(gradient_smooth**2) / (2 * np.std(gradient_smooth)**2))
            weights = weights / np.max(weights) + 0.2
            
            # Fit curve (decay for first region, growth for second)
            if i == 0:  # First region (descent)
                popt, r_squared, curve_func = fit_exp_decay(x_data, y_data, weights)
                curve_type = "decay"
            else:  # Second region (ascent)
                popt, r_squared, curve_func = fit_exp_growth(x_data, y_data, weights)
                curve_type = "growth"
            
            # Generate points for the fitted curve
            x_fit = np.linspace(x_data[0], x_data[-1], 1000)
            y_fit = curve_func(x_fit - x_data[0], *popt)
            
            # Plot fitted region with size proportional to weights
            sizes = 100 * weights
            plt.scatter(region_data['timestamp'], region_data['temperature'],
                       s=sizes, alpha=0.7, color=color, label=f'Region {i+1} Data')
            
            # Plot fitted curve
            timestamps_fit = pd.date_range(
                start=region_data['timestamp'].iloc[0],
                periods=len(x_fit),
                freq=pd.Timedelta(seconds=(x_fit[-1] - x_fit[0]) / (len(x_fit) - 1))
            )
            
            plt.plot(timestamps_fit, y_fit, '--',
                    color=color,
                    label=f'Region {i+1} Fit ({curve_type}): {popt[0]:.2f} + {popt[1]:.2f}e^(-{popt[2]:.4f}t)\nR² = {r_squared:.4f}',
                    linewidth=2)
            
            # Add vertical lines for region boundaries
            plt.axvline(x=start_time, color=color, linestyle=':', alpha=0.5)
            plt.axvline(x=end_time, color=color, linestyle=':', alpha=0.5)
            
            print(f"\nRegion {i+1} fitting parameters ({curve_type}):")
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Equation: T = {popt[0]:.2f} + {popt[1]:.2f}e^(-{popt[2]:.4f}t)")
            print(f"R² = {r_squared:.4f}")
            
        except Exception as e:
            print(f"Error fitting curve for region {i+1}: {e}")

plt.title('Device Temperature Over Time with Decay and Growth Curves')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('temperature_plot.png', bbox_inches='tight')
print("\nPlot has been saved as 'temperature_plot.png'")

# Print sample of the data in tabular format
print("\nSample of Temperature Readings (first 10 entries):")
table_data = [{
    'send_time': reading['send_time'],
    'temperature': reading['temperature']
} for reading in temperature_readings]
print(tabulate(table_data[:10], headers="keys", tablefmt="grid"))
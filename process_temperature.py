import pandas as pd
import numpy as np
from datetime import datetime
from scipy import interpolate

def process_excel_data():
    # Read the Excel file
    print("Reading Excel file...")
    df = pd.read_excel('BSF4905127.xlsx')
    
    # The columns are known from the file
    temp_col = 'Temperature (°C)'
    time_col = 'Time'
    
    # Clean and prepare the data
    print("Processing data...")
    
    # Convert temperature to numeric, removing any non-numeric values
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    
    # Convert timestamp to datetime, first removing the timezone info
    df[time_col] = df[time_col].str.replace(' MST', '')  # Remove MST timezone
    df[time_col] = pd.to_datetime(df[time_col], format='mixed', errors='coerce')
    
    # Remove rows with invalid data
    df = df.dropna(subset=[time_col, temp_col])
    
    # Sort by timestamp
    df = df.sort_values(time_col)
    
    # Convert timestamp to minutes from start
    start_time = df[time_col].min()
    df['normalized_time'] = (df[time_col] - start_time).dt.total_seconds() / 60
    
    # Create base dataframe with just time and temperature
    base_df = df[['normalized_time', temp_col]].copy()
    base_df.columns = ['normalized_time', 'temperature']
    
    # Round time to nearest second to avoid floating point issues
    base_df['normalized_time'] = base_df['normalized_time'].round(6)
    
    # Create interpolation function
    print("Interpolating missing minutes...")
    if len(base_df) < 2:
        print("Error: Not enough valid data points for interpolation")
        return
        
    f = interpolate.interp1d(base_df['normalized_time'], base_df['temperature'], kind='linear')
    
    # Create evenly spaced time points for every minute
    max_minutes = int(np.ceil(base_df['normalized_time'].max()))
    even_minutes = np.arange(0, max_minutes + 1)
    
    # Interpolate temperatures at even minute intervals
    interpolated_temps = f(even_minutes)
    
    # Create DataFrame with original data
    original_df = pd.DataFrame({
        'normalized_time': base_df['normalized_time'],
        'temperature': base_df['temperature'],
        'is_original': True
    })
    
    # Create DataFrame with interpolated data
    interpolated_df = pd.DataFrame({
        'normalized_time': even_minutes,
        'temperature': interpolated_temps,
        'is_original': False
    })
    
    # Combine original and interpolated data
    combined_df = pd.concat([original_df, interpolated_df])
    combined_df = combined_df.sort_values('normalized_time').reset_index(drop=True)
    
    # Remove duplicates but keep the original data points
    combined_df = combined_df.drop_duplicates(subset='normalized_time', keep='first')
    
    # Round temperatures to 1 decimal place
    combined_df['temperature'] = combined_df['temperature'].round(1)
    
    print("\nSaving files...")
    # Save to CSV files
    # Save only time and temperature
    result_df = combined_df[['normalized_time', 'temperature']]
    result_df.to_csv('temperature_vs_time.csv', index=False)
    
    # Save with additional column showing which points are original
    combined_df.to_csv('temperature_vs_time_with_source.csv', index=False)
    
    print("\nData summary:")
    print(f"Number of original data points: {len(original_df)}")
    print(f"Number of interpolated points: {len(interpolated_df)}")
    print(f"Time range: 0 to {max_minutes} minutes")
    print(f"Temperature range: {combined_df['temperature'].min():.1f}°C to {combined_df['temperature'].max():.1f}°C")
    
    # Print some sample data
    print("\nFirst few rows of processed data:")
    print(result_df.head())
    
    print("\nFiles created:")
    print("1. temperature_vs_time.csv - Contains normalized time and temperature")
    print("2. temperature_vs_time_with_source.csv - Also shows which points are original vs interpolated")

if __name__ == "__main__":
    try:
        process_excel_data()
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("\nPlease ensure:")
        print("1. The Excel file 'BSF4905127.xlsx' is in the current directory")
        print("2. The file contains the required columns:")
        print("   - 'Temperature (°C)'")
        print("   - 'Time'") 
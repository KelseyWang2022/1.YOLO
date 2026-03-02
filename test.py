import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Load the data
data_path = "traffic_predictions_models_corrected.csv"  # Update this path as needed
data = pd.read_csv(data_path)

# Generate time series from 2023-11-13 07:21:57 to 12:56:57 (each minute)
start_time = datetime.strptime("2023-11-13 07:21:57", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2023-11-13 12:56:57", "%Y-%m-%d %H:%M:%S")
time_series = pd.date_range(start=start_time, end=end_time, freq='T')

# Ensure the data length matches the time series length
if len(data) > len(time_series):
    data = data.head(len(time_series))
elif len(data) < len(time_series):
    time_series = time_series[:len(data)]

# Plot the data with more distinct colors for actual value and ST-Transformer
plt.figure(figsize=(15, 8))

# Plot actual value and ST-Transformer with solid lines and distinct colors
plt.plot(time_series, data['Actual value'], label='Actual Value', linewidth=0.5, linestyle='-', color='black')
plt.plot(time_series, data['ST-Transformer'], label='ST-Transformer', linewidth=0.5, linestyle='-', color='orange')

# Plot other models with dashed lines
plt.plot(time_series, data['LSTM'], label='LSTM', linewidth=0.5, linestyle='--')
plt.plot(time_series, data['Transformer'], label='Transformer', linewidth=0.5, linestyle='--')
plt.plot(time_series, data['PDformer'], label='PDformer', linewidth=0.5, linestyle='--')
plt.plot(time_series, data['Steformer'], label='Steformer', linewidth=0.5, linestyle='--')

# Define the major ticks for x-axis at specific times
major_ticks = [
    "2023-11-13 07:30:00", "2023-11-13 08:00:00", "2023-11-13 08:30:00", "2023-11-13 09:00:00",
    "2023-11-13 09:30:00", "2023-11-13 10:00:00", "2023-11-13 10:30:00", "2023-11-13 11:00:00",
    "2023-11-13 11:30:00", "2023-11-13 12:00:00", "2023-11-13 12:30:00"
]
major_ticks = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in major_ticks]

# Set x-axis major ticks
plt.gca().set_xticks(major_ticks)
plt.gca().set_xticklabels([tick.strftime('%H:%M') for tick in major_ticks])

# Rotate x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Add title and labels
plt.title('Comparision of the actual and predicted value in FI-LI-PI(CCTV point)')
plt.xlabel('Time')
plt.ylabel('Speed (km/h)')

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(True)

# Save the plot
visualization_output_path = "traffic_predictions_visualization_hour_minute.png"
plt.savefig(visualization_output_path)

# Show the plot
plt.show()

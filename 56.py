import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
data_path = 'traffic_predictions_models_corrected.csv'  # Update this with the path to your data
data = pd.read_csv(data_path)

# Assuming the first 50 rows correspond to the first 50 time periods
time_series = pd.date_range(start="2023-11-13 08:00:00", periods=50, freq='T')  # Generate the corresponding time series

# Create the plot
plt.figure(figsize=(15, 8))

# Plot actual value and ST-Transformer with solid lines
plt.plot(time_series, data['Actual value'][:50], label='Actual Value', color='black', linestyle='-', linewidth=1.5)
plt.plot(time_series, data['ST-Transformer'][:50], label='ST-Transformer', color='orange', linestyle='-', linewidth=1.5)

# Plot other models with dashed lines
plt.plot(time_series, data['LSTM'][:50], label='LSTM', linestyle='--', color='blue', linewidth=1)
plt.plot(time_series, data['Transformer'][:50], label='Transformer', linestyle='--', color='green', linewidth=1)
plt.plot(time_series, data['PDformer'][:50], label='PDformer', linestyle='--', color='red', linewidth=1)
plt.plot(time_series, data['Steformer'][:50], label='Steformer', linestyle='--', color='purple', linewidth=1)

# Customize the x-axis to show labels more clearly
plt.xticks(time_series[::10], [t.strftime('%H:%M') for t in time_series[::10]], rotation=45)

# Add grid, legend, title, and labels
plt.grid(True)
plt.legend()
plt.title('Traffic Speed Predictions for the First 50 Time Periods')
plt.xlabel('Time (HH:MM)')
plt.ylabel('Speed (km/h)')

# Display the plot
plt.show()

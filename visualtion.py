import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the adjusted predictions data
adjusted_data_path = r"C:\Users\Kelsey\PycharmProjects\computer_vision\final_traffic_predictions_adjusted_all (1).csv"
adjusted_data = pd.read_csv(adjusted_data_path)

# Plot the data for only Density and ST-Transformer
plt.figure(figsize=(12, 6))
plt.plot(adjusted_data['Speed'], label='Actual value', linewidth=0.5)
plt.plot(adjusted_data['ST-Transformer'], label='ST-Transformer', linewidth=0.5)

# Set the y-axis limits
plt.ylim(50, 80)

# Add title and labels
plt.title('Comparsion of the actual and predicted value in PEMS08')
plt.xlabel('Time_Step')
plt.ylabel('Speed')

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(True)

# Save the plot
st_transformer_visualization_path = "traffic_predictions_st_transformer_visualization.png"
plt.savefig(st_transformer_visualization_path)

# Show the plot
plt.show()

# Provide the path to the saved visualization
print(f"Visualization saved at: {st_transformer_visualization_path}")

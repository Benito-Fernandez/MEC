import pandas as pd

# Sample data dictionary
data = {
    "timestamp": pd.date_range(start="2025-05-23 00:00:00", periods=10, freq="1S"),
    "raw_signal": [100, 102, 101, None, 105, 500, 103, "ERROR", 104, 106]  # Includes missing, outlier, sensor error
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/industrial_sensor_data.csv", index=False)

print("Sample CSV file 'data/industrial_sensor_data.csv' has been created.")

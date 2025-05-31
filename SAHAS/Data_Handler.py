import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d

def process_data(file_path, sampling_rate="1S"):
    """
    Processes industrial sensor data with noise reduction, interpolation, detrending,
    outlier removal, and resampling.

    :param file_path: Path to CSV file.
    :param sampling_rate: Desired resampling frequency (e.g., "1S" for 1-second intervals).
    :return: Cleaned and resampled DataFrame.
    """
    # Load CSV file
    df = pd.read_csv(file_path)

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.set_index("timestamp", inplace=True)

    # Convert signal column to numeric, handling errors (for sensor failure strings)
    df["raw_signal"] = pd.to_numeric(df["raw_signal"], errors="coerce")

    # 1. Noise Reduction - Moving average filter
    window_size = 5
    df["filtered_signal"] = df["raw_signal"].rolling(window=window_size, center=True).mean()

    # 2. Interpolation for missing values
    df["interpolated_signal"] = df["filtered_signal"].interpolate(method="linear")

    # 3. Drift & Bias Correction - Detrending
    df["detrended_signal"] = signal.detrend(df["interpolated_signal"])

    # 4. Outlier Removal - Using Interquartile Range (IQR)
    Q1 = df["detrended_signal"].quantile(0.25)
    Q3 = df["detrended_signal"].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df["detrended_signal"] >= Q1 - 1.5 * IQR) & (df["detrended_signal"] <= Q3 + 1.5 * IQR)]

    # 5. Resampling to Known Sampling Frequency
    df_resampled = df_clean.resample(sampling_rate).mean()

    return df_resampled

# Example usage:
if __name__ == "__main__":
    processed_data = process_data("data/industrial_sensor_data.csv")
    print(processed_data.head())

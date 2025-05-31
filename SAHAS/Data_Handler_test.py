import unittest
import pandas as pd
import numpy as np
from Data_Handler import process_data  # Assuming we encapsulated processing in a function `process_data`

class TestDataHandler(unittest.TestCase):

    def setUp(self):
        """Setup synthetic test data."""
        data = {
            "timestamp": pd.date_range(start="2025-05-23 00:00:00", periods=10, freq="1S"),
            "raw_signal": [100, 102, 101, np.nan, 105, 500, 103, "ERROR", 104, 106]  # Includes noise, missing data, outlier, and error string
        }
        self.df = pd.DataFrame(data)
        self.df.to_csv("test_data.csv", index=False)

    def test_data_loading(self):
        """Test loading of CSV files."""
        df_processed = process_data("test_data.csv")
        self.assertIsInstance(df_processed, pd.DataFrame)

    def test_missing_data_interpolation(self):
        """Ensure missing values are interpolated correctly."""
        df_processed = process_data("test_data.csv")
        self.assertFalse(df_processed["raw_signal"].isnull().any(), "Missing values should be interpolated.")

    def test_outlier_removal(self):
        """Verify outliers are removed."""
        df_processed = process_data("test_data.csv")
        self.assertLess(df_processed["raw_signal"].max(), 200, "Extreme outlier should be removed.")

    def test_sensor_failure_handling(self):
        """Ensure sensor error strings are handled properly."""
        df_processed = process_data("test_data.csv")
        self.assertNotIn("ERROR", df_processed["raw_signal"].values, "Sensor failure strings should be removed.")

    def test_resampling(self):
        """Validate resampling to consistent frequency."""
        df_processed = process_data("test_data.csv")
        expected_freq = "1S"  # Expected sampling frequency
        self.assertEqual(df_processed.index.freqstr, expected_freq, "Data should be resampled to expected frequency.")

if __name__ == "__main__":
    unittest.main()

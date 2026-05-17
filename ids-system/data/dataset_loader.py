import os
import pandas as pd
import numpy as np

class FlowDatasetLoader:
    """
    Handles loading of raw tabular flow network datasets like CICIDS2017, UNSW-NB15, etc.
    """
    def __init__(self, data_path: str, chunk_size: int = 100000):
        self.data_path = data_path
        self.chunk_size = chunk_size

    def load_csv(self, filepath: str, label_col: str = None) -> pd.DataFrame:
        """
        Loads a single CSV file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        if label_col is not None and label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset. Columns available: {df.columns.tolist()}")
            
        return df
        
    def create_synthetic_data(self, num_samples: int = 10000, num_features: int = 78) -> pd.DataFrame:
        """
        Generates synthetic flow data for testing the pipeline when raw datasets aren't available.
        """
        print(f"Generating synthetic dataset with {num_samples} samples and {num_features} features.")
        np.random.seed(42)
        
        # Synthetic features
        features = np.random.randn(num_samples, num_features).astype(np.float32)
        
        # Synthetic labels (0 = Benign, 1 = DoS, 2 = PortScan, 3 = Exfiltration)
        labels = np.random.choice(["Benign", "DoS", "PortScan", "Exfiltration"], size=num_samples, p=[0.7, 0.15, 0.1, 0.05])
        
        df = pd.DataFrame(features, columns=[f"Feature_{i}" for i in range(num_features)])
        df["Label"] = labels
        
        return df

if __name__ == "__main__":
    # Simple test for dataset loader
    loader = FlowDatasetLoader(data_path=".")
    df = loader.create_synthetic_data(num_samples=100)
    print(f"Dataset Shape: {df.shape}")
    print(f"Label Distribution:\n{df['Label'].value_counts()}")

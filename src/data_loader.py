"""
data_loader.py - Loads the public transport delays dataset from Kaggle or a local CSV.
"""
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

KAGGLE_DATASET = "khushikyad001/public-transport-delays-with-weather-and-events"

def load_dataset(local_csv_path=None):
    """Returns the dataset as a pandas DataFrame."""
    if local_csv_path is not None:
        df = pd.read_csv(local_csv_path)
        print(f"Loaded local CSV: {local_csv_path}")
        return df
    
    print("Downloading dataset from Kaggle...")
    ds_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = sorted([p for p in ds_path.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {ds_path}")
    
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, KAGGLE_DATASET, csv_files[0].name)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df
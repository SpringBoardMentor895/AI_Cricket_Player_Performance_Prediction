import pandas as pd

def safe_read_csv(path):
    """
    Reads a CSV safely and prints useful debug info.
    """
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print("Shape:", df.shape)
    return df

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path} | Shape: {df.shape}")

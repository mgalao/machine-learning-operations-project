from pathlib import Path
import pandas as pd

# Get project root (assuming this script is under src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def load_dataset(data_subfolder: str, filename: str, **kwargs) -> pd.DataFrame:
    """
    Load a dataset from a specified subfolder in the `data` directory.

    Args:
        data_subfolder (str): Subfolder name under the data directory (e.g., "01_raw").
        filename (str): Name of the file to load (e.g., "data.csv").
        **kwargs: Additional arguments passed to `pandas.read_csv`.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = DATA_DIR / data_subfolder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)

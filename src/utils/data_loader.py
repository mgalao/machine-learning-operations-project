from pathlib import Path
import pandas as pd

# Correctly set project root two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def load_dataset(data_subfolder: str, filename: str, **kwargs) -> pd.DataFrame:
    file_path = DATA_DIR / data_subfolder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, **kwargs)

"""Configuration for latest_trans_concept module."""

from pathlib import Path

# Get the root directory of the project (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "datasets" / "train.parquet"


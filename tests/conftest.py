"""Shared fixtures and paths for ARGUS tests."""
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "synthetic"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "argus_features.csv"
MODEL_FILE = BASE_DIR / "models" / "argus_xgboost.json"
RESULTS_DIR = BASE_DIR / "results"


@pytest.fixture
def base_dir():
    return BASE_DIR

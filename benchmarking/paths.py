from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_DERIVED = DATA / "derived"
RESULTS = ROOT / "results"


def ensure_data_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)

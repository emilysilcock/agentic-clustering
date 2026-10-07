from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_DERIVED = DATA / "derived"
RESULTS = ROOT / "results"
TABLES = ROOT / "paper"


def ensure_data_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)


def table_out_dir(description: str | None = None) -> Path:
    """Parse ``--out-dir`` for the paper-table builders and create the directory."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=TABLES,
        help="Where to write the .tex table (default: paper/).",
    )
    out_dir = p.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

from pathlib import Path

from analysis.model_analysis import run_scope_analyses

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "lags.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "lags"


if __name__ == "__main__":
    run_scope_analyses(DATA_PATH, OUTPUT_DIR)

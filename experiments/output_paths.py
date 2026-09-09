"""Shared location for generated figures."""
import os
from pathlib import Path


def output_path(name):
    directory = Path(os.environ.get("JMLR_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name

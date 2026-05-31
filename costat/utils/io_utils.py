"""Small filesystem helpers shared across the pipeline."""

import csv
import os
from typing import Any, Dict, List


def ensure_dir(directory_path: str) -> str:
    """Create a directory (and parents) if missing and return its path."""
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write a list of uniform dictionaries to a CSV file.

    The column order follows the keys of the first row, which keeps the report
    readable and stable across runs.
    """
    if not rows:
        return
    ensure_dir(os.path.dirname(output_path))
    field_names: List[str] = list(rows[0].keys())
    with open(output_path, "w", encoding="ascii", newline="") as handle:
        writer: csv.DictWriter = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

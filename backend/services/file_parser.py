"""
backend/services/file_parser.py

Reads an uploaded file into a DataFrame. Just bytes-on-disk -> table -
column meaning and cleaning are the Ingestion/Validation Agents' job,
not this module's.
"""

from pathlib import Path
import pandas as pd


class FileParseError(Exception):
    pass


def parse_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise FileParseError(f"Unsupported file type: {suffix!r}. Expected .csv, .xlsx, or .xls.")
    except FileParseError:
        raise
    except Exception as e:
        raise FileParseError(f"Could not read {path.name}: {e}") from e

    if df.empty:
        raise FileParseError(f"{path.name} contains no data rows.")

    return df
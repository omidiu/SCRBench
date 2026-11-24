import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class InputDatasetSpec:
    folder_path: str
    filename: str
    required_columns: Sequence[str]



def validate_columns(df: pd.DataFrame, required_columns: Sequence[str], *, context: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}. Observed columns: {list(df.columns)}")


def iter_csv_in_chunks(
    path: str,
    *,
    chunksize: int,
    max_rows: Optional[int] = None,
) -> Iterable[pd.DataFrame]:
    """
    Stream a potentially huge CSV using pandas chunking.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")

    
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if max_rows is None:
            yield chunk
            continue

        remaining = max_rows - seen
        if remaining <= 0:
            return

        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        yield chunk
        seen += len(chunk)


def count_csv_rows(path: str) -> int:
    """
    Count data rows (excluding header) in a CSV file.

    This enables "how many left" progress reporting even when streaming in chunks.
    """
    # NOTE:
    # Many of our CSVs store code in string columns, which can contain embedded newlines.
    # Counting raw '\n' bytes will overcount rows in that case. So we count rows by
    # actually parsing the CSV (streaming in chunks to keep memory bounded).
    #
    # This is slower than newline-counting, but it is correct.
    n = 0
    for chunk in pd.read_csv(path, chunksize=200_000):
        n += len(chunk)
    return int(n)


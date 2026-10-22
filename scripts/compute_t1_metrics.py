"""Compute T1 accuracy metrics from a reference_llm_judge JSON file.

The input is a list of records with ``real_class``, ``predicted_class``, and
``paths``. Unparseable predictions (missing / non-boolean ``predicted_class``)
are counted as incorrect.

Usage:
  python scripts/compute_t1_metrics.py INPUT.json OUTPUT.json
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "yes", "1"):
            return True
        if token in ("false", "no", "0"):
            return False
        if token in ("", "nan", "none", "null", "error"):
            return None
    return None


def _path_length(paths: Any) -> int:
    if isinstance(paths, str):
        paths = ast.literal_eval(paths)
    if not paths:
        raise ValueError("empty paths")
    lengths = {len(path) for path in paths}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent path lengths: {sorted(lengths)}")
    return lengths.pop()


def _pct(correct: int, n: int, *, decimals: int) -> float:
    if n == 0:
        raise ValueError("cannot compute percentage with n=0")
    return round(100.0 * correct / n, decimals)


def _rate_block(correct: int, n: int, *, decimals: int, key: str) -> dict[str, int | float]:
    return {
        "correct": correct,
        "n": n,
        key: _pct(correct, n, decimals=decimals),
    }


def compute_t1_metrics(records: list[dict[str, Any]], *, decimals: int = 2) -> dict[str, Any]:
    by_k: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_y: dict[bool, dict[str, int]] = {
        True: {"n": 0, "correct": 0},
        False: {"n": 0, "correct": 0},
    }
    overall_correct = 0
    unparseable = 0

    for index, row in enumerate(records):
        real = _to_bool(row.get("real_class"))
        if real is None:
            raise ValueError(f"row {index}: missing or invalid real_class")
        pred = _to_bool(row.get("predicted_class"))
        k = _path_length(row.get("paths"))
        correct = pred is not None and pred == real
        if pred is None:
            unparseable += 1
        overall_correct += int(correct)
        by_k[k]["n"] += 1
        by_k[k]["correct"] += int(correct)
        by_y[real]["n"] += 1
        by_y[real]["correct"] += int(correct)

    n = len(records)
    return {
        "n": n,
        "unparseable": unparseable,
        "overall": _rate_block(overall_correct, n, decimals=decimals, key="accuracy_pct"),
        "by_k": [
            {"k": k, **_rate_block(stats["correct"], stats["n"], decimals=decimals, key="accuracy_pct")}
            for k, stats in sorted(by_k.items())
        ],
        "recall": {
            "equivalent": _rate_block(
                by_y[True]["correct"], by_y[True]["n"], decimals=decimals, key="recall_pct"
            ),
            "non_equivalent": _rate_block(
                by_y[False]["correct"], by_y[False]["n"], decimals=decimals, key="recall_pct"
            ),
        },
    }


def write_t1_metrics(input_path: Path, output_path: Path, *, decimals: int = 2) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a JSON list of records")

    metrics = compute_t1_metrics(records, decimals=decimals)
    metrics["input_path"] = str(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute T1 accuracy metrics from a judge JSON file.")
    parser.add_argument("json_path", type=Path, help="Path to reference_llm_judge JSON")
    parser.add_argument("output_path", type=Path, help="Path to write metrics JSON")
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for reported percentages (default: 2)",
    )
    args = parser.parse_args()
    write_t1_metrics(args.json_path, args.output_path, decimals=args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

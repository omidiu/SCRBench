"""Compute T2 accuracy metrics from a reference_llm_judge JSON file.

The input is an object with ``metadata`` and ``rows``. Each row has
``ground_truth_altering_id``, ``predicted_id``, and ``k``. Unparseable
predictions (missing / non-integer ``predicted_id``) are counted as incorrect.

Usage:
  python scripts/compute_t2_metrics.py INPUT.json OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_int(value: Any) -> int | None:
    if value is True or value is False or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        if value == int(value):
            return int(value)
        return None
    if isinstance(value, str):
        token = value.strip()
        if token == "" or token.lower() in ("nan", "none", "null", "error"):
            return None
        try:
            parsed = float(token)
        except ValueError:
            return None
        if parsed == int(parsed):
            return int(parsed)
        return None
    return None


def _path_length(paths: Any) -> int:
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


def compute_t2_metrics(
    payload: dict[str, Any],
    *,
    decimals: int = 2,
) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("expected an object with a JSON list field 'rows'")

    transformations = payload.get("metadata", {}).get("transformations", {})
    by_k: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_gold: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    overall_correct = 0
    unparseable = 0

    for index, row in enumerate(rows):
        gold = _to_int(row.get("ground_truth_altering_id"))
        if gold is None:
            raise ValueError(f"row {index}: missing or invalid ground_truth_altering_id")
        pred = _to_int(row.get("predicted_id"))
        k = row.get("k")
        k = int(k) if k is not None else _path_length(row.get("paths"))
        if pred is None:
            unparseable += 1
            correct = False
        else:
            correct = pred == gold
        overall_correct += int(correct)
        by_k[k]["n"] += 1
        by_k[k]["correct"] += int(correct)
        by_gold[gold]["n"] += 1
        by_gold[gold]["correct"] += int(correct)

    n = len(rows)
    return {
        "n": n,
        "unparseable": unparseable,
        "overall": _rate_block(overall_correct, n, decimals=decimals, key="accuracy_pct"),
        "by_k": [
            {"k": k, **_rate_block(stats["correct"], stats["n"], decimals=decimals, key="accuracy_pct")}
            for k, stats in sorted(by_k.items())
        ],
        "recall_by_altering_transformation": [
            {
                "id": gold_id,
                "name": transformations.get(str(gold_id), {}).get("name"),
                **_rate_block(stats["correct"], stats["n"], decimals=decimals, key="recall_pct"),
            }
            for gold_id, stats in sorted(by_gold.items())
        ],
    }


def write_t2_metrics(input_path: Path, output_path: Path, *, decimals: int = 2) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{input_path}: expected a JSON object with 'rows'")

    metrics = compute_t2_metrics(payload, decimals=decimals)
    metrics["input_path"] = str(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute T2 accuracy metrics from a judge JSON file.")
    parser.add_argument("json_path", type=Path, help="Path to reference_llm_judge JSON")
    parser.add_argument("output_path", type=Path, help="Path to write metrics JSON")
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for reported percentages (default: 2)",
    )
    args = parser.parse_args()
    write_t2_metrics(args.json_path, args.output_path, decimals=args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

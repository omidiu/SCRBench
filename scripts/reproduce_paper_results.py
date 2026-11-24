"""Recompute the paper's tables and figure from saved model responses."""

from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal, ROUND_HALF_UP
from html import escape
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from core.experiments import TaskType
from core.experiments.paper_responses import PAPER_RESPONSES_REPO, restore_responses, sha256
from core.experiments.published_data import PUBLISHED_REFERENCE_PATHS, resolve_reference_file
from core.experiments.tasks import get_prompt_spec

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "Base": "base",
    "Base+CoT": "base_cot",
    "Base+one-shot": "base_one_shot",
    "Base+CoT+one-shot": "base_cot_one_shot",
}
MODELS = [
    "Qwen3-235B", "GPT-OSS-120B", "Mistral-Small-24B",
    "Llama-3.1-8B", "GPT-4.1-nano",
]
PREDICTION_COLUMNS = {
    "T1": "semantic_is_equal",
    "T2": "altering_transformation_id",
    "T3": "predicted_path",
}


def parse_paper_response(task: str, raw: str) -> bool | int | list[int] | None:
    """Implement the parsing rules published in Appendix B of arXiv v1."""
    if task == "T1":
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        final = lines[-1].upper() if lines else ""
        return {"YES": True, "NO": False}.get(final)

    if task == "T2":
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        match = re.search(r"-?\d+", lines[-1]) if lines else None
        if match is None:
            match = re.search(r"-?\d+", raw)
        value = int(match.group()) if match else None
        return value if value in range(1, 10) else None

    if task != "T3":
        raise ValueError(f"Unknown task: {task}")
    lines = raw.splitlines()
    if not lines:
        return None
    sequence = []
    for field in lines[-1].split(","):
        match = re.search(r"-?\d+", field)
        if match:
            value = int(match.group())
            if value in range(1, 10):
                sequence.append(value)
    return sequence or None


def accuracy(correct: int, total: int) -> str:
    if total <= 0:
        raise ValueError("Accuracy requires a nonempty group.")
    return str((Decimal(correct) * 100 / Decimal(total)).quantize(
        Decimal(".01"), rounding=ROUND_HALF_UP
    ))


def is_correct(task: str, prediction: Any, reference: dict, failed: bool = False) -> bool:
    if failed or prediction is None:
        return False
    if task == "T1":
        return type(prediction) is bool and prediction == reference["label"]
    if task == "T2":
        return type(prediction) is int and prediction == reference["altering_id"]
    if task == "T3":
        return prediction in reference["sequence_set"]
    raise ValueError(f"Unknown task: {task}")


def read_reference(path: Path) -> dict[int, dict]:
    references = {}
    frame = pd.read_parquet(path)
    for row in frame.itertuples(index=False):
        pair_id = int(row.id)
        if pair_id in references:
            raise ValueError(f"{path}: duplicate reference ID {pair_id}")
        raw_paths = row.paths.tolist() if hasattr(row.paths, "tolist") else row.paths
        sequences = [[int(item) for item in sequence] for sequence in raw_paths]
        k = int(row.k)
        label = bool(row.is_semantically_equivalent)
        if type(label) is not bool or not sequences:
            raise ValueError(f"{path}: invalid ground truth for {pair_id}")
        altering_ids = []
        for sequence in sequences:
            if (len(sequence) != k or len(set(sequence)) != k
                    or any(type(t) is not int or t not in range(1, 10) for t in sequence)):
                raise ValueError(f"{path}: invalid sequence for {pair_id}")
            altering_ids.append({t for t in sequence if t <= 4})
        if (any(ids != altering_ids[0] for ids in altering_ids)
                or len(altering_ids[0]) != (0 if label else 1)):
            raise ValueError(f"{path}: inconsistent altering IDs for {pair_id}")
        references[pair_id] = {
            "k": k, "label": label, "sequence_set": sequences,
            "altering_id": next(iter(altering_ids[0]), None),
        }
    return references


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_run(spec: dict, path: Path, references: dict[int, dict]) -> tuple[dict, list[dict], list[dict]]:
    """Use raw text as the prediction source; saved predictions are only compared."""
    task = spec["task"]
    current_parser = get_prompt_spec(TaskType[task], PROMPTS[spec["prompt"]]).parse_response
    groups: dict[str, list[bool]] = {}
    seen = set()
    rows = []
    differences = []
    parser_disagreements = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "raw_response", "err", PREDICTION_COLUMNS[task]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
        for row in reader:
            pair_id = int(row["pair_id"])
            if pair_id in seen or pair_id not in references:
                raise ValueError(f"{path}: duplicate or unknown pair ID {pair_id}")
            seen.add(pair_id)
            reference = references[pair_id]
            raw = row["raw_response"]
            prediction = parse_paper_response(task, raw)
            current, unknown = current_parser(raw)
            parser_disagreements += current != prediction or unknown != (prediction is None)
            failed = bool(row["err"].strip())
            correct = is_correct(task, prediction, reference, failed)
            try:
                saved = ast.literal_eval(row[PREDICTION_COLUMNS[task]])
            except (SyntaxError, ValueError):
                saved = None
            if saved != prediction:
                differences.append({
                    "task": task, "model": spec["model"], "prompt": spec["prompt"],
                    "id": pair_id, "stored_prediction": json.dumps(saved),
                    "reparsed_prediction": json.dumps(prediction),
                    "old_correct": is_correct(task, saved, reference, failed),
                    "new_correct": correct, "original_error": row["err"],
                    "final_line": raw.splitlines()[-1] if raw.splitlines() else "",
                })
            group_names = [
                "overall", f"k{reference['k']}",
                "equivalent" if reference["label"] else "non_equivalent",
            ]
            if reference["altering_id"] is not None:
                group_names.append(f"altering_{reference['altering_id']}")
            for group in group_names:
                groups.setdefault(group, []).append(correct)
            rows.append({
                "id": pair_id, "k": reference["k"],
                "prediction": json.dumps(prediction), "correct": correct,
                "parse_unknown": prediction is None, "original_error": row["err"],
            })
    if seen != references.keys():
        raise ValueError(f"{path}: missing {len(references.keys() - seen)} reference IDs")
    metrics = {
        group: {"correct": sum(values), "total": len(values),
                "accuracy": accuracy(sum(values), len(values))}
        for group, values in groups.items()
    }
    return {
        "task": task, "model": spec["model"], "prompt": spec["prompt"],
        "run_id": spec["run_id"], "metrics": metrics,
        "parse_unknown": sum(row["parse_unknown"] for row in rows),
        "original_error_rows": sum(bool(row["original_error"]) for row in rows),
        "current_parser_disagreements": parser_disagreements,
    }, rows, differences


def build_tables(runs: list[dict], output: Path) -> dict[str, list[dict]]:
    tables = {}
    for task in ("T1", "T2", "T3"):
        tables[task] = [
            {"model": run["model"], "prompt": run["prompt"],
             **{key: run["metrics"][key]["accuracy"]
                for key in ("overall", "k1", "k2", "k3", "k4", "k5")}}
            for run in runs if run["task"] == task
        ]
    tables["T2_by_transformation"] = [
        {"model": run["model"],
         **{name: run["metrics"][f"altering_{i}"]["accuracy"]
            for i, name in enumerate(("AOM", "BLM", "LOM", "COM"), 1)}}
        for run in runs if run["task"] == "T2" and run["prompt"] == "Base+CoT+one-shot"
    ]
    tables["T1_by_class"] = [
        {"model": run["model"], "prompt": run["prompt"],
         **{key: run["metrics"][key]["accuracy"] for key in ("equivalent", "non_equivalent")}}
        for run in runs if run["task"] == "T1"
    ]
    for name, rows in tables.items():
        write_csv(output / f"table_{name}.csv", rows)
    return tables


def plot_comparison(runs: list[dict], output: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = {(r["task"], r["model"]): r for r in runs if r["prompt"] == "Base+CoT+one-shot"}
    rows = [
        {"model": model, **{task: indexed[task, model]["metrics"]["overall"]["accuracy"]
                           for task in ("T1", "T2", "T3")}}
        for model in MODELS
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for task, marker, color, offset in [
        ("T1", "o", "#2866a2", .18), ("T2", "s", "#d17625", 0), ("T3", "^", "#318055", -.18)
    ]:
        positions = [i + offset for i in range(len(rows))]
        values = [float(row[task]) for row in rows]
        ax.scatter(values, positions, marker=marker, color=color, label=task, s=45)
        for value, y in zip(values, positions):
            ax.annotate(f"{value:.2f}", (value, y), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9, color=color)
    ax.set(yticks=range(len(MODELS)), yticklabels=MODELS, xlim=(-2, 113),
           xlabel="Overall accuracy (%)", title="SCRBench · Base+CoT+one-shot")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False)
    fig.savefig(output / "figure1.png", dpi=180)
    fig.savefig(output / "figure1.pdf")
    plt.close(fig)
    write_csv(output / "figure1_values.csv", rows)
    return rows


def report_html(summary: dict, tables: dict, output: Path) -> None:
    def table(rows):
        headers = list(rows[0])
        return ('<table><tr>' + ''.join(f"<th>{escape(h)}</th>" for h in headers) + '</tr>'
                + ''.join('<tr>' + ''.join(f"<td>{escape(str(row[h]))}</td>" for h in headers)
                          + '</tr>' for row in rows) + '</table>')
    differences = summary["prediction_differences"]
    html = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>SCRBench raw response reproduction</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#20303a}'
        'table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        'th{background:#eff3f5}img{max-width:100%}h2{margin-top:35px}code{background:#eee}</style><body>',
        '<h1>SCRBench · Reproduction from raw model responses</h1>',
        f'<p>Reference: <a href="{escape(summary["paper_url"])}">published arXiv v1</a>.</p>',
        f'<p><strong>{summary["paper_metrics_matched"]}/{len(summary["comparisons"])} paper metrics match.</strong> '
        f'{len(summary["runs"])} runs and {summary["evaluated_rows"]:,} responses were re-parsed offline. '
        'Scores come from raw response text and the published Parquet evaluation pairs.</p>',
        '<p>The independent Appendix B parser and the current task parser are compared for every response. '
        f'Disagreements: {summary["current_parser_disagreements"]}. '
        'Original error rows count as incorrect. Original files are verified by SHA-256 and left intact.</p>',
        '<h2>Reconstructed cross-task figure</h2><img src="figure1.png" alt="Reconstructed model accuracies">',
        f'<p>{sum(c["matches"] for c in summary["figure_comparisons"])}/15 values match the published figure. '
        'All plotted values come from recomputed metrics. Figure styling is regenerated.</p>',
    ]
    for name, rows in tables.items():
        html.extend([f'<h2>{escape(name)}</h2>', table(rows),
                     f'<p><a href="table_{escape(name)}.csv">Download table CSV</a></p>'])
    html.append('<h2>Changed saved predictions</h2>')
    if differences:
        display = [{key: row[key] for key in
                    ("task", "model", "prompt", "id", "stored_prediction",
                     "reparsed_prediction", "old_correct", "new_correct")} for row in differences]
        html.append(table(display))
        html.append('<p>Details are in <a href="prediction_differences.csv">prediction_differences.csv</a>. '
                    'A non-YES/NO answer in T1 is unparseable even when the historical prediction field contains false.</p>')
    else:
        html.append('<p>No saved prediction differs.</p>')
    html.append('<h2>Evidence</h2><ul><li><a href="summary.json">Summary, metrics and source checksums</a></li>'
                '<li><a href="paper_comparison.csv">Metric-by-metric paper comparison</a></li>'
                '<li><a href="figure1.pdf">Figure PDF</a></li></ul></body></html>')
    (output / "report.html").write_text('\n'.join(html), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-responses",
        action="store_true",
        help="Download and verify the archived responses before scoring.",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiment_results")
    parser.add_argument("--paper-reference", type=Path, default=ROOT / "paper/expected_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.paper_reference.read_text())
    if args.output_dir.exists():
        parser.error("Choose a new output directory to preserve previous reproductions.")
    if args.download_responses:
        restore_responses(PAPER_RESPONSES_REPO, args.results_dir)
    references = {}
    dataset_for_path = {path: dataset for dataset, path in PUBLISHED_REFERENCE_PATHS.items()}
    for hub_path, expected_hash in spec["reference_sha256"].items():
        path = resolve_reference_file(dataset_for_path[hub_path])
        if sha256(path) != expected_hash:
            raise ValueError(f"Reference checksum mismatch: {hub_path}")
        references[hub_path] = read_reference(path)
    # Validate every input before creating outputs.
    for run in spec["runs"]:
        path = args.results_dir / run["csv"]
        if sha256(path) != run["sha256"]:
            raise ValueError(f"Original result checksum mismatch: {path}")
    args.output_dir.mkdir(parents=True)
    runs = []
    differences = []
    comparisons = []
    sources = []
    for run in spec["runs"]:
        path = args.results_dir / run["csv"]
        reference = references["data/t2_test.parquet" if run["task"] == "T2" else "data/t1_t3_test.parquet"]
        scored, rows, changed = score_run(run, path, reference)
        runs.append(scored)
        differences.extend(changed)
        name = f'{run["task"]}__{run["model"]}__{PROMPTS[run["prompt"]]}'
        write_csv(args.output_dir / f"{name}.csv", rows)
        (args.output_dir / f"{name}.metrics.json").write_text(json.dumps(scored, indent=2) + "\n")
        sources.append({"csv": run["csv"], "sha256": run["sha256"], "run_id": run["run_id"]})
        for metric, expected in run["paper_metrics"].items():
            actual = scored["metrics"][metric]["accuracy"]
            comparisons.append({
                "task": run["task"], "model": run["model"], "prompt": run["prompt"],
                "metric": metric, "paper": expected, "recomputed": actual, "matches": actual == expected,
            })
        print(f'{run["task"]} / {run["model"]} / {run["prompt"]}: {scored["metrics"]["overall"]["accuracy"]}%', flush=True)
    summary = {
        "paper_url": spec["paper_url"], "evaluated_rows": sum(r["metrics"]["overall"]["total"] for r in runs),
        "paper_metrics_matched": sum(c["matches"] for c in comparisons),
        "current_parser_disagreements": sum(r["current_parser_disagreements"] for r in runs),
        "runs": runs, "comparisons": comparisons, "prediction_differences": differences,
        "source_files": sources, "reference_sha256": spec["reference_sha256"],
    }
    write_csv(args.output_dir / "paper_comparison.csv", comparisons)
    write_csv(args.output_dir / "prediction_differences.csv", differences)
    tables = build_tables(runs, args.output_dir)
    summary["figure_values"] = plot_comparison(runs, args.output_dir)
    figure_by_model = {row["model"]: row for row in summary["figure_values"]}
    summary["figure_comparisons"] = [
        {**expected,
         "recomputed": figure_by_model[expected["model"]][expected["task"]],
         "matches": figure_by_model[expected["model"]][expected["task"]] == expected["accuracy"]}
        for expected in spec["figure_values"]
    ]
    write_csv(args.output_dir / "figure_comparison.csv", summary["figure_comparisons"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report_html(summary, tables, args.output_dir)
    print(f'Report: {args.output_dir / "report.html"}')
    matched = all(c["matches"] for c in comparisons + summary["figure_comparisons"])
    return 0 if matched and not summary["current_parser_disagreements"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

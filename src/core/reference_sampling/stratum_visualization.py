"""Visualize per-stratum transformation presence counts."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.data_builder.transformations.config import ID_TO_TRANSFORMER

TRANSFORMATION_IDS = sorted(ID_TO_TRANSFORMER)
TRANSFORMATION_NAMES = [ID_TO_TRANSFORMER[tid] for tid in TRANSFORMATION_IDS]

PRESERVING_COLOR = "#1f77b4"
ALTERING_COLOR = "#ff7f0e"
FEASIBLE_MEAN_LINE = "#2ca02c"
EQUAL_TARGET_LINE = "#d62728"


def _presence_counts(sub_df: pd.DataFrame) -> dict[int, int]:
    counts = {tid: 0 for tid in TRANSFORMATION_IDS}
    for _, row in sub_df.iterrows():
        sequence_set = ast.literal_eval(row["paths"])
        present: set[int] = set()
        for path in sequence_set:
            present.update(int(tid) for tid in path)
        for tid in present:
            counts[tid] += 1
    return counts


def _reference_count_lines(n_rows: int, counts: dict[int, int]) -> tuple[float, float, int]:
    feasible_counts = [counts[tid] for tid in TRANSFORMATION_IDS if counts[tid] > 0]
    n_feasible = len(feasible_counts)
    if n_feasible == 0:
        return 0.0, 0.0, 0
    feasible_mean = float(np.mean(feasible_counts))
    equal_target = n_rows / n_feasible
    return feasible_mean, equal_target, n_feasible


@dataclass(frozen=True)
class StratumRecord:
    k: int
    label: str
    is_preserving: bool
    n_rows: int
    counts: dict[int, int]
    feasible_mean: float
    equal_target: float
    n_feasible: int

    @property
    def stratum_id(self) -> str:
        return f"k={self.k} | {self.label}"

    def count_series(self) -> pd.Series:
        return pd.Series(
            {ID_TO_TRANSFORMER[tid]: self.counts[tid] for tid in TRANSFORMATION_IDS},
            name=self.stratum_id,
        )


def build_stratum_records(sample_df: pd.DataFrame) -> list[StratumRecord]:
    records: list[StratumRecord] = []
    for k in sorted(sample_df["k"].unique()):
        for label, is_preserving in [("preserving", True), ("altering", False)]:
            sub_df = sample_df[
                (sample_df["k"] == k)
                & (sample_df["is_semantically_equivalent"] == is_preserving)
            ]
            counts = _presence_counts(sub_df)
            feasible_mean, equal_target, n_feasible = _reference_count_lines(
                len(sub_df), counts
            )
            records.append(
                StratumRecord(
                    k=int(k),
                    label=label,
                    is_preserving=is_preserving,
                    n_rows=len(sub_df),
                    counts=counts,
                    feasible_mean=feasible_mean,
                    equal_target=equal_target,
                    n_feasible=n_feasible,
                )
            )
    return records


def _records_to_count_frame(records: list[StratumRecord]) -> pd.DataFrame:
    return pd.DataFrame({record.stratum_id: record.count_series() for record in records}).T


def _add_reference_lines(ax: Axes, record: StratumRecord) -> None:
    if record.n_feasible == 0:
        return
    ax.axhline(
        record.feasible_mean,
        color=FEASIBLE_MEAN_LINE,
        linestyle="--",
        linewidth=1.5,
        label=f"feasible mean ({record.feasible_mean:.1f})",
    )
    ax.axhline(
        record.equal_target,
        color=EQUAL_TARGET_LINE,
        linestyle=":",
        linewidth=1.5,
        label=f"equal target ({record.equal_target:.1f})",
    )


def _bar_chart_for_record(ax: Axes, record: StratumRecord) -> None:
    counts = [record.counts[tid] for tid in TRANSFORMATION_IDS]
    bar_color = PRESERVING_COLOR if record.is_preserving else ALTERING_COLOR
    bars = ax.bar(TRANSFORMATION_NAMES, counts, color=bar_color, edgecolor="white")
    _add_reference_lines(ax, record)
    ax.set_title(f"{record.stratum_id} (n={record.n_rows})")
    ax.set_ylabel("Row count")
    ax.grid(axis="y", alpha=0.25)
    ymax = max(counts + [record.feasible_mean, record.equal_target, 1])
    ax.set_ylim(0, ymax * 1.15)
    ax.bar_label(bars, labels=[str(v) for v in counts], padding=2, fontsize=8)


def plot_heatmap(records: list[StratumRecord], *, title: str) -> Figure:
    count_df = _records_to_count_frame(records)
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(count_df.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(TRANSFORMATION_NAMES)))
    ax.set_xticklabels(TRANSFORMATION_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(count_df.index)))
    ax.set_yticklabels(count_df.index)
    ax.set_title(title)
    ax.set_xlabel("Transformation")
    ax.set_ylabel("Stratum")
    for row_idx, row_name in enumerate(count_df.index):
        for col_idx, value in enumerate(count_df.loc[row_name].to_numpy()):
            ax.text(col_idx, row_idx, int(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Row count")
    fig.tight_layout()
    return fig


def plot_ten_stratum_panels(records: list[StratumRecord], *, title: str) -> Figure:
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
    fig.suptitle(title, y=1.01, fontsize=14)
    for ax, record in zip(axes.flatten(), records):
        _bar_chart_for_record(ax, record)
        ax.tick_params(axis="x", rotation=45)
    handles = [
        plt.Line2D([0], [0], color=PRESERVING_COLOR, lw=6, label="preserving"),
        plt.Line2D([0], [0], color=ALTERING_COLOR, lw=6, label="altering"),
        plt.Line2D([0], [0], color=FEASIBLE_MEAN_LINE, lw=2, linestyle="--", label="feasible mean"),
        plt.Line2D([0], [0], color=EQUAL_TARGET_LINE, lw=2, linestyle=":", label="equal target"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return fig


def plot_grouped_by_k(records: list[StratumRecord], *, title: str) -> Figure:
    k_values = sorted({record.k for record in records})
    x = np.arange(len(TRANSFORMATION_NAMES))
    width = 0.38
    fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 5), sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        preserving = next(r for r in records if r.k == k and r.is_preserving)
        altering = next(r for r in records if r.k == k and not r.is_preserving)
        preserving_counts = [preserving.counts[tid] for tid in TRANSFORMATION_IDS]
        altering_counts = [altering.counts[tid] for tid in TRANSFORMATION_IDS]
        ax.bar(x - width / 2, preserving_counts, width, color=PRESERVING_COLOR, label="preserving")
        ax.bar(x + width / 2, altering_counts, width, color=ALTERING_COLOR, label="altering")
        ax.axhline(
            preserving.feasible_mean,
            color=FEASIBLE_MEAN_LINE,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.axhline(
            preserving.equal_target,
            color=EQUAL_TARGET_LINE,
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.axhline(
            altering.feasible_mean,
            color=FEASIBLE_MEAN_LINE,
            linestyle="-.",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.axhline(
            altering.equal_target,
            color=EQUAL_TARGET_LINE,
            linestyle=(0, (1, 1)),
            linewidth=1.0,
            alpha=0.7,
        )
        ax.set_title(f"k={k}")
        ax.set_xticks(x)
        ax.set_xticklabels(TRANSFORMATION_NAMES, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Row count")
    fig.suptitle(title)
    handles = [
        plt.Line2D([0], [0], color=PRESERVING_COLOR, lw=6, label="preserving"),
        plt.Line2D([0], [0], color=ALTERING_COLOR, lw=6, label="altering"),
        plt.Line2D([0], [0], color=FEASIBLE_MEAN_LINE, lw=2, linestyle="--", label="feasible mean"),
        plt.Line2D([0], [0], color=EQUAL_TARGET_LINE, lw=2, linestyle=":", label="equal target"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout()
    return fig


def plot_facet_grid(records: list[StratumRecord], *, title: str) -> Figure:
    k_values = sorted({record.k for record in records})
    k_to_col = {k: idx for idx, k in enumerate(k_values)}
    fig, axes = plt.subplots(2, len(k_values), figsize=(4 * len(k_values), 8), sharex=True)
    row_map = {True: 0, False: 1}
    for record in records:
        col = k_to_col[record.k]
        _bar_chart_for_record(axes[row_map[record.is_preserving], col], record)
        axes[row_map[record.is_preserving], col].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("preserving\ncount")
    axes[1, 0].set_ylabel("altering\ncount")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_transformation_trends(records: list[StratumRecord], *, title: str) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    panels = [("preserving", True, PRESERVING_COLOR), ("altering", False, ALTERING_COLOR)]
    k_values = sorted({record.k for record in records})
    for ax, (panel_name, is_preserving, color) in zip(axes, panels):
        panel_records = [r for r in records if r.is_preserving == is_preserving]
        for tid in TRANSFORMATION_IDS:
            name = ID_TO_TRANSFORMER[tid]
            ys = [r.counts[tid] for r in panel_records]
            ax.plot(k_values, ys, marker="o", linewidth=2, label=name)
        ax.set_title(panel_name)
        ax.set_xlabel("k")
        ax.set_ylabel("Row count")
        ax.set_xticks(k_values)
        ax.grid(alpha=0.25)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_all_stratum_views(
    sample_df: pd.DataFrame,
    *,
    dataset_name: str,
) -> list[Figure]:
    records = build_stratum_records(sample_df)
    title_prefix = f"{dataset_name} — transformation presence counts"
    figures = [
        plot_heatmap(records, title=f"{title_prefix} (heatmap)"),
        plot_ten_stratum_panels(records, title=f"{title_prefix} (10 strata)"),
        plot_grouped_by_k(records, title=f"{title_prefix} (grouped by k)"),
        plot_facet_grid(records, title=f"{title_prefix} (2×5 facet grid)"),
        plot_transformation_trends(
            records, title=f"{title_prefix} (count vs k per transformation)"
        ),
    ]
    return figures

"""
Build reference samples that maximize the minimum transformation presence per stratum.

Strata are (k, is_semantically_equivalent), sampled to a fixed size each. Within a
stratum, the "required" transformation ids depend on the label:

- label True: the preserving ids (5-9). Altering ids never occur in True strata.
- label False, k == 1: the altering ids (1-4). Paths are a single altering step,
  so preserving ids never occur there.
- label False, k > 1: all ids (1-9).

For each stratum we search (binary search + randomized greedy cover) for the largest
threshold x such that a sample of the requested size exists in which every required id
appears in at least x rows. The cover rows are kept and the remainder of the sample is
filled uniformly at random without replacement.
"""

from __future__ import annotations

import ast
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import project_config
from core.data_builder.transformations import (
    ID_TO_TRANSFORMER,
)
from core.data_builder.transformations import (
    ALTERING_TRANSFORMATION_IDS as _ALTERING_ID_SET,
)
from core.data_builder.transformations import (
    PRESERVING_TRANSFORMATION_IDS as _PRESERVING_ID_SET,
)

ALL_TRANSFORMATION_IDS = sorted(ID_TO_TRANSFORMER)
ALTERING_TRANSFORMATION_IDS = sorted(_ALTERING_ID_SET)
PRESERVING_TRANSFORMATION_IDS = sorted(_PRESERVING_ID_SET)
TID_LABELS = ID_TO_TRANSFORMER

RANDOM_STATE = 42


@dataclass(frozen=True)
class MinPresenceSpec:
    name: str
    source_filename: str
    output_filename: str
    non_equiv_output_filename: str
    rows_per_stratum: int


StratumKey = tuple[int, bool]


def _normalize_aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    return out


def transformation_presence_vector(sequence_set) -> np.ndarray:
    """Binary vector: 1 if transformation id appears in any path, else 0."""
    if isinstance(sequence_set, str):
        sequence_set = ast.literal_eval(sequence_set)
    present: set[int] = set()
    for transformation_sequence in sequence_set:
        present.update(int(tid) for tid in transformation_sequence)
    return np.array(
        [1.0 if tid in present else 0.0 for tid in ALL_TRANSFORMATION_IDS],
        dtype=float,
    )


def required_transformation_ids(k: int, is_equiv: bool) -> list[int]:
    if is_equiv:
        return PRESERVING_TRANSFORMATION_IDS
    if k == 1:
        return ALTERING_TRANSFORMATION_IDS
    return ALL_TRANSFORMATION_IDS


def _presence_matrix(stratum_df: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        stratum_df["paths"].map(transformation_presence_vector).to_numpy()
    )


def _greedy_cover(
    vectors: np.ndarray,
    required_columns: np.ndarray,
    threshold: int,
    n: int,
    rng: np.random.Generator,
) -> list[int] | None:
    """
    Pick rows until every required id is present in >= threshold of them.

    Returns the chosen row positions (<= n of them) or None if the budget n is
    insufficient. Greedy criterion: the row covering the most still-deficient ids.
    """
    req = vectors[:, required_columns]
    deficits = np.full(req.shape[1], float(threshold))
    available = np.ones(len(vectors), dtype=bool)
    chosen: list[int] = []

    while deficits.max() > 0:
        if len(chosen) >= n:
            return None
        gain = req[:, deficits > 0].sum(axis=1)
        gain[~available] = -1.0
        best_gain = gain.max()
        if best_gain <= 0:
            return None
        candidates = np.flatnonzero(gain == best_gain)
        pick = int(rng.choice(candidates))
        chosen.append(pick)
        available[pick] = False
        deficits = np.maximum(deficits - req[pick], 0.0)

    return chosen


def max_min_presence_selection(
    vectors: np.ndarray,
    required_columns: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    attempts_per_threshold: int = 3,
) -> tuple[int, list[int]]:
    """
    Find the largest threshold x for which a greedy cover of size <= n exists,
    then return n row positions: the cover plus a uniform random fill.
    """
    if len(vectors) < n:
        raise ValueError(f"Need at least {n} rows, found {len(vectors)}")

    supplies = vectors[:, required_columns].sum(axis=0)
    hi = int(min(supplies.min(), n))
    lo = 0
    best_cover: list[int] = []

    def attempt(threshold: int) -> list[int] | None:
        for _ in range(attempts_per_threshold):
            cover = _greedy_cover(vectors, required_columns, threshold, n, rng)
            if cover is not None:
                return cover
        return None

    while lo < hi:
        mid = (lo + hi + 1) // 2
        cover = attempt(mid)
        if cover is not None:
            lo = mid
            best_cover = cover
        else:
            hi = mid - 1

    chosen = list(best_cover)
    remaining_pool = np.setdiff1d(np.arange(len(vectors)), np.array(chosen, dtype=int))
    fill_n = n - len(chosen)
    if fill_n > 0:
        fill = rng.choice(remaining_pool, size=fill_n, replace=False)
        chosen.extend(int(pos) for pos in fill)
    rng.shuffle(chosen)
    return lo, chosen


def _stratum_seed(*, k: int, is_equiv: bool, spec_name: str, random_state: int) -> int:
    key = f"{spec_name}|k={k}|equiv={is_equiv}|seed={random_state}"
    return zlib.crc32(key.encode("utf-8"))


def build_min_presence_sample(
    df: pd.DataFrame,
    spec: MinPresenceSpec,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[StratumKey, int]]:
    """Sample rows_per_stratum rows from every (k, label) stratum of df."""
    df = _normalize_aggregate_df(df)
    sampled_parts: list[pd.DataFrame] = []
    thresholds: dict[StratumKey, int] = {}

    for k in project_config.K_VALUES:
        for is_equiv in (True, False):
            stratum_df = df[
                (df["k"] == k) & (df["is_semantically_equivalent"] == is_equiv)
            ]
            vectors = _presence_matrix(stratum_df)
            required = required_transformation_ids(k, is_equiv)
            required_columns = np.array(
                [ALL_TRANSFORMATION_IDS.index(tid) for tid in required]
            )
            rng = np.random.default_rng(
                _stratum_seed(
                    k=k, is_equiv=is_equiv, spec_name=spec.name, random_state=random_state
                )
            )
            threshold, positions = max_min_presence_selection(
                vectors, required_columns, spec.rows_per_stratum, rng
            )
            thresholds[(k, is_equiv)] = threshold
            sampled_parts.append(stratum_df.iloc[positions].copy())

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return (
        sampled.sample(frac=1, random_state=random_state).reset_index(drop=True),
        thresholds,
    )


def derive_non_equiv_subset(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Dataset 2 is a pure filter of dataset 1: label=False rows only, no resampling."""
    return (
        sample_df[~sample_df["is_semantically_equivalent"]]
        .copy()
        .reset_index(drop=True)
    )


def _stratum_presence_counts(stratum_df: pd.DataFrame) -> np.ndarray:
    return _presence_matrix(stratum_df).sum(axis=0)


def verify_min_presence_samples(
    population_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    non_equiv_df: pd.DataFrame,
    spec: MinPresenceSpec,
    thresholds: dict[StratumKey, int],
) -> None:
    population_df = _normalize_aggregate_df(population_df)

    expected_total = len(project_config.K_VALUES) * 2 * spec.rows_per_stratum
    assert len(sample_df) == expected_total, (len(sample_df), expected_total)

    stratum_sizes = sample_df.groupby(["k", "is_semantically_equivalent"]).size()
    assert (stratum_sizes == spec.rows_per_stratum).all(), stratum_sizes.to_dict()

    print(f"=== Verification: {spec.name} ===")
    print(f"Total rows: {len(sample_df)} "
          f"({spec.rows_per_stratum} per stratum, {2 * spec.rows_per_stratum} per k)")

    for k in project_config.K_VALUES:
        for is_equiv in (True, False):
            samp = sample_df[
                (sample_df["k"] == k)
                & (sample_df["is_semantically_equivalent"] == is_equiv)
            ]
            pop = population_df[
                (population_df["k"] == k)
                & (population_df["is_semantically_equivalent"] == is_equiv)
            ]
            samp_counts = _stratum_presence_counts(samp)
            pop_counts = _stratum_presence_counts(pop)
            required = required_transformation_ids(k, is_equiv)
            threshold = thresholds[(k, is_equiv)]

            if is_equiv:
                altering_cols = [
                    ALL_TRANSFORMATION_IDS.index(tid)
                    for tid in ALTERING_TRANSFORMATION_IDS
                ]
                assert samp_counts[altering_cols].sum() == 0, (
                    f"altering transformation present in (k={k}, True) stratum"
                )

            report = pd.DataFrame(
                {
                    "transformation": [TID_LABELS[tid] for tid in ALL_TRANSFORMATION_IDS],
                    "required": [tid in required for tid in ALL_TRANSFORMATION_IDS],
                    "population_supply": pop_counts.astype(int),
                    "sample_count": samp_counts.astype(int),
                }
            )
            required_counts = [
                samp_counts[ALL_TRANSFORMATION_IDS.index(tid)] for tid in required
            ]
            assert min(required_counts) >= threshold, (k, is_equiv, threshold)

            supply_cap = int(
                min(pop_counts[ALL_TRANSFORMATION_IDS.index(tid)] for tid in required)
            )
            print(f"\nk={k}, is_semantically_equivalent={is_equiv} "
                  f"— min required presence achieved: {int(min(required_counts))} "
                  f"(threshold {threshold}, supply cap {min(supply_cap, spec.rows_per_stratum)})")
            print(report.to_string(index=False))

    # Dataset 2 must be exactly the label=False rows of dataset 1.
    expected_non_equiv = sample_df[~sample_df["is_semantically_equivalent"]]
    assert len(non_equiv_df) == len(expected_non_equiv)
    sort_cols = list(non_equiv_df.columns)
    pd.testing.assert_frame_equal(
        non_equiv_df.sort_values(sort_cols).reset_index(drop=True),
        expected_non_equiv.sort_values(sort_cols).reset_index(drop=True),
    )
    print(f"\nDerived non-equiv subset: {len(non_equiv_df)} rows "
          "(exact filter of the main sample, no recomputation).")

import logging
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from core import project_config
from core.data_builder.base import ProgramPairsByLength
from core.data_builder.cache_manager.corpus_cache import CorpusCache
from core.data_builder.cache_manager.collection_cache import (
    CollectionCache,
)
from core.data_builder.transformations.utils import is_altering_transformation

logger = logging.getLogger(__name__)


class CorpusValidator:
    @staticmethod
    def _ensure_file_logging(*, log_dir: Optional[str] = None, log_filename: str = "aggregated_dataset_validation.log") -> str:
        """
        Attach a FileHandler to this module logger (idempotent) so validation logs are persisted to disk.
        Returns the absolute log path.
        """
        log_dir = log_dir or project_config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.abspath(os.path.join(log_dir, log_filename))

        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if os.path.abspath(getattr(h, "baseFilename", "")) == log_path:
                        return log_path
                except Exception:
                    pass

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        return log_path

    """
    Validations for aggregated per-k datasets.

    Required invariants (as requested):
      1) For each k, sum of all `len(paths)` in aggregate_k equals the total number of valid
         rows across prefix collections (k, alt_pos) for alt_pos in [0..k].
      2) No duplicate (source, destination) pairs between different k aggregates.

    Notes:
      - In addition to (1), this validator also checks that the aggregated set of
        (source, destination, path) triplets matches the valid collection rows exactly.
        This makes failures actionable (missing/extra paths vs duplicated collection rows).
    """

    @staticmethod
    def _tk_col(k: int) -> str:
        return f"t{k}"

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, context: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{context}: missing required columns: {missing}. Found columns: {list(df.columns)}")

    @staticmethod
    def _flatten_aggregate_triplets(df: pd.DataFrame) -> List[Tuple[str, str, Tuple[int, ...]]]:
        triplets: List[Tuple[str, str, Tuple[int, ...]]] = []
        for _, row in df.iterrows():
            src = str(row["source"])
            dst = str(row["destination"])
            sequence_set = row["paths"]
            if not isinstance(sequence_set, list):
                raise ValueError(
                    "Aggregate dataset has invalid 'paths' type. "
                    f"Expected list, got {type(sequence_set)} for (source,destination)=({src!r},{dst!r})."
                )
            for p in sequence_set:
                if not isinstance(p, list):
                    raise ValueError(
                        "Aggregate dataset has invalid path element type. "
                        f"Expected list[int], got {type(p)} for (source,destination)=({src!r},{dst!r})."
                    )
                triplets.append((src, dst, tuple(int(x) for x in p)))
        return triplets

    @staticmethod
    def _validate_sequences_for_row(
        *,
        k: int,
        row_idx: int,
        source: str,
        destination: str,
        is_semantically_equivalent: bool,
        sequence_set: object,
    ) -> Tuple[int, int]:
        """
        Validate user-requested per-row invariants for aggregate_k.

        Returns
        -------
        Tuple[int, int]
            (num_sequences_validated_for_pair, altering_per_sequence_for_pair)
        """
        if not isinstance(sequence_set, list):
            raise ValueError(
                f"[aggregate_paths_rules] k={k} row_idx={row_idx}: invalid 'paths' type: {type(sequence_set)} "
                f"for (source,destination)=({source!r},{destination!r}). Expected list[list[int]]."
            )
        if len(sequence_set) == 0:
            raise ValueError(
                f"[aggregate_paths_rules] k={k} row_idx={row_idx}: empty 'paths' for "
                f"(source,destination)=({source!r},{destination!r}). Expected at least 1 path."
            )

        altering_counts: List[int] = []

        for p_idx, p in enumerate(sequence_set):
            if not isinstance(p, list):
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: invalid path type at paths[{p_idx}]: {type(p)} "
                    f"for (source,destination)=({source!r},{destination!r}). Expected list[int]."
                )

            # Rule: each path length must equal k
            if len(p) != k:
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: invalid path length at paths[{p_idx}]. "
                    f"Expected len(path)=={k}, got {len(p)}. path={p} "
                    f"(source,destination)=({source!r},{destination!r})."
                )

            # Rule: transformation ids must be unique within path
            try:
                ids = [int(x) for x in p]
            except Exception:
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: path contains non-int values at paths[{p_idx}]. "
                    f"path={p} (source,destination)=({source!r},{destination!r})."
                )
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: path contains duplicate transformation ids "
                    f"at paths[{p_idx}]. path={ids} (source,destination)=({source!r},{destination!r})."
                )

            # Rule: number of altering transformations per path must not exceed 1
            num_altering = sum(1 for tid in ids if is_altering_transformation(tid))
            if num_altering > 1:
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: more than 1 altering transformation in "
                    f"paths[{p_idx}]. altering_count={num_altering} path={ids} "
                    f"(source,destination)=({source!r},{destination!r})."
                )
            altering_counts.append(num_altering)

        # Rule: if one path has an altering transformation, all paths must have exactly one (or none)
        if len(set(altering_counts)) != 1:
            raise ValueError(
                f"[aggregate_paths_rules] k={k} row_idx={row_idx}: inconsistent altering presence across paths. "
                f"altering_counts={altering_counts} is_semantically_equivalent={is_semantically_equivalent} "
                f"(source,destination)=({source!r},{destination!r})."
            )

        per_sequence_altering = altering_counts[0]  # all same due to check above

        # Rules relating is_semantically_equivalent to altering counts
        if is_semantically_equivalent:
            # Rule: if is_semantically_equivalent=True => all transformations must be preserving (0 altering per path)
            if per_sequence_altering != 0:
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: is_semantically_equivalent=True but found "
                    f"altering transformation(s) in paths. altering_per_path={per_sequence_altering} "
                    f"(source,destination)=({source!r},{destination!r})."
                )
        else:
            # Rule: if is_semantically_equivalent=False => exactly 1 semantic altering in each path
            if per_sequence_altering != 1:
                raise ValueError(
                    f"[aggregate_paths_rules] k={k} row_idx={row_idx}: is_semantically_equivalent=False but altering "
                    f"count per path is {per_sequence_altering} (expected exactly 1). "
                    f"(source,destination)=({source!r},{destination!r})."
                )

        # Rule: if any path has an altering transformation => is_semantically_equivalent must be False
        if per_sequence_altering == 1 and is_semantically_equivalent:
            # Should already be caught above, but keep it explicit.
            raise ValueError(
                f"[aggregate_paths_rules] k={k} row_idx={row_idx}: found semantic altering transformation but "
                "is_semantically_equivalent=True. (source,destination)=({source!r},{destination!r})."
            )

        return len(sequence_set), per_sequence_altering

    @staticmethod
    def _load_aggregates() -> ProgramPairsByLength:
        return CorpusCache.read_all_k_s()

    @staticmethod
    def _all_collections_exist_for_k(k: int) -> bool:
        return all(CollectionCache.exists(k, alt_pos) for alt_pos in range(0, k + 1))

    @staticmethod
    def validate(
        *,
        validate_row_coverage: bool = True,
        validate_cross_k_overlap: bool = True,
        require_collection_cache_for_row_coverage: bool = True,
        log_dir: Optional[str] = None,
        log_filename: str = "aggregated_dataset_validation.log",
    ) -> None:
        """
        Validate aggregated datasets.

        Parameters
        ----------
        datasets:
            If None, loads aggregates from cache via :class:`CorpusCache`.
        validate_row_coverage:
            Enforce invariant (1) and the triplet set equality check for each k.
        validate_cross_k_overlap:
            Enforce invariant (2).
        require_collection_cache_for_row_coverage:
            If True, missing collection cache files will raise (otherwise, row-coverage is skipped for that k).
        """
        log_path = CorpusValidator._ensure_file_logging(log_dir=log_dir, log_filename=log_filename)
        logger.info(
            "[corpus_validator] start validate: validate_row_coverage=%s validate_cross_k_overlap=%s require_collection_cache_for_row_coverage=%s",
            validate_row_coverage,
            validate_cross_k_overlap,
            require_collection_cache_for_row_coverage,
        )
        logger.info("[corpus_validator] log_file=%s", log_path)
        datasets = CorpusValidator._load_aggregates()
        datasets_dict = datasets.to_dict()

        # Must have exactly k1..k5 (as defined by project_config.K_VALUES / ProgramPairsByLength schema).
        if len(project_config.K_VALUES) != 5:
            raise ValueError(
                f"Expected exactly 5 k values in project_config.K_VALUES, got {project_config.K_VALUES}."
            )
        expected_keys = {f"k{k}" for k in project_config.K_VALUES}
        if set(datasets_dict.keys()) != expected_keys:
            raise ValueError(
                "Aggregated datasets are not exactly k1..k5.\n"
                f"Expected keys: {sorted(expected_keys)}\n"
                f"Found keys: {sorted(datasets_dict.keys())}"
            )
        logger.info("[corpus_validator] loaded aggregates: keys=%s", sorted(datasets_dict.keys()))

        # -------------------------
        # Invariant (2): cross-k overlap
        # -------------------------
        if validate_cross_k_overlap:
            pairs_by_k: Dict[str, Set[Tuple[str, str]]] = {}
            for k in project_config.K_VALUES:
                key = f"k{k}"
                df = datasets_dict[key]
                CorpusValidator._require_columns(df, ["source", "destination"], context=f"aggregate {key}")
                pairs_by_k[key] = set(zip(df["source"].astype(str), df["destination"].astype(str)))

            logger.info(
                "[corpus_validator] cross-k overlap inputs: pair_counts=%s",
                {k: len(v) for k, v in sorted(pairs_by_k.items())},
            )

            keys = sorted(pairs_by_k.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]
                    inter = pairs_by_k[ki] & pairs_by_k[kj]
                    if inter:
                        example_src, example_dst = next(iter(inter))
                        raise ValueError(
                            f"Duplicate (source, destination) detected across aggregates: overlap between {ki} and {kj} "
                            f"for (source, destination)=({example_src!r}, {example_dst!r})."
                        )
            logger.info("[corpus_validator] cross-k overlap check: PASSED (no duplicate (source,destination) across k1..k5)")

        for k in project_config.K_VALUES:
            key = f"k{k}"
            agg_df = datasets_dict[key]
            CorpusValidator._require_columns(
                agg_df, ["source", "destination", "paths", "is_semantically_equivalent"], context=f"aggregate {key}"
            )

            # -------------------------
            # Aggregate per-row path invariants (requested)
            # -------------------------
            total_sequences_checked = 0
            rows_sem_eq_true = 0
            rows_sem_eq_false = 0
            rows_with_altering = 0
            rows_with_no_altering = 0
            total_paths_altering = 0
            total_paths_preserving = 0
            min_sequences_per_pair: int | None = None
            max_sequences_per_pair: int | None = None

            logger.info(
                "[corpus_validator] k=%s aggregate row/path checks: starting (rows=%s)",
                k,
                len(agg_df),
            )
            for row_idx, row in agg_df.iterrows():
                src = str(row["source"])
                dst = str(row["destination"])
                is_sem_eq = bool(row["is_semantically_equivalent"])
                if is_sem_eq:
                    rows_sem_eq_true += 1
                else:
                    rows_sem_eq_false += 1

                num_sequences, altering_per_sequence = CorpusValidator._validate_sequences_for_row(
                    k=k,
                    row_idx=int(row_idx),
                    source=src,
                    destination=dst,
                    is_semantically_equivalent=is_sem_eq,
                    sequence_set=row["paths"],
                )
                total_sequences_checked += num_sequences
                if min_sequences_per_pair is None or num_sequences < min_sequences_per_pair:
                    min_sequences_per_pair = num_sequences
                if max_sequences_per_pair is None or num_sequences > max_sequences_per_pair:
                    max_sequences_per_pair = num_sequences

                if altering_per_sequence == 1:
                    rows_with_altering += 1
                    total_paths_altering += num_sequences
                else:
                    rows_with_no_altering += 1
                    total_paths_preserving += num_sequences

            logger.info(
                "[corpus_validator] k=%s aggregate row/path checks: PASSED (rows=%s total_paths=%s)",
                k,
                len(agg_df),
                total_sequences_checked,
            )
            # More detailed success metrics per rule group (without per-row spam).
            avg_sequences_per_pair = (total_sequences_checked / len(agg_df)) if len(agg_df) else 0.0
            logger.info(
                "[corpus_validator] k=%s aggregate metrics: paths_per_row(min=%s max=%s avg=%.3f)",
                k,
                min_sequences_per_pair,
                max_sequences_per_pair,
                avg_sequences_per_pair,
            )
            logger.info(
                "[corpus_validator] k=%s rule_summary: "
                "path_len==k PASSED; ids_unique_within_path PASSED; altering_per_path<=1 PASSED",
                k,
            )
            logger.info(
                "[corpus_validator] k=%s rule_summary: "
                "rows(is_semantically_equivalent=True)=%s rows(is_semantically_equivalent=False)=%s",
                k,
                rows_sem_eq_true,
                rows_sem_eq_false,
            )
            logger.info(
                "[corpus_validator] k=%s rule_summary: "
                "rows(with_altering_per_path=1)=%s rows(with_altering_per_path=0)=%s "
                "paths(altering)=%s paths(preserving_only)=%s",
                k,
                rows_with_altering,
                rows_with_no_altering,
                total_paths_altering,
                total_paths_preserving,
            )
            logger.info(
                "[corpus_validator] k=%s rule_summary: "
                "is_semantically_equivalent=True => 0 altering PASSED; "
                "is_semantically_equivalent=False => exactly 1 altering PASSED; "
                "altering presence consistent across row PASSED",
                k,
            )

            # -------------------------
            # Invariant (1): row coverage per k
            # -------------------------
            if not validate_row_coverage:
                continue

            if not CorpusValidator._all_collections_exist_for_k(k):
                msg = (
                    f"Missing collection cache files for k={k} (expected alt_pos=0..{k}). "
                    "Row-coverage validation requires the prefix collections to exist on disk."
                )
                if require_collection_cache_for_row_coverage:
                    raise FileNotFoundError(msg)
                logger.warning("[corpus_validator] %s Skipping row-coverage validation for k=%s.", msg, k)
                continue

            # Read valid collection rows and normalize to triplets.
            tk = CorpusValidator._tk_col(k)
            collection_triplets: List[Tuple[str, str, Tuple[int, ...]]] = []
            total_valid_rows = 0
            valid_rows_by_alt_pos: Dict[int, int] = {}
            for alt_pos in range(0, k + 1):
                df = CollectionCache.read(k, alt_pos)
                CorpusValidator._require_columns(
                    df, ["is_valid", "t0", tk, "path"], context=f"collection (k={k}, alt_pos={alt_pos})"
                )
                valid = df[df["is_valid"] == True]
                vcount = len(valid)
                valid_rows_by_alt_pos[alt_pos] = vcount
                total_valid_rows += vcount
                if len(valid) == 0:
                    continue
                for _, r in valid.iterrows():
                    collection_triplets.append(
                        (
                            str(r["t0"]),
                            str(r[tk]),
                            tuple(int(x) for x in r["path"]),
                        )
                    )

            # Compute "sum of all paths len" (requested).
            aggregated_total_paths = int(agg_df["paths"].map(len).sum()) if len(agg_df) else 0
            if aggregated_total_paths != total_valid_rows:
                # Provide a strong hint: duplicates in valid collection rows usually cause this.
                unique_triplets = len(set(collection_triplets))
                dup_rows = total_valid_rows - unique_triplets
                raise ValueError(
                    "Aggregate row-coverage validation failed.\n"
                    f"k={k}\n"
                    f"Expected total_valid_rows (from prefix collections, alt_pos=0..{k}): {total_valid_rows}\n"
                    f"Observed sum(len(paths)) in aggregate_k_{k}: {aggregated_total_paths}\n"
                    f"Unique (source,destination,path) triplets in valid collection rows: {unique_triplets}\n"
                    f"Duplicate valid collection rows (by triplet) implied: {dup_rows}\n"
                    "This indicates missing/extra aggregated paths or duplicated collection rows."
                )

            # Exact set equality check (helps explain mismatches even when counts match).
            collection_set = set(collection_triplets)
            aggregated_set = set(CorpusValidator._flatten_aggregate_triplets(agg_df))
            if collection_set != aggregated_set:
                missing = list(collection_set - aggregated_set)[:5]
                extra = list(aggregated_set - collection_set)[:5]
                raise ValueError(
                    "Aggregate content mismatch between valid collection rows and aggregated paths.\n"
                    f"k={k}\n"
                    f"Missing triplets (up to 5): {missing}\n"
                    f"Extra triplets (up to 5): {extra}"
                )

            logger.info(
                "[corpus_validator] k=%s row-coverage: PASSED (total_valid_rows=%s sum_len_paths=%s unique_pairs=%s)",
                k,
                total_valid_rows,
                aggregated_total_paths,
                len(agg_df),
            )
            logger.info(
                "[corpus_validator] k=%s row-coverage details: valid_rows_by_alt_pos=%s",
                k,
                dict(sorted(valid_rows_by_alt_pos.items())),
            )

        if not validate_row_coverage:
            logger.info("[corpus_validator] row-coverage validation: SKIPPED (validate_row_coverage=False)")
        logger.info("[corpus_validator] SUCCESS: all enabled validations passed")



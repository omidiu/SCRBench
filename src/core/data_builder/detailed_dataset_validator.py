import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from core import project_config
from core.data_builder.cache_manager.transformation_dataset_cache import (
    TransformationDatasetCache,
)
from core.data_builder.generator import DetailsGenerator
from core.data_builder.transformations.base import TransformationType, get_transformations_by_type


@dataclass(frozen=True)
class DetailedDatasetKey:
    k: int
    alt_pos: int


class DetailedDatasetValidator:
    """
    Validate cached detailed datasets in `project_config.CACHE_FOLDER_NAME`.

    Validation runs in three sequential stages:
      - Stage 1: Per (k, alt_pos) validation
      - Stage 2: Intra-k cross validation
      - Stage 3: Inter-k cross validation

    All reports are logged to a file (informational only).
    All critical validations raise exceptions and halt processing.
    """

    _FILENAME_RE = re.compile(r"^detailed_k_(?P<k>\d+)__alt-pos_(?P<alt_pos>\d+)\.csv$")
    _EXPECTED_NUM_DETAILED_DATASET_FILES = 21

    def __init__(
        self,
        *,
        cache_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_filename: str = "detailed_dataset_validation.log",
        include_base_k0: bool = True,
        show_progress: bool = True,
    ):
        self.cache_dir = cache_dir or os.path.join(project_config.PROJECT_ROOT, project_config.CACHE_FOLDER_NAME)
        self.log_dir = log_dir or project_config.LOG_DIR
        self.log_filename = log_filename
        self.include_base_k0 = include_base_k0
        self.show_progress = show_progress

        self._logger = self._setup_logger()

        # Cache these counts once (they are used in Stage 1.2).
        self._num_preserving = len(get_transformations_by_type(TransformationType.PRESERVING))
        self._num_altering = len(get_transformations_by_type(TransformationType.ALTERING))

    def _setup_logger(self) -> logging.Logger:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, self.log_filename)

        logger = logging.getLogger("detailed_dataset_validation")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _progress(self, msg: str) -> None:
        """
        Minimal progress output to stdout so users can see which stage is running.
        This is intentionally *not* detailed; detailed information is written to the log file.
        """
        if self.show_progress:
            print(msg, flush=True)

    def _discover_detailed_files(self) -> List[DetailedDatasetKey]:
        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(
                f"Cache directory not found: '{self.cache_dir}'. "
                f"Expected it to match project_config.CACHE_FOLDER_NAME."
            )

        keys: List[DetailedDatasetKey] = []
        for name in os.listdir(self.cache_dir):
            m = self._FILENAME_RE.match(name)
            if not m:
                continue
            k = int(m.group("k"))
            alt_pos = int(m.group("alt_pos"))
            if not self.include_base_k0 and k == 0 and alt_pos == 0:
                continue
            keys.append(DetailedDatasetKey(k=k, alt_pos=alt_pos))

        keys.sort(key=lambda x: (x.k, x.alt_pos))
        return keys

    @staticmethod
    def _tk_col(k: int) -> str:
        return f"t{k}"

    @staticmethod
    def _as_str_reason(x) -> str:
        # Handles enum (NotValidReason), string, None, NaN.
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "None"
        if hasattr(x, "value"):
            return str(x.value)
        return str(x)

    @staticmethod
    def _preview_text(x, *, max_len: int = 180) -> str:
        """
        Produce a compact, single-line preview for logging long code/text fields.
        - Collapses whitespace/newlines into single spaces.
        - Truncates to `max_len`.
        """
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "None"
        s = " ".join(str(x).split())
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    @staticmethod
    def _applied_type_counts(k: int, alt_pos: int) -> Tuple[int, int]:
        """
        Return (num_applied_preserving, num_applied_altering) for the dataset (k, alt_pos).

        This matches the project’s “one altering transformation at most” design:
          - alt_pos == 0: all k steps are preserving
          - alt_pos > 0: exactly 1 altering + (k-1) preserving
        """
        if k < 0:
            raise ValueError(f"Invalid k={k}. Expected k >= 0.")
        if k == 0:
            return 0, 0
        if alt_pos == 0:
            return k, 0
        return k - 1, 1

    def _load_dataset(self, key: DetailedDatasetKey) -> pd.DataFrame:
        # Use the canonical cache loader so `path` / `tests` / reason conversion is consistent.
        return TransformationDatasetCache.read(key.k, key.alt_pos)

    def _pairs_for_df(self, df: pd.DataFrame, k: int) -> Set[Tuple[str, str]]:
        tk = self._tk_col(k)
        if "t0" not in df.columns or tk not in df.columns:
            raise ValueError(f"Missing required columns for k={k}: expected 't0' and '{tk}'.")
        # Normalize to strings for stable hashing/comparisons.
        return set(zip(df["t0"].astype(str), df[tk].astype(str)))

    def validate(self) -> None:
        keys = self._discover_detailed_files()
        if not keys:
            raise FileNotFoundError(
                f"No detailed dataset CSVs found in '{self.cache_dir}'. "
                f"Expected files matching: detailed_k_*__alt-pos_*.csv"
            )
        if len(keys) != self._EXPECTED_NUM_DETAILED_DATASET_FILES:
            raise RuntimeError(
                f"Expected exactly {self._EXPECTED_NUM_DETAILED_DATASET_FILES} detailed dataset CSVs in '{self.cache_dir}' (got {len(keys)})."
            )

        self._logger.info("============================================================")
        self._logger.info("Detailed Dataset Validation START")
        self._logger.info(f"Cache directory: {os.path.abspath(self.cache_dir)}")
        self._logger.info(f"Discovered {len(keys)} detailed dataset file(s)")
        self._logger.info("============================================================")

        current_stage = "init"
        try:
            current_stage = "STAGE 1"
            self._progress("Running validation: STAGE 1/3 (per (k, alt_pos))")
            self._stage1_per_k_alt_pos(keys)
            self._progress("Completed: STAGE 1/3")

            current_stage = "STAGE 2"
            self._progress("Running validation: STAGE 2/3 (intra-k cross checks)")
            self._stage2_intra_k_cross_validation(keys)
            self._progress("Completed: STAGE 2/3")

            current_stage = "STAGE 3"
            self._progress("Running validation: STAGE 3/3 (inter-k cross checks)")
            self._stage3_inter_k_cross_validation(keys)
            self._progress("Completed: STAGE 3/3")
        except Exception as e:
            # Ensure the failure is captured in the validation log file.
            self._logger.error("============================================================")
            self._logger.error("Detailed Dataset Validation FAILED")
            self._logger.error(f"Error: {e}")
            self._logger.error("============================================================")
            self._progress(f"Validation FAILED during {current_stage}. See log file for details.")
            raise

        self._logger.info("============================================================")
        self._logger.info("Detailed Dataset Validation SUCCESS (all stages passed)")
        self._logger.info("============================================================")
        self._progress("Validation SUCCESS (all stages passed)")

    # -----------------------------
    # Stage 1: Per (k, alt_pos)
    # -----------------------------
    def _stage1_per_k_alt_pos(self, keys: Iterable[DetailedDatasetKey]) -> None:
        self._logger.info("STAGE 1: Per (k, alt_pos) Validation")

        for key in keys:
            df = self._load_dataset(key)
            tk = self._tk_col(key.k)

            self._logger.info(f"  (k={key.k}, alt_pos={key.alt_pos})")
            self._logger.info(f"    Columns: {list(df.columns)}")

            # 📊 Required Reports
            total_rows = len(df)
            if "is_valid" not in df.columns:
                raise ValueError(
                    f"Missing required column 'is_valid' in dataset (k={key.k}, alt_pos={key.alt_pos})."
                )
            valid_rows = int(df["is_valid"].sum())
            invalid_rows = total_rows - valid_rows
            self._logger.info(f"    Total rows: {total_rows}")
            self._logger.info(f"    Total valid rows: {valid_rows}")
            self._logger.info(f"    Total invalid rows: {invalid_rows}")

            if "not_valid_reason" in df.columns:
                invalid_df = df[df["is_valid"] == False].copy()
                if len(invalid_df) == 0:
                    self._logger.info("    Invalid row count grouped by invalid_reason: (none)")
                else:
                    invalid_df["_invalid_reason_norm"] = invalid_df["not_valid_reason"].apply(self._as_str_reason)
                    counts = (
                        invalid_df.groupby("_invalid_reason_norm", dropna=False)
                        .size()
                        .sort_values(ascending=False)
                    )
                    self._logger.info("    Invalid row count grouped by invalid_reason (not_valid_reason):")
                    for reason, cnt in counts.items():
                        self._logger.info(f"      - {reason}: {int(cnt)}")
            else:
                self._logger.info(
                    "    Invalid row count grouped by invalid_reason: column 'not_valid_reason' not present"
                )

            # ❗ 1.1 Uniqueness Check: no duplicate (t0, tk) within the same (k, alt_pos)
            if key.k >= 1:
                if "t0" not in df.columns or tk not in df.columns:
                    raise ValueError(
                        f"Missing required columns for uniqueness check in (k={key.k}, alt_pos={key.alt_pos}). "
                        f"Expected 't0' and '{tk}'."
                    )
                # Apply uniqueness only on valid rows; invalid rows can legitimately share (t0, tk).
                valid_df = df[df["is_valid"] == True]
                if len(valid_df) == 0:
                    dup_count = 0
                    dup_mask = None
                else:
                    dup_mask = valid_df.duplicated(subset=["t0", tk], keep=False)
                    dup_count = int(dup_mask.sum())
                if dup_count > 0:
                    cols = ["t0", tk]
                    has_path = "path" in valid_df.columns
                    if has_path:
                        cols.append("path")

                    dup_rows = valid_df.loc[dup_mask, cols].copy()
                    dup_rows["t0"] = dup_rows["t0"].astype(str)
                    dup_rows[tk] = dup_rows[tk].astype(str)
                    dup_rows["row_id"] = valid_df.index[dup_mask]

                    # For each duplicated pair, report which row_ids it appears on.
                    grouped = (
                        dup_rows.groupby(["t0", tk], dropna=False)["row_id"]
                        .apply(list)
                        .reset_index()
                        .rename(columns={"row_id": "row_ids"})
                    )
                    grouped["duplicate_count"] = grouped["row_ids"].apply(len)
                    grouped = grouped.sort_values("duplicate_count", ascending=False).head(10)

                    examples = grouped.to_dict("records")

                    # Log the duplicated IDs in a readable multi-line format (so they don't get lost in one long line).
                    # NOTE: This is informational and does NOT fail validation.
                    self._logger.warning(
                        "    Stage 1.1 WARNING: duplicate (t0, tk) pairs detected among valid rows (is_valid == True)"
                    )
                    self._logger.warning(
                        f"      Duplicate valid-row count (including repeats): {dup_count}"
                    )
                    self._logger.warning("      Example duplicates (up to 10):")
                    for ex in examples:
                        t0_val = ex.get("t0")
                        tk_val = ex.get(tk)
                        row_ids = ex.get("row_ids", [])
                        t0_preview = self._preview_text(t0_val)
                        tk_preview = self._preview_text(tk_val)
                        self._logger.warning(
                            f"        - duplicate_count={ex.get('duplicate_count')} "
                            f"t0_preview={t0_preview} {tk}_preview={tk_preview}"
                        )
                        for rid in row_ids:
                            if has_path:
                                # Look up the path for each row_id (if present) and log it explicitly.
                                try:
                                    row_path = valid_df.loc[rid, "path"]
                                except Exception:
                                    row_path = None
                                self._logger.warning(f"            row_id={rid} path={row_path}")
                            else:
                                self._logger.warning(f"            row_id={rid} path=(missing 'path' column)")
                else:
                    self._logger.info("    Stage 1.1 passed: no duplicate (t0, tk) pairs among valid rows")

            # ❗ 1.2 Row Count Validation
            if key.k == 0 and key.alt_pos == 0:
                self._logger.info("    Stage 1.2 skipped for base dataset (k=0, alt_pos=0)")
                continue

            if (key.k, key.alt_pos) not in DetailsGenerator.PARENT_MAPPING:
                raise ValueError(
                    f"Stage 1.2 failed: no parent mapping defined for dataset (k={key.k}, alt_pos={key.alt_pos})."
                )

            parent_k, parent_alt_pos, transformation_type = DetailsGenerator.PARENT_MAPPING[(key.k, key.alt_pos)]
            parent_key = DetailedDatasetKey(k=parent_k, alt_pos=parent_alt_pos)

            parent_df = self._load_dataset(parent_key)
            parent_valid_rows = int((parent_df["is_valid"] == True).sum())

            applied_preserving, applied_altering = self._applied_type_counts(parent_k, parent_alt_pos)
            if transformation_type == TransformationType.PRESERVING:
                max_num_valid_transformations = self._num_preserving - applied_preserving
            elif transformation_type == TransformationType.ALTERING:
                max_num_valid_transformations = self._num_altering - applied_altering
            else:
                raise ValueError(
                    f"Stage 1.2 failed: unknown transformation_type={transformation_type} "
                    f"for dataset (k={key.k}, alt_pos={key.alt_pos})."
                )

            if max_num_valid_transformations < 0:
                raise ValueError(
                    "Stage 1.2 failed: computed max_num_valid_transformations is negative.\n"
                    f"Dataset: (k={key.k}, alt_pos={key.alt_pos})\n"
                    f"Parent: (k={parent_k}, alt_pos={parent_alt_pos})\n"
                    f"transformation_type: {transformation_type}\n"
                    f"applied_preserving(parent): {applied_preserving}, total_preserving: {self._num_preserving}\n"
                    f"applied_altering(parent): {applied_altering}, total_altering: {self._num_altering}"
                )

            expected_rows = parent_valid_rows * max_num_valid_transformations
            self._logger.info("    Stage 1.2 expected row-count calculation:")
            self._logger.info(f"      parent: (k={parent_k}, alt_pos={parent_alt_pos})")
            self._logger.info(f"      parent_valid_rows: {parent_valid_rows}")
            self._logger.info(f"      transformation_type: {transformation_type.value}")
            self._logger.info(f"      max_num_valid_transformations: {max_num_valid_transformations}")
            self._logger.info(f"      expected_rows: {expected_rows}")

            if total_rows != expected_rows:
                raise ValueError(
                    "Stage 1.2 failed: row count does not match expected calculation.\n"
                    f"Dataset: (k={key.k}, alt_pos={key.alt_pos})\n"
                    f"Observed total rows: {total_rows}\n"
                    f"Expected total rows: {expected_rows}\n"
                    f"Parent: (k={parent_k}, alt_pos={parent_alt_pos})\n"
                    f"parent_valid_rows: {parent_valid_rows}\n"
                    f"transformation_type: {transformation_type.value}\n"
                    f"max_num_valid_transformations: {max_num_valid_transformations}"
                )

        self._logger.info("STAGE 1: PASSED")

    # -----------------------------
    # Stage 2: Intra-K Cross Validation
    # -----------------------------
    def _stage2_intra_k_cross_validation(self, keys: Iterable[DetailedDatasetKey]) -> None:
        self._logger.info("STAGE 2: Intra-K Cross Validation")

        by_k: Dict[int, List[DetailedDatasetKey]] = {}
        for key in keys:
            if key.k == 0:
                continue
            by_k.setdefault(key.k, []).append(key)

        for k, k_keys in sorted(by_k.items(), key=lambda kv: kv[0]):
            self._logger.info(f"  k={k}")

            alt0 = next((x for x in k_keys if x.alt_pos == 0), None)
            altered = [x for x in k_keys if x.alt_pos != 0]

            # If there is no altered dataset, there's nothing to cross-check.
            if not altered:
                self._logger.info("    Stage 2.1 skipped (no altered alt_pos datasets present)")
                continue

            if alt0 is None:
                raise ValueError(
                    f"Stage 2.1 failed: missing non-altered dataset (k={k}, alt_pos=0), "
                    f"but altered datasets exist: {[x.alt_pos for x in altered]}"
                )

            df0 = self._load_dataset(alt0)
            tk = self._tk_col(k)
            if "is_valid" not in df0.columns:
                raise ValueError(
                    f"Stage 2.1 failed: missing required column 'is_valid' in dataset (k={k}, alt_pos=0)."
                )
            df0_valid = df0[df0["is_valid"] == True]
            non_altered_pairs = self._pairs_for_df(df0_valid, k)

            altered_pairs_union: Set[Tuple[str, str]] = set()
            altered_dfs_valid: Dict[int, pd.DataFrame] = {}
            for key in sorted(altered, key=lambda x: x.alt_pos):
                dfa = self._load_dataset(key)
                if "is_valid" not in dfa.columns:
                    raise ValueError(
                        f"Stage 2.1 failed: missing required column 'is_valid' in dataset (k={k}, alt_pos={key.alt_pos})."
                    )
                dfa_valid = dfa[dfa["is_valid"] == True]
                altered_dfs_valid[key.alt_pos] = dfa_valid
                altered_pairs_union |= self._pairs_for_df(dfa_valid, k)

            intersection = non_altered_pairs & altered_pairs_union
            if intersection:
                example_pairs = list(sorted(intersection))[:10]
                altered_alt_positions = sorted(altered_dfs_valid.keys())

                # Emit a descriptive, multi-line error report to the log file.
                self._logger.error(
                    "    Stage 2.1 FAILED: duplicated (t0, tk) pairs found between non-altered (alt_pos=0) "
                    "and altered (alt_pos!=0) datasets within the same k"
                )
                self._logger.error(f"      k={k}")
                self._logger.error(f"      Duplicate pair count: {len(intersection)}")
                self._logger.error(f"      Altered alt_pos values compared against alt_pos=0: {altered_alt_positions}")
                self._logger.error("      Example duplicate pairs (up to 10):")

                def _row_ids_and_paths_for_pair(dfx: pd.DataFrame, t0_val: str, tk_val: str) -> List[Tuple[int, object]]:
                    # Keep matching consistent with _pairs_for_df (string normalization).
                    if "t0" not in dfx.columns or tk not in dfx.columns:
                        return []
                    m = (dfx["t0"].astype(str) == t0_val) & (dfx[tk].astype(str) == tk_val)
                    if not bool(m.any()):
                        return []
                    sub = dfx.loc[m]
                    if "path" in sub.columns:
                        return list(zip(sub.index.tolist(), sub["path"].tolist()))
                    return [(int(i), None) for i in sub.index.tolist()]

                for i, (t0_val, tk_val) in enumerate(example_pairs, start=1):
                    self._logger.error(f"        - example_{i}:")
                    self._logger.error(f"            t0_preview={self._preview_text(t0_val)}")
                    self._logger.error(f"            {tk}_preview={self._preview_text(tk_val)}")

                    # Occurrences in non-altered (alt_pos=0)
                    occ0 = _row_ids_and_paths_for_pair(df0_valid, t0_val, tk_val)
                    self._logger.error("            occurrences:")
                    self._logger.error(f"              - alt_pos=0: {len(occ0)} row(s)")
                    for rid, p in occ0[:10]:
                        self._logger.error(f"                  row_id={rid} path={p}")
                    if len(occ0) > 10:
                        self._logger.error(f"                  ... ({len(occ0) - 10} more)")

                    # Occurrences in altered datasets, grouped by alt_pos
                    total_altered = 0
                    for alt_pos, dfa in altered_dfs_valid.items():
                        occa = _row_ids_and_paths_for_pair(dfa, t0_val, tk_val)
                        if not occa:
                            continue
                        total_altered += len(occa)
                        self._logger.error(f"              - alt_pos={alt_pos}: {len(occa)} row(s)")
                        for rid, p in occa[:10]:
                            self._logger.error(f"                  row_id={rid} path={p}")
                        if len(occa) > 10:
                            self._logger.error(f"                  ... ({len(occa) - 10} more)")

                    if total_altered == 0:
                        # Shouldn't happen (pair is in union), but keep diagnostics robust.
                        self._logger.error("              - altered: 0 row(s) (unexpected)")

                # Raise a short exception (details are in the log).
                raise ValueError(
                    "Stage 2.1 failed: duplicated (t0, tk) pairs found between alt_pos=0 and alt_pos!=0 datasets.\n"
                    f"k={k}\n"
                    f"Duplicate pair count: {len(intersection)}\n"
                    "See validation log for a detailed, multi-line breakdown (row_id + path)."
                )

            self._logger.info(
                "    Stage 2.1 passed: no (t0, tk) overlap between alt_pos=0 and alt_pos!=0 datasets"
            )

        self._logger.info("STAGE 2: PASSED")

    # -----------------------------
    # Stage 3: Inter-K Cross Validation
    # -----------------------------
    def _stage3_inter_k_cross_validation(self, keys: Iterable[DetailedDatasetKey]) -> None:
        self._logger.info("STAGE 3: Inter-K Cross Validation")

        # pair -> (k, alt_pos) for first observation
        seen: Dict[Tuple[str, str], DetailedDatasetKey] = {}

        for key in keys:
            if key.k == 0:
                continue

            df = self._load_dataset(key)
            if "is_valid" not in df.columns:
                raise ValueError(
                    f"Stage 3.1 failed: missing required column 'is_valid' in dataset (k={key.k}, alt_pos={key.alt_pos})."
                )
            df_valid = df[df["is_valid"] == True]
            pairs = self._pairs_for_df(df_valid, key.k)

            for pair in pairs:
                if pair not in seen:
                    seen[pair] = key
                    continue
                prev = seen[pair]
                if prev.k != key.k:
                    # Emit a descriptive, multi-line error report to the log file.
                    t0_val, tk_val = pair
                    prev_tk_col = self._tk_col(prev.k)
                    curr_tk_col = self._tk_col(key.k)

                    self._logger.error(
                        "  Stage 3.1 FAILED: duplicated (t0, tk) pair found across different k values"
                    )
                    self._logger.error(
                        f"    Pair previews: t0_preview={self._preview_text(t0_val)} "
                        f"value_preview={self._preview_text(tk_val)}"
                    )
                    self._logger.error(
                        f"    First seen in: (k={prev.k}, alt_pos={prev.alt_pos}) "
                        f"using columns (t0, {prev_tk_col})"
                    )
                    self._logger.error(
                        f"    Also found in: (k={key.k}, alt_pos={key.alt_pos}) "
                        f"using columns (t0, {curr_tk_col})"
                    )

                    def _occurrences(dfx: pd.DataFrame, *, tk_col: str) -> pd.DataFrame:
                        if "t0" not in dfx.columns or tk_col not in dfx.columns:
                            return dfx.iloc[0:0]
                        m = (dfx["t0"].astype(str) == str(t0_val)) & (dfx[tk_col].astype(str) == str(tk_val))
                        return dfx.loc[m]

                    def _log_occurrences(label: str, dfx: pd.DataFrame, *, tk_col: str) -> None:
                        # Stage 3 checks are valid-only; log valid-only occurrences to match.
                        if "is_valid" in dfx.columns:
                            dfx = dfx[dfx["is_valid"] == True]
                        occ_all = _occurrences(dfx, tk_col=tk_col)
                        self._logger.error(f"    Occurrences in {label}: {len(occ_all)} row(s)")
                        if len(occ_all) == 0:
                            return

                        has_path = "path" in occ_all.columns
                        show_df = occ_all

                        # Log up to 10 row_ids for readability.
                        max_rows = 10
                        for rid in show_df.index.tolist()[:max_rows]:
                            row = show_df.loc[rid]
                            if has_path:
                                p = row.get("path", None)
                            else:
                                p = "(missing 'path' column)"
                            self._logger.error(f"      row_id={rid} path={p}")

                        if len(show_df) > max_rows:
                            self._logger.error(f"      ... ({len(show_df) - max_rows} more)")

                    # Re-load both datasets to provide row_id/path context.
                    prev_df = self._load_dataset(prev)
                    curr_df = self._load_dataset(key)
                    _log_occurrences(f"(k={prev.k}, alt_pos={prev.alt_pos})", prev_df, tk_col=prev_tk_col)
                    _log_occurrences(f"(k={key.k}, alt_pos={key.alt_pos})", curr_df, tk_col=curr_tk_col)

                    # Raise a short exception (details are in the log).
                    raise ValueError(
                        "Stage 3.1 failed: duplicated (t0, tk) pair found across different k values.\n"
                        f"First seen in: (k={prev.k}, alt_pos={prev.alt_pos})\n"
                        f"Also found in: (k={key.k}, alt_pos={key.alt_pos})\n"
                        "See validation log for a detailed, multi-line breakdown (row_id + path)."
                    )

        self._logger.info("  Stage 3.1 passed: no cross-k duplicate (t0, tk) pairs were found")
        self._logger.info("STAGE 3: PASSED")



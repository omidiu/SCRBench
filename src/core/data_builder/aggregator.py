import logging
import pandas as pd

from typing import Dict, Iterable, List, Tuple

from .base import KDatasets
from .cache_manager.transformation_dataset_cache import TransformationDatasetCache
from .cache_manager.cache_aggregate import CacheAggregate
from core.project_config import K_VALUES


logger = logging.getLogger(__name__)


class Aggregator:
    """
    Aggregates valid rows from all detailed datasets for each k (across alt_pos=0..k)
    into a per-k dataset keyed by unique (t0, tk).
    """

    @staticmethod
    def aggregate_detailed_datasets_group_by_k() -> KDatasets:
        """
        For each k, loads all (k, alt_pos) detailed datasets (alt_pos=0..k),
        filters to `is_valid == True`, and aggregates into unique (source, destination)
        pairs with all transformation-id paths.

        Output columns:
          - source: t0
          - destination: tk
          - paths: List[List[int]]
          - is_semantically_equivalent: bool (True iff alt_pos == 0)

        Notes:
          - If the same (source, destination) appears in both alt_pos=0 and alt_pos!=0
            among valid rows, this raises (should not happen if detailed datasets were validated).
        """
        datasets: Dict[str, pd.DataFrame] = {}
        for k in K_VALUES:
            frames: List[pd.DataFrame] = []
            tk_col = f"t{k}"
            total_valid_rows = 0
            logger.info(f"[aggregate] k={k}: starting (alt_pos=0..{k})")

            for alt_pos in range(0, k + 1):
                df = TransformationDatasetCache.read(k, alt_pos)
                if "is_valid" not in df.columns:
                    raise ValueError(f"Missing required column 'is_valid' in detailed dataset (k={k}, alt_pos={alt_pos}).")
                if "t0" not in df.columns or tk_col not in df.columns:
                    raise ValueError(
                        f"Missing required columns in detailed dataset (k={k}, alt_pos={alt_pos}). "
                        f"Expected 't0' and '{tk_col}'."
                    )
                if "path" not in df.columns:
                    raise ValueError(f"Missing required column 'path' in detailed dataset (k={k}, alt_pos={alt_pos}).")

                valid_df = df[df["is_valid"] == True].copy()
                total_rows = len(df)
                valid_rows = len(valid_df)
                total_valid_rows += valid_rows
                logger.info(f"[aggregate] k={k} alt_pos={alt_pos}: loaded_rows={total_rows} valid_rows={valid_rows}")
                if len(valid_df) == 0:
                    continue

                out = valid_df[["t0", tk_col, "path"]].rename(
                    columns={"t0": "source", tk_col: "destination"}
                )
                out["is_semantically_equivalent"] = (alt_pos == 0)
                frames.append(out)

            if not frames:
                logger.warning(f"[aggregate] k={k}: no valid rows; emitting empty dataset")
                datasets[f"k{k}"] = pd.DataFrame(
                    columns=["source", "destination", "paths", "is_semantically_equivalent"]
                )
                continue

            total_k = pd.concat(frames, ignore_index=True)
            total_k["row_order"] = total_k.index

            # Validate semantic label consistency per (source, destination)
            dup = total_k.duplicated(subset=["source", "destination"], keep=False)
            if dup.any():
                nunq = total_k[dup].groupby(["source", "destination"])["is_semantically_equivalent"].nunique()
                conf = nunq[nunq > 1]
                if not conf.empty:
                    src, dst = conf.index[0]
                    vals = total_k[(total_k.source == src) & (total_k.destination == dst)][
                        "is_semantically_equivalent"
                    ].tolist()
                    logger.error(
                        "[aggregate] k=%s: conflicting is_semantically_equivalent for (source,destination)=(%r,%r): %s",
                        k,
                        src,
                        dst,
                        vals,
                    )
                    raise ValueError(
                        f"Conflicting is_semantically_equivalent for (source, destination)=({src!r}, {dst!r}): {vals}"
                    )

            def _unique_paths_preserve_order(paths: Iterable[List[int]]) -> List[List[int]]:
                seen: set[Tuple[int, ...]] = set()
                out_paths: List[List[int]] = []
                for p in paths:
                    t = tuple(p)
                    if t not in seen:
                        seen.add(t)
                        out_paths.append(list(t))
                return out_paths

            first_idx = (
                total_k.groupby(["source", "destination"], sort=False)["row_order"]
                .min()
                .reset_index(name="first_idx")
            )

            aggregated = (
                total_k.groupby(["source", "destination"], sort=False, as_index=False)
                .agg(
                    paths=("path", _unique_paths_preserve_order),
                    is_semantically_equivalent=("is_semantically_equivalent", "first"),
                )
            )

            # Restore ordering by first occurrence (stable)
            aggregated = aggregated.merge(first_idx, on=["source", "destination"], how="left")
            aggregated = (
                aggregated.sort_values("first_idx", kind="stable")
                .drop(columns="first_idx")
                .reset_index(drop=True)
            )

            datasets[f"k{k}"] = aggregated[["source", "destination", "paths", "is_semantically_equivalent"]]
            CacheAggregate.write_k(datasets[f"k{k}"], k)

            logger.info(
                "[aggregate] k=%s: done. total_valid_rows=%s unique_pairs=%s",
                k,
                total_valid_rows,
                len(datasets[f"k{k}"]),
            )

            del frames

        return KDatasets.from_dict(datasets)

    @staticmethod
    def create_final_file():
        dfs = []
        for k in K_VALUES:
            df = CacheAggregate.read_k(k)
            cols = ['source', 'destination', 'paths', 'is_semantically_equivalent']
            df = df[cols].copy()
            df['k'] = k
            dfs.append(df)
        aggregate_all = pd.concat(dfs, ignore_index=True)
        aggregate_all.insert(0, 'id', range(1, len(aggregate_all)+1))
        CacheAggregate.write_final_file(aggregate_all)


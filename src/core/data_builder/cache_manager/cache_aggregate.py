import ast
import logging
import os
from typing import Dict

from core import project_config
import pandas as pd

from ..base import KDatasets
from core.utils import df_checksum

logger = logging.getLogger(__name__)

K_VALUES = project_config.K_VALUES
K_KEYS = project_config.K_KEYS


class CacheAggregate:
    """
    Handles reading and writing of aggregated CSV files for each k.

    Current aggregate schema (v2):
      - source (str): t0
      - destination (str): tk
      - paths (List[List[int]]): all transformation-id sequences reaching destination
      - is_semantically_equivalent (bool): True iff dataset came from alt_pos=0
    """

    FINAL_FILE_CHECKSUM = '2b0535deaf48b9cc1f7ce471e3bf1a7a419dc61aacf206a7a00684be3aec77e3'
    CACHE_PATH_FINAL = os.path.join(
        project_config.PROJECT_ROOT,
        project_config.CACHE_FOLDER_NAME,
        project_config.FINAL_FILE_NAME,
    )

    K_FILE_CHECKSUMS = {
        1: 'c88dd67b8e0803669e7b9dadabcc3b025818b78a482da7a520e0797a771308a1',
        2: '5e63dcc4433afa246780b5326a2ace9bb07946f94c072cb2056808988e75d74c',
        3: 'e991f5411f8075d4259c389ff778d5df9f81a7cdacbf97a96af17c15566d838e',
        4: '8973ca6dce8305293eeca9b736a18f9a05bdfcdbf5a7317b48c94d5c28285467',
        5: '7fe9d280db47806be79643ea809b9dbb9b0875f71352e999ab1321b339dc7c14'
    }

    CACHE_PATH_PER_K_TEMPLATE = os.path.join(
        project_config.PROJECT_ROOT,
        project_config.CACHE_FOLDER_NAME,
        project_config.CACHE_PATH_AGGREGATE_K_TEMPLATE,
    )

    @staticmethod
    def _prepare_for_save_k(df: pd.DataFrame) -> pd.DataFrame:
        df_to_save = df.copy()
        if "paths" in df_to_save.columns:
            df_to_save["paths"] = df_to_save["paths"].apply(str)
        return df_to_save

    @staticmethod
    def _prepare_after_load(df: pd.DataFrame) -> pd.DataFrame:
        if "paths" in df.columns:
            df["paths"] = df["paths"].apply(ast.literal_eval)
        if "is_semantically_equivalent" in df.columns:
            # Normalize to bool even if CSV loads as string.
            def _as_bool(x):
                if isinstance(x, bool):
                    return x
                if isinstance(x, (int, float)) and not pd.isna(x):
                    return bool(int(x))
                if isinstance(x, str):
                    s = x.strip().lower()
                    if s in {"true", "1"}:
                        return True
                    if s in {"false", "0"}:
                        return False
                return bool(x)

            df["is_semantically_equivalent"] = df["is_semantically_equivalent"].apply(_as_bool)
        return df

    @staticmethod
    def get_path_per_k(k: int) -> str:
        """Get the path to the aggregate CSV file for given k."""
        return CacheAggregate.CACHE_PATH_PER_K_TEMPLATE.format(k=k)

    @staticmethod
    def read_k(k: int) -> pd.DataFrame:
        """Read the aggregate CSV file for given k."""
        path = CacheAggregate.get_path_per_k(k)
        logger.info(f"[cache_aggregate] reading k={k} from {path}")
        df = pd.read_csv(path)
        if df_checksum(df) != CacheAggregate.K_FILE_CHECKSUMS[k]:
            raise ValueError(
                f"Checksum mismatch for key={k}. Cached data is invalid."
            )
        return CacheAggregate._prepare_after_load(df)

    @staticmethod
    def write_k(df: pd.DataFrame, k: int) -> None:
        """Write the dataframe to the aggregate CSV file for given k."""
        path = CacheAggregate.get_path_per_k(k)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_to_save = CacheAggregate._prepare_for_save_k(df)
        if df_checksum(df_to_save) != CacheAggregate.K_FILE_CHECKSUMS[k]:
            raise ValueError(
                f"Checksum mismatch for key={k}. Cached data is invalid."
            )
        df_to_save.to_csv(path, index=False)
        logger.info(f"[cache_aggregate] wrote k={k} to {path} (rows={len(df_to_save)})")

    @staticmethod
    def all_aggregated_exist() -> bool:
        """Check if all aggregate CSV files for k=1..5 exist."""
        cache_paths = [CacheAggregate.get_path_per_k(k) for k in K_VALUES]
        return all(os.path.exists(path) for path in cache_paths)

    @staticmethod
    def read_all_k_s() -> KDatasets:
        """Read in all k-aggregated datasets as a KDatasets model."""
        datasets: Dict[str, pd.DataFrame] = {}
        for k in K_VALUES:
            logger.info(f"Loading cached k-dataset for k={k}")
            datasets[f"k{k}"] = CacheAggregate.read_k(k)
        return KDatasets.from_dict(datasets)

    @staticmethod
    def write_final_file(df: pd.DataFrame) -> None:
        path = CacheAggregate.CACHE_PATH_FINAL
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if df_checksum(df) != CacheAggregate.FINAL_FILE_CHECKSUM:
            raise ValueError(
                'Checksum mismatch for final file'
            )
        df.to_csv(path, index=False)
        logger.info(f"[cache_aggregate] wrote final file to {path}")

    @staticmethod
    def read_final_file():
        path = CacheAggregate.CACHE_PATH_FINAL
        logger.info(f"[cache_aggregate] reading final file")
        df = pd.read_csv(path)
        if df_checksum(df) != CacheAggregate.FINAL_FILE_CHECKSUM:
            raise ValueError(
                'Checksum mismatch for final file.'
            )
        return df
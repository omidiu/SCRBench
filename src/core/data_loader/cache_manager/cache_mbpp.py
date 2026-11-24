import os

import logging

from core import project_config
import pandas as pd

logger = logging.getLogger(__name__)


class CacheMBPP:
    """
    Handles reading and writing of the cached MBPP dataset.
    """

    CACHE_PATH = os.path.join(
        project_config.PROJECT_ROOT,
        project_config.CACHE_FOLDER_NAME,
        project_config.BASE_MBPP_NAME
    )

    @staticmethod
    def is_cached():
        """
        Check if the MBPP dataset cache exists.
        """
        return os.path.exists(CacheMBPP.CACHE_PATH)

    @staticmethod
    def read():
        """
        Read the cached MBPP dataset from disk.
        """
        logger.info(f"Loading cached dataset from {CacheMBPP.CACHE_PATH}")
        return pd.read_pickle(CacheMBPP.CACHE_PATH)

    @staticmethod
    def write(df: pd.DataFrame):
        """
        Write the MBPP dataset to cache.
        """
        logger.info(f"Caching dataset to {CacheMBPP.CACHE_PATH}")
        df.to_pickle(CacheMBPP.CACHE_PATH)


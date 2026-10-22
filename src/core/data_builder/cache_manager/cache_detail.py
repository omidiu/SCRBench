import os
import pandas as pd

from core import project_config

class CacheDetail:
    """
    Handles reading and writing of detailed transformation sequence CSV files
    for specific (k, alt_pos) configurations.
    """

    CACHE_PATH_TEMPLATE = os.path.join(
        project_config.CACHE_FOLDER_NAME,
        project_config.CACHE_PATH_DETAILED_K_TEMPLATE,
    )

    @staticmethod
    def read(k: int, alt_pos: int):
        """
        Read a detailed CSV file corresponding to given k and alt_pos.
        """
        cache_path = CacheDetail.CACHE_PATH_TEMPLATE.format(k=k, alt_pos=alt_pos)
        return pd.read_csv(cache_path)

    @staticmethod
    def write(df: pd.DataFrame, k: int, alt_pos: int):
        """
        Write the dataframe to the detailed cache CSV for given k and alt_pos.
        """
        cache_path = CacheDetail.CACHE_PATH_TEMPLATE.format(k=k, alt_pos=alt_pos)
        df.to_csv(cache_path, index=False)
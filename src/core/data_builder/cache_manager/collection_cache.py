import os
from pathlib import Path
import ast
import logging
import pandas as pd
from typing import Dict, Tuple
from core import project_config
from core.data_builder.invalid_prefix_reason import InvalidPrefixReason
from core.utils import df_checksum


class CollectionCache:
    """
    Handles reading and writing of prefix collections
    for specific (k, alt_pos) configurations.
    Manages data serialization/deserialization for CSV storage.
    """

    FILE_CHECKSUMS: Dict[Tuple[int, int], str] = {
        (0, 0): '6585d43407e09a656cc344b08454f20997a19867074cd08cb1bfc26ebf31348f',
        (1, 0): '123183cd8bd53fdbe0d86bd4d58c55333d82d5ca69a4b20c838fbe096cf53dca',
        (1, 1): '8244b3de99633607a7253493a8499ad0b1e8e33cf2aab2eaaa922a9582e1d536',
        (2, 0): 'ae851bfee8f5e72d27ad9b2b6320372064c0ddcf10a603d77067d8512a87ea01',
        (2, 1): '1e5d57d57b214337ff6ee979d2e7922e686e3054cb4c4a699690d4aaf60ca22b',
        (2, 2): 'cb4bbde42c7a0858162c4d5c64ae6404cfea01d531f88fdc1087f63c2395552c',
        (3, 0): 'cad25d82fb693f8bdcaaffc385830499e329ea604e85172d8ad8ec42e31cd734',
        (3, 1): 'ce547f5198f14ef44be593249a4548d1dcd194e3bd0dd6de69e8ca65444014e8',
        (3, 2): 'd62000ece8a8cc91957953e216e5ba452704b02f67ffab1e8ca12fcf94975067',
        (3, 3): '4ee7830149688c2b3b478535664ddb18335271f8f8c007426dfe97169342f2c3',
        (4, 0): '2676bcece7f711f325bfd6c9de42f36adfb26e9d61305e9ab403ef881f879512',
        (4, 1): 'd98b6d141693ba28d9bdbda6db6593c0e808c0dcc3da0dcb49bef5895d308ff6',
        (4, 2): '32ac646de8f82d1455fc1c808d664b3a1810da029495951ff4fb01ef590cedf4',
        (4, 3): '0417cd8534e1458e3f780036917041c3c6b272f87f840a1943390acabb7190f8',
        (4, 4): '282a635071658f906f7dfa5278dbea72ba05b13b1889551f6abe86fa65304bf2',
        (5, 0): '45ff796d9e003b25c9d7a0f5e9c56cf1928d12821868e4c79ae2066607fac6c2',
        (5, 1): '80677df0962efae53f5922833123c4d2091b7824861e385cf7db0dd3efc7939f',
        (5, 2): '7bb71f9a4a9422f669cffc5853e9bb6be07657f4c17a6bb4c4b6aaaf8a6e547c',
        (5, 3): '8118fca9052a0e205f677e1c7ec66e88e7a624f2e550a75e64fd031fb5b51dd2',
        (5, 4): 'e7acc06a7a05954e67c3afd7fbc701b1cdddf81a090f1ffc6dfbf0c24fc3dcfc',
        (5, 5): 'c4371992229edca04e18e8baa9af6de9273ebd6954f36afd8dfdd07e9a523f48',
    }

    CACHE_PATH_TEMPLATE = os.path.join(
        project_config.PROJECT_ROOT,
        project_config.CACHE_FOLDER_NAME,
        project_config.CACHE_PATH_COLLECTION_TEMPLATE,
    )

    @staticmethod
    def _prepare_for_save(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe for CSV saving by converting complex types to strings.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            DataFrame with converted columns
        """
        df_to_save = df.copy()
        
        # Convert path from list to string
        if 'path' in df_to_save.columns:
            df_to_save['path'] = df_to_save['path'].apply(str)
        
        # Convert tests from list to string
        if 'tests' in df_to_save.columns:
            df_to_save['tests'] = df_to_save['tests'].apply(str)
        
        # Convert not_valid_reason enum to value (string)
        if 'not_valid_reason' in df_to_save.columns:
            df_to_save['not_valid_reason'] = df_to_save['not_valid_reason'].apply(
                lambda x: x.value if isinstance(x, InvalidPrefixReason) else (None if pd.isna(x) or x is None else x)
            )
        
        return df_to_save

    @staticmethod
    def _prepare_after_load(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe after loading from CSV by converting strings back to complex types.
        
        Args:
            df: DataFrame loaded from CSV
            
        Returns:
            DataFrame with converted columns
        """
        # Convert path from string to list
        if 'path' in df.columns:
            df['path'] = df['path'].apply(ast.literal_eval)
        
        # Convert tests from string to list if needed
        if 'tests' in df.columns:
            df['tests'] = df['tests'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (list(x) if not isinstance(x, list) else x)
            )
        
        # Convert not_valid_reason from string to enum if needed
        if 'not_valid_reason' in df.columns:
            def convert_not_valid_reason(x):
                if pd.isna(x) or x is None or x == 'None':
                    return None
                if isinstance(x, str):
                    # Try to match enum value
                    for reason in InvalidPrefixReason:
                        if reason.value == x:
                            return reason
                    # If not found, return the string (for backward compatibility)
                    return x
                return x
            df['not_valid_reason'] = df['not_valid_reason'].apply(convert_not_valid_reason)
        
        return df

    @staticmethod
    def get_cache_path(k: int, alt_pos: int) -> str:
        """
        Get the cache file path for given k and alt_pos.
        
        Args:
            k: Number of transformations
            alt_pos: Position of altering transformation
            
        Returns:
            Absolute path to cache file
        """
        return CollectionCache.CACHE_PATH_TEMPLATE.format(k=k, alt_pos=alt_pos)

    @staticmethod
    def _existing_path(k: int, alt_pos: int) -> str:
        path = CollectionCache.get_cache_path(k, alt_pos)
        if os.path.exists(path):
            return path
        # Older releases stored these same collections under this filename.
        legacy = Path(path).with_name(f"detailed_k_{k}__alt-pos_{alt_pos}.csv")
        return str(legacy) if legacy.is_file() else path

    @staticmethod
    def exists(k: int, alt_pos: int) -> bool:
        """
        Check if cache file exists for given k and alt_pos.
        
        Args:
            k: Number of transformations
            alt_pos: Position of altering transformation
            
        Returns:
            True if cache file exists, False otherwise
        """
        cache_path = CollectionCache._existing_path(k, alt_pos)
        return os.path.exists(cache_path)

    @staticmethod
    def read(k: int, alt_pos: int) -> pd.DataFrame:
        """
        Read a prefix collection from cache for given k and alt_pos.
        
        Args:
            k: Number of transformations
            alt_pos: Position of altering transformation
            
        Returns:
            DataFrame with properly converted data types
            
        Raises:
            FileNotFoundError: If cache file does not exist
        """
        cache_path = CollectionCache._existing_path(k, alt_pos)
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Collection cache (k={k}, alt_pos={alt_pos}) not found at {cache_path}. "
                f"Please build collections in order."
            )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading collection (k={k}, alt_pos={alt_pos}) from cache")
        
        df = pd.read_csv(cache_path)
        if df_checksum(df) != CollectionCache.FILE_CHECKSUMS[(k, alt_pos)]:
            raise ValueError(
                f"Checksum mismatch for key={(k, alt_pos)}. Cached data is invalid."
            )

        df = CollectionCache._prepare_after_load(df)
        
        return df

    @staticmethod
    def write(df: pd.DataFrame, k: int, alt_pos: int):
        """
        Write a prefix collection to cache for given k and alt_pos.
        
        Args:
            df: DataFrame to save
            k: Number of transformations
            alt_pos: Position of altering transformation
        """
        cache_path = CollectionCache.get_cache_path(k, alt_pos)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Prepare dataframe for saving
        df_to_save = CollectionCache._prepare_for_save(df)

        # TODO: CHECK THIS
        if df_checksum(df_to_save) != CollectionCache.FILE_CHECKSUMS[(k, alt_pos)]:
            raise ValueError(
                f"Checksum mismatch for key={(k, alt_pos)}. Cached data is invalid."
            )
        
        # Save the collection to CSV
        df_to_save.to_csv(cache_path, index=False)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Saved collection (k={k}, alt_pos={alt_pos}) to {cache_path}")


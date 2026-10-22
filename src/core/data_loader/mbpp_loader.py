import ast, astor
import warnings

import pandas as pd
from datasets import load_dataset
import logging

from .cache_manager.cache_mbpp import CacheMBPP

logger = logging.getLogger(__name__)

class MBPPDatasetLoader:
    """
    Handles loading, normalizing and caching the MBPP dataset.
    Usage:
        df = MBPPDatasetManager.get_dataset()
    """
    DATASET_NAME = "google-research-datasets/mbpp"
    DATASET_TYPE = "sanitized"

    EXTRA_TEST_CASES = {
        0: 'assert first_repeated_char("aa") == "a"',
        84: 'assert check_expression("()))") == False',
        93: "assert count_divisors(2) == True",
        96: 'assert mul_even_odd([2,3])==6',
        109: 'assert left_rotate(0x80000000, 1) == 0x100000001',
        127: 'assert is_woodall(1) == True',
        140: 'assert is_sublist([1,2,3],[])==True',
        170: 'assert eulerian_num(-1, 0) == 0',
    }

    _cached_df = None

    @staticmethod
    def download_and_prepare():
        logger.info("Downloading dataset and preparing it")
        raw_ds = load_dataset(MBPPDatasetLoader.DATASET_NAME, MBPPDatasetLoader.DATASET_TYPE)
        sources = (
            raw_ds["train"]["code"] + raw_ds["test"]["code"] + raw_ds["validation"]["code"]
        )
        tests = (
            raw_ds['train']['test_list'] +
            raw_ds['test']['test_list'] +
            raw_ds['validation']['test_list']
        )
        df = pd.DataFrame({"source": sources, "tests": tests})
        df['tests'] = df['tests'].apply(lambda x: list(x) if not isinstance(x, list) else x)
        return df

    @staticmethod
    def normalize_sources(df):
        logger.info("Normalizing code sources with astor")
        # MBPP contains some code snippets with regex strings like "\w" or "\.".
        # In Python string literals, these are *invalid escape sequences* and can emit
        # `SyntaxWarning: invalid escape sequence ...` during parsing.
        #
        # We parse MBPP source via `ast.parse`, which uses filename "<unknown>" by default,
        # hence warnings like: "<unknown>:4: SyntaxWarning: invalid escape sequence '\w'".
        #
        # These warnings are harmless for our purposes (we only normalize formatting),
        # so we silence them to avoid noisy notebooks.
        def _normalize_one(row_id: int, code: str) -> str:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=SyntaxWarning,
                    message=r".*invalid escape sequence.*",
                )
                tree = ast.parse(code, filename=f"<mbpp:{row_id}>")
            return astor.to_source(tree)

        df['source'] = [_normalize_one(row_id, code) for row_id, code in df['source'].items()]
        return df

    @staticmethod
    def add_test_cases(df):
        logger.info("Adding extra test cases to selected rows")
        df['tests'] = df['tests'].apply(lambda x: list(x) if not isinstance(x, list) else x)
        for row_id, test_case in MBPPDatasetLoader.EXTRA_TEST_CASES.items():
            if row_id in df.index:
                current_tests = list(df.at[row_id, 'tests'])
                current_tests.append(test_case)
                df.at[row_id, 'tests'] = current_tests
        return df

    @staticmethod
    def deduplicate_on_source(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate dataset rows based on 'source' and log how many were removed.

        Keeps the first occurrence of each unique source.
        """
        if "source" not in df.columns:
            raise ValueError(
                'Expected column "source" to exist for deduplication, but it was not found.'
            )

        duplicate_rows = int(df.duplicated(subset=["source"], keep="first").sum())
        if duplicate_rows:
            before = len(df)
            df = df.drop_duplicates(subset=["source"], keep="first")
            deleted = before - len(df)
            logger.info(
                f'Deduplicated dataset on "source": deleted {deleted} rows '
                f"(kept first occurrence for each source)"
            )

        if not df["source"].is_unique:
            raise ValueError('Deduplication failed: "source" values are still not unique.')

        return df

    @staticmethod
    def prepare():
        if CacheMBPP.is_cached():
            logger.info(f"Found cache at {CacheMBPP.CACHE_PATH}")
            return CacheMBPP.read()
        else:
            df = MBPPDatasetLoader.download_and_prepare()
            df = MBPPDatasetLoader.normalize_sources(df)
            df = MBPPDatasetLoader.add_test_cases(df)
            df = MBPPDatasetLoader.deduplicate_on_source(df)
            
            CacheMBPP.write(df)
            logger.info(f"Dataset cached to {CacheMBPP.CACHE_PATH}")
            return df

    @staticmethod
    def get_dataset():
        if MBPPDatasetLoader._cached_df is None:
            MBPPDatasetLoader._cached_df = MBPPDatasetLoader.prepare()
        return MBPPDatasetLoader._cached_df

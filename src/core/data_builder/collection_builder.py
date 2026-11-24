import os
import logging
import ast
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from .transformations import ID_TO_TRANSFORMER
from .transformations.base import TransformationType, get_transformations_by_type
from .transformations.utils import is_altering_transformation

from .prefix_validation import get_transformation_invalid_reason

from core.project_config import K_VALUES
from core import project_config

from .invalid_prefix_reason import InvalidPrefixReason
from .cache_manager.collection_cache import CollectionCache


class CollectionBuilder:
    """
    Implements incremental generation of prefix collections based on parent collections.
    Each collection (k, alt_pos) is built from its parent collection by applying one transformation.
    """

    @staticmethod
    def get_parent_collection(
        k: int, altering_position: int
    ) -> Tuple[int, int, TransformationType]:
        """Select the parent and transformation pool using Algorithm 1."""
        if type(k) is not int or type(altering_position) is not int:
            raise ValueError("Sequence length and altering position must be integers.")
        if k not in K_VALUES or not 0 <= altering_position <= k:
            raise ValueError(f"Invalid collection: k={k}, altering_position={altering_position}.")
        if altering_position == 0:
            return k - 1, 0, TransformationType.PRESERVING
        if altering_position == k:
            return k - 1, 0, TransformationType.ALTERING
        return k - 1, altering_position, TransformationType.PRESERVING

    @staticmethod
    def _setup_log_dir():
        """Ensure log directory exists. Reads path from project_config."""
        log_dir = project_config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @staticmethod
    def _setup_logger(k: int, alt_pos: int) -> logging.Logger:
        """Setup a logger for a specific (k, alt_pos) configuration."""
        log_dir = CollectionBuilder._setup_log_dir()

        log_file = os.path.join(log_dir, f'collection_builder_k_{k}_alt_pos_{alt_pos}.log')

        logger = logging.getLogger(f'collection_builder_{k}_{alt_pos}')
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def _get_module_logger() -> logging.Logger:
        """Get or create the module-level logger with file handler."""
        logger = logging.getLogger(__name__)

        # Check if file handler already exists
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)

        if not has_file_handler:
            log_dir = CollectionBuilder._setup_log_dir()
            log_file = os.path.join(log_dir, 'collection_builder_module.log')

            logger.setLevel(logging.INFO)

            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Console handler (if not already present)
            has_console_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
            if not has_console_handler:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

        return logger

    @staticmethod
    def _build_columns(k: int) -> List[str]:
        """Build column names for collection with k transformations."""
        columns = [f't{i}' for i in range(k + 1)]  # t0, t1, ..., tk
        columns.extend(['not_valid_reason', 'is_valid', 'path', 'tests'])
        return columns

    @staticmethod
    def _extend_prefix(
        parent_prefix: Dict,
        transformation_ids: List[int],
        transformation_type: TransformationType,
        k: int,
        parent_k: int,
        *,
        validation_timeout: float = 30.0,
    ) -> List[Dict]:
        """Record one result for every eligible transformation of a valid parent."""
        available = {t.id: t for t in get_transformations_by_type(transformation_type)}
        if len(set(transformation_ids)) != len(transformation_ids):
            raise ValueError("Requested transformation IDs must be unique.")
        unknown = set(transformation_ids) - available.keys()
        if unknown:
            raise ValueError(f"Unknown {transformation_type.value} transformation IDs: {sorted(unknown)}.")

        transformation_sequence = parent_prefix["path"]
        tests = parent_prefix["tests"]
        if isinstance(transformation_sequence, str):
            transformation_sequence = ast.literal_eval(transformation_sequence)
        if isinstance(tests, str):
            tests = ast.literal_eval(tests)
        if (
            not isinstance(transformation_sequence, list)
            or len(transformation_sequence) != parent_k
            or any(type(tid) is not int or tid not in ID_TO_TRANSFORMER for tid in transformation_sequence)
            or len(set(transformation_sequence)) != parent_k
        ):
            raise ValueError(f"Invalid parent path for k={parent_k}: {transformation_sequence!r}.")
        if not isinstance(tests, list) or not tests or not all(isinstance(test, str) for test in tests):
            raise ValueError(f"Invalid tests for parent path {transformation_sequence!r}.")

        prefix = {f"t{i}": parent_prefix[f"t{i}"] for i in range(k)}
        if any(not isinstance(code, str) for code in prefix.values()):
            raise ValueError(f"Missing source for parent path {transformation_sequence!r}.")
        old_code = prefix[f"t{parent_k}"]
        altering_applied = (
            transformation_type == TransformationType.ALTERING
            or any(is_altering_transformation(tid) for tid in transformation_sequence)
        )
        eligible = [available[tid] for tid in transformation_ids if tid not in transformation_sequence]
        rows = []
        logger = logging.getLogger(__name__)
        for transformation in eligible:
            new_sequence = [*transformation_sequence, transformation.id]
            new_code = None
            try:
                new_code = transformation.apply(old_code)
                if not isinstance(new_code, str):
                    raise TypeError("Transformation must return Python source.")
            except Exception:
                logger.exception("Transformation failed for path %s.", new_sequence)
                new_code = None
                reason = InvalidPrefixReason.TRANSFORMATION_ERROR
            else:
                try:
                    reason = get_transformation_invalid_reason(
                        old_code, new_code, altering_applied, tests, timeout=validation_timeout
                    )
                except Exception:
                    logger.exception("Validation failed for path %s.", new_sequence)
                    reason = InvalidPrefixReason.VALIDATION_ERROR

            rows.append({
                **prefix,
                f"t{k}": new_code,
                "not_valid_reason": reason.value if reason else None,
                "is_valid": reason is None,
                "path": new_sequence,
                "tests": tests,
            })
        return rows

    @staticmethod
    def build_base_collection(mbpp: pd.DataFrame) -> pd.DataFrame:
        """
        Create the base collection (0, 0) from mbpp.

        Columns: t0, not_valid_reason, is_valid, path, tests
        """
        logger = CollectionBuilder._get_module_logger()
        logger.info(f"Creating base collection (0, 0) from {len(mbpp)} MBPP sources")

        records = []
        for idx, row in mbpp.iterrows():
            tests = row["tests"]
            if isinstance(tests, str):
                tests = ast.literal_eval(tests)
            if not isinstance(tests, list) or not tests or not all(isinstance(test, str) for test in tests):
                raise ValueError(f"Invalid MBPP tests at row {idx!r}.")
            record = {
                't0': row['source'],
                'not_valid_reason': None,
                'is_valid': True,
                'path': [],
                'tests': tests.copy()
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Created base collection with {len(df)} rows")
        return df

    @staticmethod
    def _extend_collection(
        parent_collection_df: pd.DataFrame,
        k: int,
        alt_pos: int,
        transformation_type: TransformationType,
        *,
        validation_timeout: float = 30.0,
    ) -> pd.DataFrame:
        """
        Generate collection (k, alt_pos) from parent collection.

        Args:
            parent_collection_df: Parent collection DataFrame
            k: Current k value
            alt_pos: Current alt_pos value
            transformation_type: Type of transformation to apply
            validation_timeout: Max seconds allowed for running tests when validating a transformation.
        """
        logger = CollectionBuilder._setup_logger(k, alt_pos)

        logger.info(f"Building collection (k={k}, alt_pos={alt_pos})")
        logger.info(f"Parent collection total rows: {len(parent_collection_df)}")

        valid_parent_rows = parent_collection_df[parent_collection_df['is_valid'] == True].copy()
        logger.info(f"Parent collection valid rows: {len(valid_parent_rows)}")

        transformations = get_transformations_by_type(transformation_type)
        logger.info(f"Available transformations of type {transformation_type.value}: {len(transformations)}")
        logger.info("Using sequential generation (single worker)")

        parent_k = k - 1 # TODO: REMOVE THIS
        transformation_ids = [t.id for t in transformations]
        parent_prefixes = valid_parent_rows.to_dict('records')

        del valid_parent_rows

        new_rows = []

        expected_rows = 0
        for parent_prefix in tqdm(
            parent_prefixes, desc=f"Processing (k={k}, alt_pos={alt_pos})"
        ):
            transformation_sequence = parent_prefix["path"]
            if isinstance(transformation_sequence, str):
                transformation_sequence = ast.literal_eval(transformation_sequence)
            expected = sum(tid not in transformation_sequence for tid in transformation_ids)
            row_results = CollectionBuilder._extend_prefix(
                parent_prefix,
                transformation_ids,
                transformation_type,
                k,
                parent_k,
                validation_timeout=validation_timeout,
            )
            if len(row_results) != expected:
                raise ValueError(f"Expected {expected} attempts for path {transformation_sequence!r}, got {len(row_results)}.")
            expected_rows += expected
            new_rows.extend(row_results)

        if len(new_rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows for {(k, alt_pos)}, got {len(new_rows)}.")

        logger.info(f"Built {len(new_rows)} total rows for (k={k}, alt_pos={alt_pos})")

        if new_rows:
            df = pd.DataFrame(new_rows)
            # Ensure all columns are present
            expected_cols = CollectionBuilder._build_columns(k)
            for col in expected_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column in DataFrame: '{col}'")
            df = df[expected_cols]  # Reorder columns
            valid_count = df['is_valid'].sum()
            logger.info(f"Valid rows: {valid_count}, Invalid rows: {len(df) - valid_count}")
        else:
            df = pd.DataFrame(columns=CollectionBuilder._build_columns(k))
            logger.warning(f"No rows built for (k={k}, alt_pos={alt_pos})")

        return df


    @staticmethod
    def build_collection(
        k: int,
        alt_pos: int,
        *, # TODO: check ()
        validation_timeout: float = 30.0,
    ) -> pd.DataFrame:
        """
        Generate collection for (k, alt_pos).

        Args:
            k: Number of transformations
            alt_pos: Position of altering transformation (0 means all preserving)
            validation_timeout: Max seconds allowed for running tests when validating a transformation.
        """
        parent_k, parent_alt_pos, transformation_type = CollectionBuilder.get_parent_collection(k, alt_pos)
        parent_collection_df = CollectionCache.read(parent_k, parent_alt_pos)
        df = CollectionBuilder._extend_collection(
            parent_collection_df,
            k,
            alt_pos,
            transformation_type,
            validation_timeout=validation_timeout,
        )

        del parent_collection_df

        return df

    @staticmethod
    def build_collections(mbpp: pd.DataFrame):
        """
        Generate all collections for k in [1, 5] and alt_pos in [0, k].
        """
        logger = CollectionBuilder._get_module_logger()
        logger.info("Starting generation of all collections")

        # Create base collection (0, 0)
        base_collection_df = CollectionBuilder.build_base_collection(mbpp)
        CollectionCache.write(base_collection_df, k=0, alt_pos=0)

        # Generate collections in order.
        for k in K_VALUES:
            for alt_pos in range(k + 1):
                try:
                    df = CollectionBuilder.build_collection(k, alt_pos)
                    CollectionCache.write(df, k=k, alt_pos=alt_pos)
                except Exception as e:
                    logger.error(f"Error generating collection (k={k}, alt_pos={alt_pos}): {e}")
                    raise

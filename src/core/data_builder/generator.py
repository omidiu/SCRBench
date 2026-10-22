import os
import logging
import ast
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from .transformations.base import TransformationType, get_transformations_by_type
from .transformations.utils import is_altering_transformation

from core.utils import (
    check_codes_are_equal,
    get_transformation_invalid_reason,
)

from core.project_config import K_VALUES
from core import project_config

from .not_valid_reason import NotValidReason
from .cache_manager.transformation_dataset_cache import TransformationDatasetCache


class DetailsGenerator:
    """
    Implements incremental generation of detailed datasets based on parent datasets.
    Each dataset (k, alt_pos) is built from its parent dataset by applying one transformation.
    """

    # Parent mapping: (k, alt_pos) -> (parent_k, parent_alt_pos, transformation_type)
    PARENT_MAPPING: Dict[Tuple[int, int], Tuple[int, int, TransformationType]] = {
        # (0, 0) is the base dataset from mbpp
        
        # alt_pos = 0 (all preserving)
        (1, 0): (0, 0, TransformationType.PRESERVING),
        (2, 0): (1, 0, TransformationType.PRESERVING),
        (3, 0): (2, 0, TransformationType.PRESERVING),
        (4, 0): (3, 0, TransformationType.PRESERVING),
        (5, 0): (4, 0, TransformationType.PRESERVING),
        
        # alt_pos = 1
        (1, 1): (0, 0, TransformationType.ALTERING),
        (2, 1): (1, 1, TransformationType.PRESERVING),
        (3, 1): (2, 1, TransformationType.PRESERVING),
        (4, 1): (3, 1, TransformationType.PRESERVING),
        (5, 1): (4, 1, TransformationType.PRESERVING),
        
        # alt_pos = 2
        (2, 2): (1, 0, TransformationType.ALTERING),
        (3, 2): (2, 2, TransformationType.PRESERVING),
        (4, 2): (3, 2, TransformationType.PRESERVING),
        (5, 2): (4, 2, TransformationType.PRESERVING),
        
        # alt_pos = 3
        (3, 3): (2, 0, TransformationType.ALTERING),
        (4, 3): (3, 3, TransformationType.PRESERVING),
        (5, 3): (4, 3, TransformationType.PRESERVING),
        
        # alt_pos = 4
        (4, 4): (3, 0, TransformationType.ALTERING),
        (5, 4): (4, 4, TransformationType.PRESERVING),
        
        # alt_pos = 5
        (5, 5): (4, 0, TransformationType.ALTERING),
    }

    @staticmethod
    def _setup_log_dir():
        """Ensure log directory exists. Reads path from project_config."""
        log_dir = project_config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @staticmethod
    def _setup_logger(k: int, alt_pos: int) -> logging.Logger:
        """Setup a logger for a specific (k, alt_pos) configuration."""
        log_dir = DetailsGenerator._setup_log_dir()
        
        log_file = os.path.join(log_dir, f'generator_v2_k_{k}_alt_pos_{alt_pos}.log')
        
        logger = logging.getLogger(f'generator_v2_{k}_{alt_pos}')
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
            log_dir = DetailsGenerator._setup_log_dir()
            log_file = os.path.join(log_dir, 'generator_v2_module.log')
            
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
        """Build column names for dataset with k transformations."""
        columns = [f't{i}' for i in range(k + 1)]  # t0, t1, ..., tk
        columns.extend(['not_valid_reason', 'is_valid', 'path', 'tests'])
        return columns

    @staticmethod
    def _create_invalid_rows_for_parent_row(
        parent_row_dict: Dict,
        transformation_ids: List[int],
        transformation_type: TransformationType,
        k: int,
        parent_k: int,
        *,
        reason: NotValidReason,
    ) -> List[Dict]:
        """
        Create invalid rows for all transformations that should be applied to a parent row.

        This is a best-effort fallback to ensure a parent row contributes rows to the
        generated dataset even if processing fails unexpectedly.
        """
        transformations = get_transformations_by_type(transformation_type)
        transformation_map = {t.id: t for t in transformations}
        transformations = [transformation_map[tid] for tid in transformation_ids if tid in transformation_map]
        
        # Get already applied transformation IDs
        already_applied_transformation_ids = parent_row_dict.get('path', [])
        if isinstance(already_applied_transformation_ids, str):
            already_applied_transformation_ids = ast.literal_eval(already_applied_transformation_ids)
        
        # Get the code to transform (tk-1 from parent)
        old_code = parent_row_dict[f't{parent_k}']
        if old_code is None:
            return []
        
        # Get tests from parent row
        tests = parent_row_dict.get('tests', [])
        if isinstance(tests, str):
            tests = ast.literal_eval(tests)
        elif not isinstance(tests, list):
            tests = list(tests)
        
        invalid_rows = []
        
        # Create timeout rows for all transformations that haven't been applied
        for transformation in transformations:
            # Skip if transformation already applied
            if transformation.id in already_applied_transformation_ids:
                continue
            
            # Try to apply transformation to get the new code
            try:
                new_code = transformation.apply(old_code)
            except Exception:
                # If transformation application fails, use old_code as placeholder
                new_code = old_code
            
            # Build timeout row
            new_row = {}
            # Copy intermediate transformations (t1, t2, ..., tk-1)
            for i in range(k):
                col_name = f't{i}'
                if col_name in parent_row_dict:
                    new_row[col_name] = parent_row_dict[col_name]
            
            new_row[f't{k}'] = new_code
            new_row['not_valid_reason'] = reason.value
            new_row['is_valid'] = False
            new_row['path'] = list(already_applied_transformation_ids) + [transformation.id]
            new_row['tests'] = tests

            invalid_rows.append(new_row)

        return invalid_rows

    @staticmethod
    def _process_parent_row(
        parent_row_dict: Dict,
        transformation_ids: List[int],
        transformation_type: TransformationType,
        k: int,
        parent_k: int,
        *,
        validation_timeout: float = 30.0,
    ) -> List[Dict]:
        """
        Process a single parent row and generate all possible new rows from transformations.
        This function is executed sequentially (single process).
        
        Args:
            parent_row_dict: Dictionary representation of parent row
            transformation_ids: List of transformation IDs to try
            transformation_type: Type of transformations (for reconstruction)
            k: Current k value
            parent_k: Parent k value
        
        Returns:
            List of new row dictionaries
        """
        # TODO: refactor this
        try:
            transformations = get_transformations_by_type(transformation_type)
            # Filter to only the IDs we need
            transformation_map = {t.id: t for t in transformations}
            transformations = [transformation_map[tid] for tid in transformation_ids if tid in transformation_map] # TODO: remove -> redundant code
            
            if len(transformations) == 0:
                return [] # TODO: raise error
            
            # Get already applied transformation IDs
            already_applied_transformation_ids = parent_row_dict.get('path', [])
            if isinstance(already_applied_transformation_ids, str):
                already_applied_transformation_ids = ast.literal_eval(already_applied_transformation_ids)
            
            # Get the code to transform (tk-1 from parent)
            old_code = parent_row_dict[f't{parent_k}']
            if old_code is None:
                return [] # TODO: raise error
            
            # Get tests from parent row
            tests = parent_row_dict.get('tests', [])
            if isinstance(tests, str):
                tests = ast.literal_eval(tests)
            elif not isinstance(tests, list):
                tests = list(tests)
            
            # Pre-compute if altering transformation has been applied
            base_altering_applied = any(
                is_altering_transformation(tid) for tid in already_applied_transformation_ids
            )
            
            new_rows = []
            
            # Try each transformation
            for transformation in transformations:
                # Skip if transformation already applied
                if transformation.id in already_applied_transformation_ids:
                    continue
                
                # Determine if altering transformation has been applied
                altering_applied = ( # TODO: CHANGE THIS (I SHOULD ONLY CHECK THIS ONCE)
                    base_altering_applied or 
                    transformation.type == TransformationType.ALTERING
                )
                
                # Apply transformation
                try:
                    new_code = transformation.apply(old_code)
                except Exception:
                    continue # TODO: not_valid_reason
                
                # Early exit: check if codes are equal before expensive test execution
                if check_codes_are_equal(old_code, new_code):
                    # Build new row for same code case
                    new_row = {}
                    for i in range(k):
                        col_name = f't{i}'
                        if col_name in parent_row_dict: # TODO: if not raise error
                            new_row[col_name] = parent_row_dict[col_name]
                    new_row[f't{k}'] = new_code
                    new_row['not_valid_reason'] = NotValidReason.SAME.value
                    new_row['is_valid'] = False
                    new_row['path'] = list(already_applied_transformation_ids) + [transformation.id]
                    new_row['tests'] = tests
                    new_rows.append(new_row)
                    continue
                
                # Determine validity with timeout protection
                try:
                    # TODO: Refactor this - it should be like sequence of validation
                    not_valid_reason = get_transformation_invalid_reason(
                        old_code,
                        new_code,
                        altering_applied,
                        tests,
                        timeout=validation_timeout,
                    )
                    is_valid = not bool(not_valid_reason)
                except Exception as e:
                    not_valid_reason = NotValidReason.VALIDATION_ERROR
                    is_valid = False
                
                # Build new path
                new_path = list(already_applied_transformation_ids) + [transformation.id]
                
                # Build new row
                new_row = {}
                
                # Copy intermediate transformations (t1, t2, ..., tk-1)
                for i in range(k):
                    col_name = f't{i}'
                    if col_name in parent_row_dict:
                        new_row[col_name] = parent_row_dict[col_name]
                
                # Add new transformation result
                new_row[f't{k}'] = new_code
                # Convert enum to value for storage (None stays None)
                # Handle both enum and string cases (for backward compatibility)
                if not_valid_reason is None:
                    new_row['not_valid_reason'] = None
                elif hasattr(not_valid_reason, 'value'):
                    new_row['not_valid_reason'] = not_valid_reason.value
                else:
                    new_row['not_valid_reason'] = not_valid_reason
                new_row['is_valid'] = is_valid
                new_row['path'] = new_path
                new_row['tests'] = tests  # Pass tests through
                
                new_rows.append(new_row)
            
            return new_rows
            
        except Exception as e:
            # Log the error for debugging - don't silently fail
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Error in _process_parent_row: {e}")
            logger.debug(traceback.format_exc())
            # Return empty list on error
            return []

    @staticmethod
    def create_base_dataset(mbpp: pd.DataFrame) -> pd.DataFrame:
        """
        Create the base dataset (0, 0) from mbpp.
        
        Columns: t0, not_valid_reason, is_valid, path, tests
        """
        logger = DetailsGenerator._get_module_logger()
        logger.info(f"Creating base dataset (0, 0) from {len(mbpp)} MBPP sources")
        
        records = []
        for idx, row in mbpp.iterrows():
            record = {
                't0': row['source'],
                'not_valid_reason': None,
                'is_valid': True,
                'path': [],
                'tests': list(row['tests']) if isinstance(row['tests'], list) else list(row['tests']) # todo: CHECK THIS
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created base dataset with {len(df)} rows")
        return df

    @staticmethod
    def _generate_dataset_from_parent(
        parent_df: pd.DataFrame,
        k: int,
        alt_pos: int,
        transformation_type: TransformationType,
        *,
        validation_timeout: float = 30.0,
    ) -> pd.DataFrame:
        """
        Generate dataset (k, alt_pos) from parent dataset.
        
        Args:
            parent_df: Parent dataset DataFrame
            k: Current k value
            alt_pos: Current alt_pos value
            transformation_type: Type of transformation to apply
            validation_timeout: Max seconds allowed for running tests when validating a transformation.
        """
        logger = DetailsGenerator._setup_logger(k, alt_pos)
        
        logger.info(f"Generating dataset (k={k}, alt_pos={alt_pos})")
        logger.info(f"Parent dataset total rows: {len(parent_df)}")
        
        valid_parent_rows = parent_df[parent_df['is_valid'] == True].copy()
        logger.info(f"Parent dataset valid rows: {len(valid_parent_rows)}")
        
        transformations = get_transformations_by_type(transformation_type)
        logger.info(f"Available transformations of type {transformation_type.value}: {len(transformations)}")
        logger.info("Using sequential generation (single worker)")
        
        parent_k = k - 1 # TODO: REMOVE THIS
        transformation_ids = [t.id for t in transformations]
        parent_row_dicts = valid_parent_rows.to_dict('records')
        
        del valid_parent_rows
        
        new_rows = []

        # Sequential processing
        for parent_row_dict in tqdm(
            parent_row_dicts,
            desc=f"Processing (k={k}, alt_pos={alt_pos})",
        ):
            try:
                row_results = DetailsGenerator._process_parent_row(
                    parent_row_dict,
                    transformation_ids,
                    transformation_type,
                    k,
                    parent_k,
                    validation_timeout=validation_timeout,
                )
                if row_results:
                    new_rows.extend(row_results)
            except Exception as e:
                logger.warning(f"Error processing parent row sequentially: {e}")
                invalid_rows = DetailsGenerator._create_invalid_rows_for_parent_row(
                    parent_row_dict,
                    transformation_ids,
                    transformation_type,
                    k,
                    parent_k,
                    reason=NotValidReason.VALIDATION_ERROR,
                )
                if invalid_rows:
                    new_rows.extend(invalid_rows)

        
        del parent_row_dicts
        
        logger.info(f"Generated {len(new_rows)} total rows for (k={k}, alt_pos={alt_pos})")
        
        if new_rows:
            df = pd.DataFrame(new_rows)
            # Ensure all columns are present
            expected_cols = DetailsGenerator._build_columns(k)
            for col in expected_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column in DataFrame: '{col}'")
            df = df[expected_cols]  # Reorder columns
            valid_count = df['is_valid'].sum()
            logger.info(f"Valid rows: {valid_count}, Invalid rows: {len(df) - valid_count}")
        else:
            df = pd.DataFrame(columns=DetailsGenerator._build_columns(k))
            logger.warning(f"No rows generated for (k={k}, alt_pos={alt_pos})")
        
        return df


    @staticmethod
    def generate_dataset(
        k: int,
        alt_pos: int,
        *, # TODO: check ()
        validation_timeout: float = 30.0,
    ) -> pd.DataFrame:
        """
        Generate dataset for (k, alt_pos).
        
        Args:
            k: Number of transformations
            alt_pos: Position of altering transformation (0 means all preserving)
            validation_timeout: Max seconds allowed for running tests when validating a transformation.
        """
        if (k, alt_pos) not in DetailsGenerator.PARENT_MAPPING:
            raise ValueError(f"No parent mapping defined for (k={k}, alt_pos={alt_pos})")
        
        parent_k, parent_alt_pos, transformation_type = DetailsGenerator.PARENT_MAPPING[(k, alt_pos)]
        parent_df = TransformationDatasetCache.read(parent_k, parent_alt_pos)
        df = DetailsGenerator._generate_dataset_from_parent(
            parent_df,
            k,
            alt_pos,
            transformation_type,
            validation_timeout=validation_timeout,
        )
        
        del parent_df
        
        return df

    @staticmethod
    def generate_all_datasets(mbpp: pd.DataFrame):
        """
        Generate all datasets for k in [1, 5] and alt_pos in [0, k].
        """
        logger = DetailsGenerator._get_module_logger()
        logger.info("Starting generation of all datasets")
        
        # Create base dataset (0, 0)
        base_df = DetailsGenerator.create_base_dataset(mbpp)
        TransformationDatasetCache.write(base_df, k=0, alt_pos=0)
        
        # Generate datasets in order.
        for k in K_VALUES:
            for alt_pos in range(k + 1):
                try:
                    df = DetailsGenerator.generate_dataset(k, alt_pos)
                    TransformationDatasetCache.write(df, k=k, alt_pos=alt_pos)
                except Exception as e:
                    logger.error(f"Error generating dataset (k={k}, alt_pos={alt_pos}): {e}")
                    raise


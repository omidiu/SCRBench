"""
Central configuration for this repository.

This module is intended to be the single source of truth for:
- Cache paths / filenames
- Logging paths
- (Future) experiment configuration knobs
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    """Repository root: nearest ancestor of this file that contains pyproject.toml."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate project root: no pyproject.toml found above "
        f"{Path(__file__).resolve()}"
    )


PROJECT_ROOT = _find_project_root()

# ============================================================
# User-configurable settings
# These values are expected to be changed depending on the
# environment.
# ============================================================
CACHE_FOLDER_NAME = "cache"
LOG_DIR = "logs"
EXPERIMENT_RESULTS_DIR = "experiment_results"
CORPUS_FILE_NAME = "corpus.csv"
BASE_MBPP_NAME = "cached_mbpp_data.pkl"


# ============================================================
# Internal / derived constants
# These values are part of the project's internal conventions
# and normally should not be changed unless the code logic
# itself is updated accordingly.
# ============================================================
K_VALUES = [1, 2, 3, 4, 5]
K_KEYS = [f"k{k}" for k in K_VALUES]
CACHE_PATH_COLLECTION_TEMPLATE = "collection_k_{k}__alt_pos_{alt_pos}.csv"
CACHE_PATH_AGGREGATE_K_TEMPLATE = "aggregate_k_{k}.csv"


"""Published SCRBench evaluation pairs on Hugging Face."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from core import project_config

from .enums import ReferenceDataset

PUBLISHED_DATASET_REPO = "omidiu/SCRBench"
PUBLISHED_DATASET_REVISION = "main"
PUBLISHED_REFERENCE_PATHS = {
    ReferenceDataset.T1_T3: "data/t1_t3_test.parquet",
    ReferenceDataset.T2: "data/t2_test.parquet",
}


def resolve_reference_file(dataset: ReferenceDataset, folder_path: str | None = None) -> Path:
    """Return the Parquet file for a published evaluation subset.

    The default downloads the files from the dataset's main branch. A folder path
    selects a local Parquet file with the same name, for a newly generated subset.
    """
    hub_path = PUBLISHED_REFERENCE_PATHS[dataset]
    if folder_path:
        local = Path(folder_path)
        if not local.is_absolute():
            local = Path(project_config.PROJECT_ROOT) / local
        candidate = local / Path(hub_path).name
        if not candidate.is_file():
            raise FileNotFoundError(f"Parquet reference not found: {candidate}")
        return candidate
    return Path(hf_hub_download(
        PUBLISHED_DATASET_REPO,
        hub_path,
        repo_type="dataset",
        revision=PUBLISHED_DATASET_REVISION,
    ))

"""Restore and verify the archived model responses used by the paper."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import hf_hub_download

PAPER_RESPONSES_REPO = "omidiu/SCRBench-results"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore_responses(repo_id: str, output_dir: Path, revision: str = "main") -> int:
    manifest_path = hf_hub_download(
        repo_id, "manifest.json", repo_type="dataset", revision=revision,
    )
    manifest = json.loads(Path(manifest_path).read_text())
    destination_root = output_dir.resolve()
    restored = 0
    for run in manifest["runs"]:
        raw = run["raw_response"]
        destination = (destination_root / raw["original_relative_path"]).resolve()
        if not destination.is_relative_to(destination_root):
            raise ValueError("Archive contains a path outside the selected output directory.")
        if destination.exists():
            if sha256(destination) != raw["uncompressed_sha256"]:
                raise ValueError(f"Existing file differs from the archive: {destination}")
            print(f"Already verified: {run['task']} / {run['model']} / {run['prompt']}")
            continue
        downloaded = Path(hf_hub_download(
            repo_id, raw["hub_path"], repo_type="dataset", revision=revision,
        ))
        if sha256(downloaded) != raw["compressed_sha256"]:
            raise ValueError(f"Downloaded archive checksum mismatch: {downloaded.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            size = 0
            with gzip.open(downloaded, "rb") as source, temporary_path.open("wb") as target:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > raw["uncompressed_bytes"]:
                        raise ValueError("Archive exceeds its declared uncompressed size.")
                    target.write(block)
            if size != raw["uncompressed_bytes"] or sha256(temporary_path) != raw["uncompressed_sha256"]:
                raise ValueError(f"Restored CSV checksum mismatch: {destination}")
            # Exclusive destination creation preserves existing files if another download finishes first.
            os.link(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        restored += 1
        print(f"Restored: {run['task']} / {run['model']} / {run['prompt']}")
    return restored

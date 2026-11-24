"""Download the 25 archived SCRBench paper runs and restore their original CSV paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.experiments.paper_responses import PAPER_RESPONSES_REPO, restore_responses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=PAPER_RESPONSES_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    args = parser.parse_args()
    count = restore_responses(args.repo_id, args.output_dir)
    print(f"Restored {count} new CSV files; existing files were checksum-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

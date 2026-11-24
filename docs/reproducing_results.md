# Reproduce the paper results

Rebuild the paper's tables and Figure 1 from the archived model responses. The first run requires
internet access.

## Run the reproduction

[Install SCRBench](../README.md#installation), then run this from the repository root. Use a new
output directory for each run.

```bash
uv run python scripts/reproduce_paper_results.py \
  --download-responses --output-dir dist/paper-reproduction
```

## Outputs and verification

Open `dist/paper-reproduction/report.html` for the tables, Figure 1, and comparison with the paper.

To rerun offline after the inputs are cached, set `HF_HUB_OFFLINE=1` and omit `--download-responses`.

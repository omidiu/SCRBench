# Running experiments

## Setup

[Install SCRBench](../README.md#installation). From the repository root, create `.env` if needed.

```bash
test -f .env || cp .env.sample .env
```

Set `OPENROUTER_API_KEY` in `.env`. Requests may incur charges.

### Redis caching

Redis is optional. With caching enabled, later runs can reuse responses for matching requests.
To start a local Redis service, run this from the repository root.

```bash
docker compose up -d redis
```

The local service uses `localhost:6379`, database `0`, with no password. For an existing Redis
service, set the `REDIS_*` values in `.env` using [`.env.sample`](../.env.sample) as a reference.

## Run an experiment

Paste this into Bash or Zsh from the repository root. It downloads the public evaluation subset
and runs T1 on ten pairs.

```bash
uv run --locked python - <<'PY'
from core.experiments import (
    ExperimentConfig,
    LLMModel,
    ReferenceDataset,
    TaskType,
    get_prompt_spec,
    run_experiment,
)

task = TaskType.T1
config = ExperimentConfig(
    model_name=LLMModel.QWEN_3_235B_A22B_2507,
    prompt_spec=get_prompt_spec(task, "base_cot_one_shot"),
    task_type=task,
    reference_file_name=ReferenceDataset.T1_T3,
    max_rows=10,
)
results_csv, report_json = run_experiment(config)
print(results_csv)
print(report_json)
PY
```

## Choose a task and input

Change `task` and `reference_file_name` together. Set `max_rows=None` to run the full subset.

| Task | Reference dataset | Prediction |
| --- | --- | --- |
| `TaskType.T1` | `ReferenceDataset.T1_T3` | Semantic equivalence |
| `TaskType.T2` | `ReferenceDataset.T2` | Altering transformation ID |
| `TaskType.T3` | `ReferenceDataset.T1_T3` | Ordered transformation sequence |

Choose a model from [`LLMModel`](../src/core/experiments/enums.py). Prompt IDs are `base`,
`base_cot`, `base_one_shot`, and `base_cot_one_shot`.

## Outputs

`experiment_results/` contains the response CSV, run report, and `reference_llm_judge_<run_id>.json`
with per-pair correctness.

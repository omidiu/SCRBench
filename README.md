# SCRBench

[![arXiv](https://img.shields.io/badge/arXiv-2610.41562-b31b1b.svg)](https://arxiv.org/abs/2610.41562)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**SCRBench** is a benchmark for evaluating how language models reason about the **semantic effects
of code changes**. It contains **67,658 Python program pairs** derived from **418 MBPP programs**,
annotated with test-based equivalence labels and valid transformation sequences.

The paper
[*SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code*](https://arxiv.org/abs/2610.41562)
describes the benchmark and its evaluation.

## Installation

Use **Python 3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/omidiu/SCRBench.git
cd SCRBench
uv sync --locked
```

## Guides

| Goal | Guide |
| --- | --- |
| Load benchmark pairs or find saved model responses | [Dataset access](docs/dataset_access.md) |
| Recompute the paper's scores and figure from saved responses | [Reproduce the paper results](docs/reproducing_results.md) |
| Evaluate a model through OpenRouter | [Run experiments](docs/running_experiments.md) |

## Citation and license

Please cite [the SCRBench paper](https://arxiv.org/abs/2610.41562) using [CITATION.cff](CITATION.cff), and acknowledge the upstream [MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp).

The source code is licensed under [MIT](LICENSE). The benchmark data and saved
responses are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

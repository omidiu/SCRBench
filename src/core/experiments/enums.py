"""
Experiment configuration enums.
"""

from enum import Enum

# StrEnum for seamless string conversion when passing to tasks/prompts
# Using (str, Enum) for compatibility with Python < 3.11


class TaskType(str, Enum):
    """Benchmark experiment tasks."""

    T1 = "T1_semantic_preserving"
    T2 = "T2_determine_altering_transformation"
    T3 = "T3_predict_transformation_path"


class LLMModel(str, Enum):
    """OpenRouter model ids used in experiments."""

    CLAUDE_3_7_SONNET = "anthropic/claude-3.7-sonnet"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    KAT_CODER_PRO_FREE = "kwaipilot/kat-coder-pro:free"
    LLAMA_3_8B_INSTRUCT = "meta-llama/llama-3-8b-instruct"
    CODESTRAL_2508 = "mistralai/codestral-2508"
    XIAOMI_MIMO_V2_5 = "xiaomi/mimo-v2.5"
    GPT_3_5_TURBO = "openai/gpt-3.5-turbo"
    GPT_3_5_sTURBO_16K = "openai/gpt-3.5-turbo-16k"
    GPT_4 = "openai/gpt-4"
    GROK_CODE_FAST_1 = "x-ai/grok-code-fast-1"
    QWEN_3_7_PLUS = "qwen/qwen3.7-plus"

    GPT_4_1_NANO = "openai/gpt-4.1-nano"
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/llama-3.1-8b-instruct"

    QWEN_3_235B_A22B_2507 = "qwen/qwen3-235b-a22b-2507"
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
    MISTRAL_SMALL_24B_INSTRUCT_2501 = "mistralai/mistral-small-24b-instruct-2501"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"



class ReferenceDataset(str, Enum):
    """CSV datasets used as experiment reference inputs."""

    AGGREGATE_MIXED_LABELS = "aggregate_mixed_labels.csv"
    AGGREGATE_NON_EQUIV_ONLY = "aggregate_non_equiv_only.csv"

    # 200 rows per k (5 × 200 = 1,000). Naive = uniform random; stratified = presence-matched.
    AGGREGATE_MIXED_LABELS_SAMPLE_NAIVE_200_PER_K_1000 = (
        "aggregate_mixed_labels_sample_naive_200_per_k_1000.csv"
    )
    AGGREGATE_MIXED_LABELS_SAMPLE_STRATIFIED_200_PER_K_1000 = (
        "aggregate_mixed_labels_sample_stratified_200_per_k_1000.csv"
    )
    AGGREGATE_NON_EQUIV_ONLY_SAMPLE_NAIVE_200_PER_K_1000 = (
        "aggregate_non_equiv_only_sample_naive_200_per_k_1000.csv"
    )
    AGGREGATE_NON_EQUIV_ONLY_SAMPLE_STRATIFIED_200_PER_K_1000 = (
        "aggregate_non_equiv_only_sample_stratified_200_per_k_1000.csv"
    )
    # 450 rows per (k, label) stratum (10 strata = 4,500). Max-min transformation
    # presence per stratum. The non-equiv file is the label=False subset of the
    # mixed-labels sample (450 per k = 2,250), derived by filtering, not resampled.
    AGGREGATE_MIXED_LABELS_SAMPLE_MIN_PRESENCE_450_PER_STRATUM_4500 = (
        "aggregate_mixed_labels_sample_min_presence_450_per_stratum_4500.csv"
    )
    AGGREGATE_NON_EQUIV_ONLY_SAMPLE_MIN_PRESENCE_DERIVED_450_PER_K_2250 = (
        "aggregate_non_equiv_only_sample_min_presence_derived_450_per_k_2250.csv"
    )

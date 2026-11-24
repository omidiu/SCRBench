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
    """The five evaluated models, with OpenRouter IDs and paper display names.

    ``value`` remains the API identifier; ``display_name`` is the paper label.
    """

    display_name: str

    def __new__(cls, model_id: str, display_name: str):
        member = str.__new__(cls, model_id)
        member._value_ = model_id
        member.display_name = display_name
        return member

    QWEN_3_235B_A22B_2507 = ("qwen/qwen3-235b-a22b-2507", "Qwen3-235B")
    GPT_OSS_120B = ("openai/gpt-oss-120b", "GPT-OSS-120B")
    MISTRAL_SMALL_24B_INSTRUCT_2501 = (
        "mistralai/mistral-small-24b-instruct-2501", "Mistral-Small-24B"
    )
    LLAMA_3_1_8B_INSTRUCT = ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B")
    GPT_4_1_NANO = ("openai/gpt-4.1-nano", "GPT-4.1-nano")


class ReferenceDataset(str, Enum):
    """Published evaluation subsets in omidiu/SCRBench."""

    T1_T3 = "t1_t3_test.parquet"  # 4,500 pairs: 450 per (sequence length, label) stratum.
    T2 = "t2_test.parquet"  # 2,250 non-equivalent pairs, filtered from T1_T3.

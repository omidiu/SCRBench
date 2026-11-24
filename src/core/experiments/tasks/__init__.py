from .base import TaskExecutionResult, TaskRowOutput
from .T1_detect_semantic_equivalence import (
    PROMPT_SPECS as t1_prompt_specs,
    T1DetectSemanticEquivalenceTask,
)
from .T2_diagnose_semantic_change import (
    PROMPT_SPECS as t2_prompt_specs,
    T2DiagnoseSemanticChangeTask,
)
from .T3_reconstruct_transformation_sequence import (
    PROMPT_SPECS as t3_prompt_specs,
    T3ReconstructTransformationSequenceTask,
)
from ..enums import TaskType
from ..prompt_spec import PromptSpec

TASK_REGISTRY = {
    T1DetectSemanticEquivalenceTask.NAME: T1DetectSemanticEquivalenceTask,
    T2DiagnoseSemanticChangeTask.NAME: T2DiagnoseSemanticChangeTask,
    T3ReconstructTransformationSequenceTask.NAME: T3ReconstructTransformationSequenceTask,
}

registered_prompt_specs = {
    T1DetectSemanticEquivalenceTask.NAME: t1_prompt_specs,
    T2DiagnoseSemanticChangeTask.NAME: t2_prompt_specs,
    T3ReconstructTransformationSequenceTask.NAME: t3_prompt_specs,
}


def get_prompt_spec(task_type: TaskType | str, spec_id: str) -> PromptSpec:
    """Return the registered PromptSpec for a task and variant id (e.g. ``"base_cot"``)."""
    task_name = task_type.value if isinstance(task_type, TaskType) else task_type
    specs = registered_prompt_specs.get(task_name)
    if specs is None:
        raise ValueError(
            f"Unknown task_type={task_name!r}. Known tasks: {sorted(registered_prompt_specs.keys())}"
        )
    if spec_id not in specs:
        raise ValueError(
            f"Unknown prompt spec id={spec_id!r} for task {task_name!r}. "
            f"Available: {sorted(specs.keys())}"
        )
    return specs[spec_id]


def resolve_prompt_spec(task_type: TaskType, spec: PromptSpec) -> PromptSpec:
    """Validate *spec* belongs to *task_type* and return the canonical registry instance."""
    canonical = get_prompt_spec(task_type, spec.id)
    if spec is not canonical:
        raise ValueError(
            f"Prompt spec {spec.id!r} does not belong to task {task_type.value!r}."
        )
    return canonical


__all__ = [
    "T1DetectSemanticEquivalenceTask",
    "T2DiagnoseSemanticChangeTask",
    "T3ReconstructTransformationSequenceTask",
    "TaskExecutionResult",
    "TaskRowOutput",
    "TASK_REGISTRY",
    "get_prompt_spec",
    "registered_prompt_specs",
    "resolve_prompt_spec",
]

from .base import TaskExecutionResult, TaskRowOutput
from .T1_semantic_preserving import (
    PROMPT_SPECS as t1_prompt_specs,
    T1SemanticPreservingTask,
)
from .T2_determine_altering import (
    PROMPT_SPECS as t2_prompt_specs,
    T2DetermineAlteringTask,
)
from .T3_predict_transformation_path import (
    PROMPT_SPECS as t3_prompt_specs,
    T3PredictTransformationPathTask,
)
from ..enums import TaskType
from ..prompt_spec import PromptSpec

TASK_REGISTRY = {
    T1SemanticPreservingTask.NAME: T1SemanticPreservingTask,
    T2DetermineAlteringTask.NAME: T2DetermineAlteringTask,
    T3PredictTransformationPathTask.NAME: T3PredictTransformationPathTask,
}

registered_prompt_specs = {
    T1SemanticPreservingTask.NAME: t1_prompt_specs,
    T2DetermineAlteringTask.NAME: t2_prompt_specs,
    T3PredictTransformationPathTask.NAME: t3_prompt_specs,
}


def get_prompt_spec(task_type: TaskType | str, spec_id: str) -> PromptSpec:
    """Return the registered PromptSpec for a task and variant id (e.g. ``"T2_v4"``)."""
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
    return get_prompt_spec(task_type, spec.id)


__all__ = [
    "T1SemanticPreservingTask",
    "T2DetermineAlteringTask",
    "T3PredictTransformationPathTask",
    "TaskExecutionResult",
    "TaskRowOutput",
    "TASK_REGISTRY",
    "get_prompt_spec",
    "registered_prompt_specs",
    "resolve_prompt_spec",
]

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class TaskRowOutput:
    """
    Base output schema for a task row. All experiment tasks share these fields.
    Subclass to add task-specific fields.
    """

    pair_id: int
    raw_response: str
    timestamp: Optional[str]
    err: Optional[str]
    prompt: str  # The prompt passed to the LLM for this row
    prompt_spec_id: Optional[str] = None


@dataclass(frozen=True)
class TaskExecutionResult:
    """
    Extra execution metadata that the runner uses for reporting.
    """

    output: TaskRowOutput
    cache_hit: bool
    parse_unknown: bool


class ExperimentTask(Protocol):
    NAME: str
    def run_row(self, row: object) -> TaskExecutionResult: ...



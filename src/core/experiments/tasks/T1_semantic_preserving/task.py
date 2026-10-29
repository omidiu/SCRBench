from __future__ import annotations

from dataclasses import dataclass

from core.experiments.llm_client.open_router import (
    commit_openrouter_cache,
    query_openrouter_chat_model_with_cache_info,
)
from core.experiments.prompt_spec import PromptSpec

from ..base import TaskExecutionResult, TaskRowOutput


@dataclass(frozen=True)
class T1SemanticPreservingTaskRowOutput(TaskRowOutput):
    """Output schema for T1 semantic preserving task."""

    semantic_is_equal: bool | None = None


@dataclass(frozen=True)
class T1SemanticPreservingTask:
    """
    T1: determine whether (source, destination) are semantically equivalent.
    Input row must provide at least: id, source, destination.
    """

    model_name: str
    prompt_spec: PromptSpec
    NAME: str = "T1_semantic_preserving"
    max_tokens: int = 4000

    def run_row(self, row: object) -> TaskExecutionResult:
        pair_id = int(getattr(row, "id"))
        prompt = self.prompt_spec.build_for_row(row)
        prompt_spec_id = self.prompt_spec.id

        raw_response_for_output = None
        timestamp = None
        cache_hit = False
        try:
            res = query_openrouter_chat_model_with_cache_info(
                prompt,
                self.model_name,
                max_tokens=self.max_tokens,
            )
            raw_response_for_output = res.response_content
            timestamp = res.server_datetime.isoformat() if res.server_datetime else None
            cache_hit = bool(res.cache_hit)
            parsed, parse_unknown = self.prompt_spec.parse_response(res.response_content)
            if parsed is None:
                output = T1SemanticPreservingTaskRowOutput(
                    pair_id=pair_id,
                    raw_response=res.response_content,
                    timestamp=timestamp,
                    err="Unparseable model response (expected exactly YES/NO).",
                    prompt_spec_id=prompt_spec_id,
                    prompt=prompt,
                    semantic_is_equal=None,
                )
            else:
                commit_openrouter_cache(res)
                output = T1SemanticPreservingTaskRowOutput(
                    pair_id=pair_id,
                    raw_response=res.response_content,
                    timestamp=timestamp,
                    err=None,
                    prompt_spec_id=prompt_spec_id,
                    prompt=prompt,
                    semantic_is_equal=bool(parsed),
                )
            return TaskExecutionResult(output=output, cache_hit=cache_hit, parse_unknown=bool(parse_unknown))
        except Exception as e:
            output = T1SemanticPreservingTaskRowOutput(
                pair_id=pair_id,
                raw_response=raw_response_for_output if raw_response_for_output is not None else f"Error: {e}",
                timestamp=timestamp,
                err=str(e),
                prompt_spec_id=prompt_spec_id,
                prompt=prompt,
                semantic_is_equal=None,
            )
            return TaskExecutionResult(output=output, cache_hit=cache_hit, parse_unknown=False)

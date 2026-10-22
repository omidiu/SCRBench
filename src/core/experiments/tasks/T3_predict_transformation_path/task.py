from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.experiments.llm_client.open_router import (
    commit_openrouter_cache,
    query_openrouter_chat_model_with_cache_info,
)
from core.experiments.prompt_spec import PromptSpec

from ..base import TaskExecutionResult, TaskRowOutput


@dataclass(frozen=True)
class T3PredictTransformationPathTaskRowOutput(TaskRowOutput):
    """Output schema for T3 predict transformation path task."""

    predicted_path: Optional[List[int]] = None


@dataclass(frozen=True)
class T3PredictTransformationPathTask:
    """
    T3: predict the sequence of transformations applied to transform source into destination.
    """

    model_name: str
    prompt_spec: PromptSpec
    NAME: str = "T3_predict_transformation_path"
    max_tokens: int = 4000

    def run_row(self, row: object) -> TaskExecutionResult:
        pair_id = int(getattr(row, "id"))
        prompt = self.prompt_spec.build_for_row(row)
        prompt_spec_id = self.prompt_spec.id

        raw_response_for_output: Optional[str] = None
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
                output = T3PredictTransformationPathTaskRowOutput(
                    pair_id=pair_id,
                    raw_response=raw_response_for_output,
                    timestamp=timestamp,
                    err="Unparseable model response (expected comma-separated transformation IDs).",
                    prompt_spec_id=prompt_spec_id,
                    prompt=prompt,
                    predicted_path=None,
                )
            else:
                commit_openrouter_cache(res)
                output = T3PredictTransformationPathTaskRowOutput(
                    pair_id=pair_id,
                    raw_response=raw_response_for_output,
                    timestamp=timestamp,
                    err=None,
                    prompt_spec_id=prompt_spec_id,
                    prompt=prompt,
                    predicted_path=parsed,
                )
            return TaskExecutionResult(output=output, cache_hit=cache_hit, parse_unknown=bool(parse_unknown))
        except Exception as e:
            output = T3PredictTransformationPathTaskRowOutput(
                pair_id=pair_id,
                raw_response=raw_response_for_output if raw_response_for_output else f"Error: {e}",
                timestamp=timestamp,
                err=str(e),
                prompt_spec_id=prompt_spec_id,
                prompt=prompt,
                predicted_path=None,
            )
            return TaskExecutionResult(output=output, cache_hit=cache_hit, parse_unknown=False)

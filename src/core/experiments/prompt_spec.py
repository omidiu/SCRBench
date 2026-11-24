from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

ParseResult = Tuple[Optional[Any], bool]


@dataclass(frozen=True)
class PromptSpec:
    """
    One experiment prompt variant: a prompt builder paired with its response parser.

    Parsers are defined alongside the prompt they belong to so parsing rules stay
    in sync when adding or changing variants.
    """

    id: str
    build_prompt: Callable[..., str]
    parse_response: Callable[[str], ParseResult]
    build_from_row: Optional[Callable[[object], str]] = None

    def build_for_row(self, row: object) -> str:
        if self.build_from_row is not None:
            return self.build_from_row(row)
        source = getattr(row, "source")
        destination = getattr(row, "destination")
        return self.build_prompt(source, destination)

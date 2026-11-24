import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ExperimentReport:
    run_id: str
    task_name: str
    model_name: str

    reference_file_path: str
    llm_judge_file_path: str
    report_file_path: str

    started_at: str
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    input_rows_processed: int = 0
    output_rows_written: int = 0

    cache_hits: int = 0
    llm_calls: int = 0
    error_rows: int = 0  # API or execution failures; excludes unparseable responses.

    response_parse_unknown: int = 0

    extra: Dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report_json(report: ExperimentReport) -> None:
    os.makedirs(os.path.dirname(report.report_file_path), exist_ok=True)
    with open(report.report_file_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)



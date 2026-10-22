import os
import re
import time
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple, Type

import pandas as pd
import json
import os
from pathlib import Path

import pandas as pd
from core import project_config

from .enums import LLMModel, ReferenceDataset, TaskType
from .prompt_spec import PromptSpec
from .tasks import resolve_prompt_spec
from .io import InputDatasetSpec, count_csv_rows, iter_csv_in_chunks, validate_columns
from .reporting import ExperimentReport, utc_now_iso, write_report_json
from .tasks import TASK_REGISTRY
from .tasks.base import ExperimentTask, TaskRowOutput
from core.utils import extract_ground_truth_altering_id, parse_paths, parse_predicted_path

def _sanitize_filename_component(s: str) -> str:
    s = s.strip()
    s = s.replace("/", "__").replace(":", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180] if len(s) > 180 else s


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: LLMModel
    prompt_spec: PromptSpec
    task_type: TaskType

    reference_file_name: ReferenceDataset
    cache_folder_path: str = project_config.CACHE_FOLDER_NAME
    experiment_folder_path: str = os.path.join(project_config.PROJECT_ROOT, project_config.EXPERIMENT_RESULTS_DIR) 

    # Execution
    chunksize: int = 250
    max_rows: Optional[int] = None
    show_progress: bool = True

    # OpenRouter
    max_tokens: int = 4000


def run_experiment(cfg: ExperimentConfig) -> Tuple[str, str]:
    """
    TODO: WRITE doc string again : ) 
    """
    task_type_str = cfg.task_type.value
    model_name_str = cfg.model_name.value
    reference_file_name_str = cfg.reference_file_name.value

    if task_type_str not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_type={cfg.task_type!r}. Known tasks: {sorted(TASK_REGISTRY.keys())}")

    prompt_spec = resolve_prompt_spec(cfg.task_type, cfg.prompt_spec)
    prompt_spec_id = prompt_spec.id
    task_cls: Type[ExperimentTask] = TASK_REGISTRY[task_type_str]
    task_kwargs = {
        "model_name": model_name_str,
        "max_tokens": cfg.max_tokens,
        "prompt_spec": prompt_spec,
    }

    task: ExperimentTask = task_cls(**task_kwargs)  # type: ignore[call-arg]

    reference_spec = InputDatasetSpec(
        cache_folder=cfg.cache_folder_path,
        filename=reference_file_name_str,
        required_columns=["id", "source", "destination", "paths", "is_semantically_equivalent", "k"],
    )
    reference_file_path = os.path.join(project_config.PROJECT_ROOT, project_config.CACHE_FOLDER_NAME, reference_spec.filename) 
    if not os.path.exists(reference_file_path):
        raise FileNotFoundError(f"Input dataset not found: {reference_file_path}")

    task_safe = _sanitize_filename_component(task_type_str)
    reference_safe = _sanitize_filename_component(reference_file_name_str)
    model_safe = _sanitize_filename_component(model_name_str)
    prompt_safe = _sanitize_filename_component(prompt_spec_id)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    folder_path = os.path.join(cfg.experiment_folder_path, task_safe, reference_safe, model_safe, prompt_safe)
    os.makedirs(folder_path, exist_ok=True)

    llm_judge_file_name = f"llm_judge_{run_id}.csv"
    report_file_name = f"report_{run_id}.json"

    llm_judge_file_path = os.path.join(folder_path, llm_judge_file_name)
    report_file_path = os.path.join(folder_path, report_file_name)

    report = ExperimentReport(
        run_id=run_id,
        task_name=task_type_str,
        model_name=model_name_str,
        reference_file_path=reference_file_path,
        llm_judge_file_path=llm_judge_file_path,
        report_file_path=report_file_path,
        started_at=utc_now_iso(),
        extra={
            "chunksize": cfg.chunksize,
            "max_rows": cfg.max_rows,
            "max_tokens": cfg.max_tokens,
            "prompt_spec_id": prompt_spec_id,
        },
    )

    header_written = False
    pbar = None
    total_rows_for_progress: Optional[int] = None

    try:
        if cfg.show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                total_rows_in_file = count_csv_rows(reference_file_path)
                total_rows_for_progress = total_rows_in_file
                if cfg.max_rows is not None: # TODO: REMOVE LATER
                    total_rows_for_progress = min(total_rows_for_progress, cfg.max_rows)

                report.extra["total_rows_in_file"] = total_rows_in_file
                report.extra["total_rows_for_run"] = total_rows_for_progress

                pbar = tqdm(
                    total=total_rows_for_progress,
                    desc=f"{task_type_str} [{model_name_str}]",
                    unit="row",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            except Exception:
                # Progress is best-effort; never break an experiment because of it.
                pbar = None

        for chunk_idx, chunk in enumerate(
            iter_csv_in_chunks(reference_file_path, chunksize=cfg.chunksize, max_rows=cfg.max_rows),
            start=1,
        ):
            if chunk_idx == 1:
                validate_columns(chunk, reference_spec.required_columns, context=reference_file_path)

            outputs: list[TaskRowOutput] = []

            for row in chunk.itertuples(index=False):
                report.input_rows_processed += 1
                result = task.run_row(row)

                if result.cache_hit:
                    report.cache_hits += 1
                else:
                    report.llm_calls += 1

                if result.parse_unknown:
                    report.response_parse_unknown += 1

                if result.output.err:
                    report.error_rows += 1

                outputs.append(result.output)

                if report.extra.get("output_columns") is None:
                    report.extra["output_columns"] = [f.name for f in fields(type(result.output))]

                if pbar is not None:
                    pbar.update(1)
                    if report.input_rows_processed % 25 == 0:
                        remaining = None
                        if total_rows_for_progress is not None:
                            remaining = max(0, total_rows_for_progress - report.input_rows_processed)
                        pbar.set_postfix(
                            {
                                "left": remaining,
                                "cache": report.cache_hits,
                                "llm": report.llm_calls,
                                "err": report.error_rows,
                            }
                        )

            out_df = pd.DataFrame([asdict(o) for o in outputs])
            out_df.to_csv(llm_judge_file_path, mode="a", index=False, header=not header_written)
            header_written = True
            report.output_rows_written += len(out_df)

    finally:

        AggregateWithReferenceFile.aggregate_with_reference_file(
            task_type=cfg.task_type,
            folder_path=folder_path,
            llm_judge_file_path=llm_judge_file_path,
            refernce_file_path=reference_file_path,
            run_id=run_id
        )

        if pbar is not None:
            pbar.close()

        finished = utc_now_iso()
        report.finished_at = finished
        try:
            from datetime import datetime

            started_dt = datetime.fromisoformat(report.started_at)
            finished_dt = datetime.fromisoformat(finished)
            report.duration_seconds = (finished_dt - started_dt).total_seconds()
        except Exception:
            report.duration_seconds = None

        write_report_json(report)

    return llm_judge_file_path, report_file_path

def _transformation_metadata() -> dict:
    from core.data_builder.transformations import ALL_TRANSFORMATIONS

    return {
        str(t.id): {
            "name": t.name,
            "short_description": t.short_description,
        }
        for t in ALL_TRANSFORMATIONS
    }


class AggregateWithReferenceFile:

    @classmethod
    def aggregate_with_reference_file(
        cls, 
        task_type: TaskType, 
        folder_path: str,
        llm_judge_file_path: str,
        refernce_file_path: str,
        run_id: str
    ):
    
        mapper = {
            TaskType.T1: cls.__create_json_file_for_t1,
            TaskType.T2: cls.__create_json_file_for_t2,
            TaskType.T3: cls.__create_json_file_for_t3,
        }

        try:
            handler = mapper[task_type]
        except KeyError as e:
            raise ValueError(f"Unsupported task type: {task_type}") from e

        return handler(folder_path, llm_judge_file_path, refernce_file_path, run_id)

    @staticmethod
    def __create_json_file_for_t1(
        folder_path: str, 
        llm_judge_file_path: str, 
        refernce_file_path: str,
        run_id: str
    ):
        llm_judge_file = pd.read_csv(llm_judge_file_path)
        reference_file = pd.read_csv(refernce_file_path)

        if len(llm_judge_file) != len(reference_file):
            raise ValueError(
                f"Row count mismatch: source has {len(reference_file)}, results has {len(llm_judge_file)}"
            )


        merged = pd.concat([reference_file, llm_judge_file], axis=1)
        merged = merged.rename(columns={
            "is_semantically_equivalent": "real_class",
            "semantic_is_equal": "predicted_class",
        })
        
        columns_to_export = ["source", "destination", "real_class", "predicted_class", "paths", 'raw_response']

        reference_llm_judge_file_name = f'reference_llm_judge_{run_id}.json'
        reference_llm_judge_path = os.path.join(folder_path, reference_llm_judge_file_name)

        merged[columns_to_export].to_json(
            reference_llm_judge_path,
            orient="records",
            force_ascii=False,
            indent=2
        )


    @staticmethod
    def __create_json_file_for_t2(
        folder_path: str, 
        llm_judge_file_path: str, 
        refernce_file_path: str,
        run_id: str
    ):

        llm_judge_file = pd.read_csv(llm_judge_file_path)
        reference_file = pd.read_csv(refernce_file_path)


        if len(reference_file) != len(llm_judge_file):
            raise ValueError(
                f"Row count mismatch: source has {len(reference_file)}, results has {len(llm_judge_file)}"
            )

        rows = []
        for i in range(len(reference_file)):
            reference_row = reference_file.iloc[i]
            llm_judge_row = llm_judge_file.iloc[i]

            pair_id = llm_judge_row.get("pair_id", llm_judge_row.get("id", i))
            paths_raw = reference_row["paths"]
            paths = parse_paths(paths_raw)
            k = len(paths[0]) if paths else 0

            # altering_transformation_id can be int, float (NaN), or missing
            raw_pred = llm_judge_row.get("altering_transformation_id")
            if pd.notna(raw_pred) and raw_pred != "":
                try:
                    predicted_id = int(float(raw_pred))
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid predicted altering_transformation_id: {raw_pred!r} at row {i}")
               
            else:
                raise ValueError(f"altering_transformation_id is missing or invalid at row {i}")
           
           

            ground_truth_altering_id = extract_ground_truth_altering_id(paths)
            correct = int(predicted_id) == int(ground_truth_altering_id)

            row = {
                "id": int(pair_id) if pd.notna(pair_id) else i,
                "source": str(reference_row["source"]) if pd.notna(reference_row["source"]) else "",
                "destination": str(reference_row["destination"]) if pd.notna(reference_row["destination"]) else "",
                "paths": paths,
                "ground_truth_altering_id": ground_truth_altering_id,
                "k": k,
                "predicted_id": predicted_id,
                "correct": correct,
                "raw_response": str(llm_judge_row.get("raw_response", "")) if pd.notna(llm_judge_row.get("raw_response")) else "",
            }
            
            row["prompt"] = str(llm_judge_row.get("prompt", "")) if pd.notna(llm_judge_row.get("prompt")) else ""
            row["timestamp"] = str(llm_judge_row.get("timestamp", "")) if pd.notna(llm_judge_row.get("timestamp")) else ""
            row["err"] = str(llm_judge_row.get("err", "")) if pd.notna(llm_judge_row.get("err")) else ""

            rows.append(row)

        output = {
            "metadata": {"transformations": _transformation_metadata()},
            "rows": rows,
        }

        reference_llm_judge_file_name = f'reference_llm_judge_{run_id}.json'
        reference_llm_judge_path = os.path.join(folder_path, reference_llm_judge_file_name)

        with open(reference_llm_judge_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Wrote {len(rows)} rows to {reference_llm_judge_path}")


    @staticmethod
    def __create_json_file_for_t3(
        folder_path: str, 
        llm_judge_file_path: str, 
        refernce_file_path: str,
        run_id: str
    ):

        llm_judge_file = pd.read_csv(llm_judge_file_path)
        reference_file = pd.read_csv(refernce_file_path)


        if len(llm_judge_file) != len(reference_file):
            raise ValueError(
                f"Row count mismatch: source has {len(reference_file)}, results has {len(llm_judge_file)}"
            )

        rows = []
        for i in range(len(reference_file)):
            reference_row = reference_file.iloc[i]
            llm_judge_row = llm_judge_file.iloc[i]

            pair_id = llm_judge_row.get("pair_id", reference_row.get("id", i))
            paths_raw = reference_row["paths"]
            paths = parse_paths(paths_raw)
            k = len(paths[0]) if paths else 0

            predicted_path = parse_predicted_path(llm_judge_row.get("predicted_path"))

            row = {
                "id": int(pair_id) if pd.notna(pair_id) else i,
                "source": str(reference_row["source"]) if pd.notna(reference_row["source"]) else "",
                "destination": str(reference_row["destination"]) if pd.notna(reference_row["destination"]) else "",
                "paths": paths,
                "k": k,
                "predicted_path": predicted_path,
                "raw_response": str(llm_judge_row.get("raw_response", "")) if pd.notna(llm_judge_row.get("raw_response")) else "",
            }
            
            row["prompt"] = str(llm_judge_row.get("prompt", "")) if pd.notna(llm_judge_row.get("prompt")) else ""
            row["timestamp"] = str(llm_judge_row.get("timestamp", "")) if pd.notna(llm_judge_row.get("timestamp")) else ""
            row["err"] = str(llm_judge_row.get("err", "")) if pd.notna(llm_judge_row.get("err")) else ""

            rows.append(row)

        output = {
            "metadata": {"transformations": _transformation_metadata()},
            "rows": rows,
        }

        reference_llm_judge_file_name = f'reference_llm_judge_{run_id}.json'
        reference_llm_judge_path = os.path.join(folder_path, reference_llm_judge_file_name)

        with open(reference_llm_judge_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Wrote {len(rows)} rows to {reference_llm_judge_path}")










import importlib
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.experiments import ExperimentConfig, LLMModel, ReferenceDataset, TaskType
from core.experiments.runner import AggregateWithReferenceFile, run_experiment
from core.experiments.tasks import TASK_REGISTRY, get_prompt_spec
from core.utils import parse_predicted_path
from scripts.compute_t1_metrics import compute_t1_metrics
from scripts.compute_t2_metrics import compute_t2_metrics


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.csv"
        self.results = self.root / "results.csv"
        self.records = [
            dict(id=101, source="source101", destination="dest101", paths="[[1]]",
                 k=1, is_semantically_equivalent=False),
            dict(id=202, source="source202", destination="dest202", paths="[[5]]",
                 k=1, is_semantically_equivalent=True),
            dict(id=303, source="source303", destination="dest303", paths="[[2, 5], [5, 2]]",
                 k=2, is_semantically_equivalent=False),
        ]
        pd.DataFrame(self.records).to_csv(self.reference, index=False)

    def export(self, task, rows):
        pd.DataFrame(rows).to_csv(self.results, index=False)
        AggregateWithReferenceFile.aggregate_with_reference_file(
            task, str(self.root), str(self.results), str(self.reference), "test"
        )
        return json.loads((self.root / "reference_llm_judge_test.json").read_text())

    def test_t1_failure_is_incorrect_even_for_non_equivalent_pair(self):
        rows = self.export(TaskType.T1, [
            dict(pair_id=303, semantic_is_equal=False, err="parse failed", raw_response="maybe"),
            dict(pair_id=202, semantic_is_equal=True),
            dict(pair_id=101, semantic_is_equal=False),
        ])
        self.assertEqual([row["source"] for row in rows], ["source303", "source202", "source101"])
        self.assertIsNone(rows[0]["predicted_class"])
        self.assertFalse(rows[0]["correct"])
        self.assertEqual(rows[0]["err"], "parse failed")
        self.assertEqual(rows[0]["raw_response"], "maybe")
        self.assertIs(rows[2]["predicted_class"], False)
        metrics = compute_t1_metrics(rows)
        self.assertEqual(metrics["overall"]["correct"], 2)
        self.assertEqual(metrics["unparseable"], 1)

    def test_t2_missing_invalid_and_failed_predictions_do_not_abort_export(self):
        for prediction, error in [(None, ""), ("invalid", ""), (1.5, ""), (99, ""), (2, "API failed")]:
            with self.subTest(prediction=prediction, error=error):
                payload = self.export(TaskType.T2, [
                    dict(pair_id=303, altering_transformation_id=prediction, err=error),
                    dict(pair_id=101, altering_transformation_id=1),
                ])
                self.assertEqual(payload["rows"][0]["source"], "source303")
                self.assertIsNone(payload["rows"][0]["predicted_id"])
                self.assertFalse(payload["rows"][0]["correct"])
                metrics = compute_t2_metrics(payload)
                self.assertEqual(metrics["overall"]["correct"], 1)
                self.assertEqual(metrics["unparseable"], 1)
                self.assertTrue(payload["metadata"]["is_partial"])

    def test_t3_any_ground_truth_order_counts_as_correct(self):
        for prediction, correct in [("[5, 2]", True), ("[2, 5]", True), ("[5]", False),
                                    (None, False), ("bad", False), ("[]", False)]:
            with self.subTest(prediction=prediction):
                payload = self.export(TaskType.T3, [dict(pair_id=303, predicted_path=prediction)])
                row = payload["rows"][0]
                self.assertEqual(row["source"], "source303")
                self.assertEqual(row["correct"], correct)
                self.assertEqual(payload["metadata"]["completed_count"], 1)
                self.assertEqual(payload["metadata"]["reference_count"], 3)
        payload = self.export(TaskType.T3, [dict(pair_id=303, predicted_path="[5, 2]", err="failure")])
        self.assertIsNone(payload["rows"][0]["predicted_path"])
        self.assertFalse(payload["rows"][0]["correct"])

    def test_partial_t1_does_not_invent_unattempted_predictions(self):
        rows = self.export(TaskType.T1, [dict(pair_id=202, semantic_is_equal=True)])
        self.assertEqual([row["id"] for row in rows], [202])

    def test_raw_response_text_is_not_treated_as_csv_missing_data(self):
        for raw in ("NA", "null", "NaN", ""):
            with self.subTest(raw=raw):
                rows = self.export(TaskType.T1, [
                    dict(pair_id=101, semantic_is_equal=None, raw_response=raw)
                ])
                self.assertEqual(rows[0]["raw_response"], raw)

    def test_bad_result_ids_are_rejected(self):
        for ids in ([101, 101], [999], [None], [1.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.export(TaskType.T1, [dict(pair_id=i, semantic_is_equal=False) for i in ids])

    def test_duplicate_reference_ids_are_rejected(self):
        pd.DataFrame([self.records[0], self.records[0]]).to_csv(self.reference, index=False)
        with self.assertRaises(ValueError):
            self.export(TaskType.T1, [dict(pair_id=101, semantic_is_equal=False)])

    def test_empty_and_not_yet_written_runs_export_empty_rows(self):
        for task in TaskType:
            for existing in (False, True):
                with self.subTest(task=task, existing=existing):
                    result_path = self.root / ("empty.csv" if existing else "absent.csv")
                    if existing:
                        result_path.write_text("")
                    AggregateWithReferenceFile.aggregate_with_reference_file(
                        task, str(self.root), str(result_path), str(self.reference), "empty"
                    )
                    payload = json.loads((self.root / "reference_llm_judge_empty.json").read_text())
                    self.assertEqual(payload if task == TaskType.T1 else payload["rows"], [])

    def test_runner_max_rows_exports_partial_results_and_writes_report(self):
        cache = self.root / "cache"
        cache.mkdir()
        dataset = ReferenceDataset.AGGREGATE_MIXED_LABELS
        pd.DataFrame(self.records).to_csv(cache / dataset.value, index=False)
        module = importlib.import_module(TASK_REGISTRY[TaskType.T1.value].__module__)
        response = SimpleNamespace(response_content="NO", server_datetime=None, cache_hit=False)
        cfg = ExperimentConfig(
            model_name=LLMModel.GPT_4,
            prompt_spec=get_prompt_spec(TaskType.T1, "base"),
            task_type=TaskType.T1,
            reference_file_name=dataset,
            experiment_folder_path=str(self.root / "runs"),
            max_rows=1,
            show_progress=False,
        )
        with patch("core.experiments.runner.project_config.PROJECT_ROOT", self.root), \
             patch.object(module, "query_openrouter_chat_model_with_cache_info", return_value=response), \
             patch.object(module, "commit_openrouter_cache"):
            csv_path, report_path = run_experiment(cfg)
        self.assertEqual(json.loads(Path(report_path).read_text())["output_rows_written"], 1)
        exported = list(Path(csv_path).parent.glob("reference_llm_judge_*.json"))
        self.assertEqual([r["id"] for r in json.loads(exported[0].read_text())], [101])

        failing_cfg = replace(cfg, experiment_folder_path=str(self.root / "failed_export"))
        with patch("core.experiments.runner.project_config.PROJECT_ROOT", self.root), \
             patch.object(module, "query_openrouter_chat_model_with_cache_info", return_value=response), \
             patch.object(module, "commit_openrouter_cache"), \
             patch.object(AggregateWithReferenceFile, "aggregate_with_reference_file", side_effect=ValueError("bad IDs")):
            with self.assertRaisesRegex(ValueError, "bad IDs"):
                run_experiment(failing_cfg)
        reports = list((self.root / "failed_export").rglob("report_*.json"))
        self.assertEqual(len(reports), 1)
        self.assertIsNotNone(json.loads(reports[0].read_text())["finished_at"])


class TaskFailureTests(unittest.TestCase):
    def test_tasks_preserve_missing_predictions_and_raw_responses(self):
        fields = {TaskType.T1: "semantic_is_equal", TaskType.T2: "altering_transformation_id",
                  TaskType.T3: "predicted_path"}
        row = SimpleNamespace(id=101, source="x = 1", destination="x = 2")
        for task_type, field in fields.items():
            task_class = TASK_REGISTRY[task_type.value]
            module = importlib.import_module(task_class.__module__)
            task = task_class(model_name="test", prompt_spec=get_prompt_spec(task_type, "base"))
            for raw in ("unreadable", ""):
                with self.subTest(task=task_type, raw=raw):
                    response = SimpleNamespace(response_content=raw, server_datetime=None, cache_hit=False)
                    with patch.object(module, "query_openrouter_chat_model_with_cache_info", return_value=response):
                        output = task.run_row(row).output
                    self.assertIsNone(getattr(output, field))
                    self.assertEqual(output.raw_response, raw)
                    self.assertTrue(output.err)
            with self.subTest(task=task_type, api_failure=True):
                with patch.object(module, "query_openrouter_chat_model_with_cache_info", side_effect=RuntimeError("offline")):
                    output = task.run_row(row).output
                self.assertIsNone(getattr(output, field))
                self.assertIn("offline", output.err)

    def test_saved_path_decoder_does_not_truncate_or_drop_invalid_ids(self):
        for raw in (None, float("nan"), "", "bad", [], "[1.5]", "[99]", "[true]", "[1, \\\"bad\\\"]"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_predicted_path(raw))
        self.assertEqual(parse_predicted_path([5, 2]), [5, 2])
        self.assertEqual(parse_predicted_path("[5, 2]"), [5, 2])


if __name__ == "__main__":
    unittest.main()

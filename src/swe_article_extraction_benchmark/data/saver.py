import copy
import csv
import importlib.metadata
import typing as t
from pathlib import Path
from uuid import uuid4

import json_arrays
from json_arrays import jsonl_iter

from swe_article_extraction_benchmark.data.dataset import Dataset
from swe_article_extraction_benchmark.evaluator import EvaluationStats
from swe_article_extraction_benchmark.metrics import BaseMetric


class DataSaver:
    """"""

    @staticmethod
    def save_extraction_result(
        results: t.Union[EvaluationStats, list[EvaluationStats]],
        dataset: Dataset,
        file_path: Path,
        extractor_name: str | None = None,
    ) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        all_extraction_maps = {}
        extractor_names = []
        if isinstance(results, list):
            for result_item in results:
                _process_evaluation_stats(
                    result_item,  # ty: ignore[invalid-argument-type]
                    all_extraction_maps=all_extraction_maps,
                    extractor_names=extractor_names,
                    extractor_name=extractor_name,
                )
        else:
            _process_evaluation_stats(
                results,
                all_extraction_maps=all_extraction_maps,
                extractor_names=extractor_names,
                extractor_name=extractor_name,
            )

        enriched_samples = []

        for sample in dataset.samples:
            sample_dict = sample.to_dict()

            for current_extractor_name in extractor_names:
                extraction_map = all_extraction_maps.get(current_extractor_name, {})
                extraction_result = extraction_map.get(sample.id)

                if extraction_result:
                    sample_dict[f"{current_extractor_name}_content"] = (
                        extraction_result.get("extracted_content", "")
                    )
                    sample_dict[f"{current_extractor_name}_content_list"] = (
                        extraction_result.get("extracted_content", "")
                    )
                    sample_dict[f"{current_extractor_name}_success"] = (
                        extraction_result.get("extracted_content", "")
                    )
                    sample_dict[f"{current_extractor_name}_time_s"] = (
                        extraction_result.get("extraction_time_s", "")
                    )

                    metrics = extraction_result.get("metrics", {})
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and metric_data.get(
                            "success", False
                        ):
                            sample_dict[
                                f"{current_extractor_name}_{metric_name}_score"
                            ] = metric_data.get("score", 0)

                    predicted_content = extraction_result.get("extracted_content", "")
                    predicted_parts = BaseMetric.extract_parts_from_markdown(
                        predicted_content
                    )
                    for part_type in ["code", "formula", "table", "text"]:
                        sample_dict[
                            f"{current_extractor_name}_predicted_{part_type}"
                        ] = predicted_parts[part_type]

            if extractor_names:
                groundtruth_content = sample_dict.get("groundtruth_content", "")
                groundtruth_parts = BaseMetric.extract_parts_from_markdown(
                    groundtruth_content
                )
                for part_type in ["code", "formula", "table", "text"]:
                    prefix = (
                        extractor_names[0]
                        if len(extractor_names) == 1
                        else "groundtruth"
                    )
                    sample_dict[f"{prefix}_groundtruth_{part_type}"] = (
                        groundtruth_parts.get(part_type, "")
                    )

            enriched_samples.append(sample_dict)

        # Save as JSONL
        DataSaver._save_jsonl_list(enriched_samples, file_path)

    @staticmethod
    def save_evaluation_results(
        results: t.Union[EvaluationStats, dict[str, EvaluationStats]],
        file_path: Path,
        format: str = "json",
    ) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(results, EvaluationStats):
            results_dict = results.to_dict()
        else:
            results_dict = {k: v.to_dict() for k, v in results.items()}

        results_dict = DataSaver._remove_content_fields(results_dict)
        json_arrays.dump_to_file(results_dict, file_path, json_format=format)

    @staticmethod
    def save_summary_report(
        results: t.Union[EvaluationStats, list[EvaluationStats]], file_path: Path
    ) -> None:
        def to_dict_if_needed(item):
            return item.to_dict() if hasattr(item, "to_dict") else item

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(results, list):
            results_list = [to_dict_if_needed(item) for item in results]
        else:
            results_list = [to_dict_if_needed(results)]

        csv_data = []
        for result in results_list:
            metadata = result.get("metadata", {})
            error_analysis = result.get("error_analysis", {})

            extractor_name = metadata.get("extractor_name", "unknown")
            try:
                package_mapping = {}
                package_name = package_mapping.get(extractor_name) or extractor_name
                extractor_version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                extractor_version = "unknown"
            row = {
                "extractor": extractor_name,
                "dataset": metadata.get("dataset_name", "unknown"),
                "total_samples": metadata.get("total_samples", 0),
                "success_rate": error_analysis.get("success_rate", 0.0),
                "extractor_version": extractor_version,
            }

            if "overall_metrics" in result:
                for metric_name, value in result["overall_metrics"].items():
                    row[metric_name] = (
                        round(value, 4) if isinstance(value, float) else value
                    )

            csv_data.append(row)

            # Sort by overall score (descending)
            def get_sort_key(row):
                return row.get("overall", 0)

            csv_data.sort(key=get_sort_key, reverse=True)

            if csv_data:
                basic_fields = [
                    "extractor",
                    "extractor_version",
                    "dataset",
                    "total_samples",
                    "success_rate",
                ]

                all_fields = set()
                for row in csv_data:
                    all_fields.update(row.keys())

                metric_fields = all_fields - set(basic_fields)

                sorted_metrics = []
                if "overall" in metric_fields:
                    sorted_metrics.append("overall")
                    metric_fields.remove("overall")
                sorted_metrics.extend(sorted(metric_fields))

                fieldnames = basic_fields + sorted_metrics

                with file_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)

    @staticmethod
    def _save_jsonl_list(data_list: list[dict[str, t.Any]], file_path: Path) -> None:
        jsonl_iter.dump_to_file(data_list, file_path)

    @staticmethod
    def _remove_content_fields(data: dict[str, t.Any]) -> dict[str, t.Any]:
        def remove_fields(obj):
            if isinstance(obj, dict):
                obj.pop("extracted_content", None)
                obj.pop("extracted_content_list", None)
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        remove_fields(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        remove_fields(item)

        cleaned_data = copy.deepcopy(data)
        remove_fields(cleaned_data)

        return cleaned_data


def _process_evaluation_stats(
    result_item: EvaluationStats,
    *,
    all_extraction_maps: dict[str, t.Any],
    extractor_names: list[str],
    extractor_name: str | None,
) -> None:
    if hasattr(result_item, "to_dict"):
        results_dict = result_item.to_dict()
    else:
        results_dict = result_item

    current_extractor_name = results_dict.get("metadata", {}).get("extractor")
    if current_extractor_name is None:
        current_extractor_name = f"{extractor_name}-{uuid4().hex}"
    extractor_names.append(current_extractor_name)

    sample_results = results_dict.get("sample_results", [])
    extraction_map = {}
    for sample_result in sample_results:
        if sample_id := sample_result.get("sample_id"):
            extraction_map[sample_id] = sample_result

    all_extraction_maps[current_extractor_name] = extraction_map

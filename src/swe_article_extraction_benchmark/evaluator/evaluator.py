import typing as t
from dataclasses import dataclass
from datetime import datetime

from swe_article_extraction_benchmark.data.dataset import DataSample, Dataset
from swe_article_extraction_benchmark.extractors import BaseExtractor
from swe_article_extraction_benchmark.metrics import MetricCalculator


class SampleResult(t.TypedDict):
    sample_id: str
    extraction_success: bool
    extraction_time_s: float
    extracted_content: str | None
    # extracted_content_list: list[dict[str, t.Any]] | None
    metrics: dict[str, t.Any]
    extraction_error: t.NotRequired[str | None]
    sample_metadata: t.NotRequired[dict[str, t.Any]]


@dataclass
class EvaluationStats:
    """Statistics of benchmark evaluation."""

    # Metadata
    dataset_name: str
    extractor_name: str
    timestamp: str
    total_samples: int

    # Overall metrics
    overall_metrics: dict[str, float]

    # Sample level results
    sample_results: list[SampleResult]

    # Category-wise metrics (if applicable)
    category_metrics: dict[str, dict[str, float]] | None = None

    # Error analysis
    error_analysis: dict[str, t.Any] | None = None

    # Configuration
    extractor_config: dict[str, t.Any] | None = None
    metric_config: dict[str, t.Any] | None = None

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary format."""
        return {
            "metadata": {
                "dataset_name": self.dataset_name,
                "extractor_name": self.extractor_name,
                "timestamp": self.timestamp,
                "total_samples": self.total_samples,
            },
            "overall_metrics": self.overall_metrics,
            "sample_results": self.sample_results,
            "category_metrics": self.category_metrics,
            "error_analysis": self.error_analysis,
            "extractor_config": self.extractor_config,
            "metric_config": self.metric_config,
        }


class Evaluator:
    """Evaluator."""

    def __init__(self, metric_config: dict[str, t.Any] | None = None) -> None:
        self.metric_calculator = MetricCalculator(metric_config)
        self.metric_config = metric_config or {}

    def evaluate(
        self,
        dataset: Dataset,
        extractor: BaseExtractor,
        categories: list[str] | None = None,
    ) -> EvaluationStats:
        """Evaluate."""

        samples_iter = dataset.samples

        if categories:
            samples_iter = [s for s in samples_iter if s.content_type in categories]

        samples_to_evaluate = samples_iter
        sample_results = []
        extraction_errors = []

        print(f"Evaluating {len(samples_to_evaluate)} samples ...")

        for i, sample in enumerate(samples_to_evaluate):
            try:
                sample_result = self._evaluate_sample(sample, extractor)
                sample_results.append(sample_result)

                # Track extracion errors
                if not sample_result.get("extraction_success", True):
                    extraction_errors.append(
                        {
                            "sample_id": sample.id,
                            "error": sample_result.get(
                                "extraction_error", "Unknown error"
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error evaluation sample {sample.id}: {e}")

                error_result = {
                    "sample_id": sample.id,
                    "extraction_success": False,
                    "extracion_error": str(e),
                    "metrics": {},
                }
                sample_results.append(error_result)
                extraction_errors.append({"sample_id": sample.id, "error": str(e)})

        # Aggregate results
        overall_metrics = self._aggregate_metrics(sample_results)
        category_metrics = self._calculate_category_metrics(
            sample_results, samples_to_evaluate
        )
        error_analysis = self._analyze_errors(extraction_errors, sample_results)

        # Create evaluation stats
        evaluation_stats = EvaluationStats(
            dataset_name=dataset.name,
            extractor_name=extractor.name,
            timestamp=datetime.now().isoformat(),
            total_samples=len(samples_to_evaluate),
            overall_metrics=overall_metrics,
            sample_results=sample_results,
            category_metrics=category_metrics,
            error_analysis=error_analysis,
            extractor_config=extractor.get_config(),
            metric_config=self.metric_config,
        )
        return evaluation_stats

    def _evaluate_sample(
        self, sample: DataSample, extractor: BaseExtractor
    ) -> SampleResult:
        extraction_result = extractor.extract(sample.html, sample.url)

        sample_result: SampleResult = {
            "sample_id": sample.id,
            "extraction_success": extraction_result.success,
            "extraction_time_s": extraction_result.extraction_time_s,
            "extracted_content": extraction_result.content
            if extraction_result.success
            else None,
            # "extracted_content_list": extraction_result.content_list
            # if extraction_result.success
            # else None,
            "metrics": {},
        }

        if not extraction_result.success:
            sample_result["extraction_error"] = extraction_result.error_message
            return sample_result

        metrics = self.metric_calculator.calculate_all(
            predicted_content=extraction_result.content,
            groundtruth_content=sample.groundtruth_content,
            # predicted_content_list=extraction_result.content_list,
            # groundtruct_content_list=sample.groundtruth_content_list,
        )

        metrics_dict = {}
        for metric_name, metric_result in metrics.items():
            metrics_dict[metric_name] = {
                "score": metric_result.score,
                "success": metric_result.success,
                "details": metric_result.details,
            }
            if not metric_result.success:
                metrics_dict[metric_name]["error"] = metric_result.error_message

        sample_result["metrics"] = metrics_dict

        sample_result["sample_metadata"] = {
            "url": sample.url,
            "domain": sample.domain,
            "language": sample.language,
            "content_type": sample.content_type,
            "difficulty": sample.difficulty,
        }

        return sample_result

    def _aggregate_metrics(
        self, sample_results: list[dict[str, t.Any]]
    ) -> dict[str, float]:
        """Aggregate metrics across all samples."""

        if not sample_results:
            return {}

        # 初始化每个指标的总分和样本数
        metric_totals = {
            "text_edit": 0.0,
            "code_edit": 0.0,
            "table_edit": 0.0,
            "table_TEDS": 0.0,
            "formula_edit": 0.0,
            "overall": 0.0,  # 全局overall单独计算
        }
        metric_counts = {k: 0 for k in metric_totals.keys()}  # 记录每个指标有效样本数

        # 累加所有样本的指标分数
        for sample in sample_results:
            metrics = sample.get("metrics", {})
            for metric_name in metric_totals.keys():
                if metric_name in metrics and metrics[metric_name].get(
                    "success", False
                ):
                    metric_totals[metric_name] += metrics[metric_name]["score"]
                    metric_counts[metric_name] += 1

        # 计算每个指标的平均值（全局overall为5个单项指标的平均值）
        overall_metrics = {}
        for metric_name in metric_totals.keys():
            if metric_counts[metric_name] > 0:
                overall_metrics[metric_name] = (
                    metric_totals[metric_name] / metric_counts[metric_name]
                )
            else:
                overall_metrics[metric_name] = 0.0  # 无有效样本时默认为0

        # 特别处理全局overall：固定为5个单项指标的平均值（无论单项是否有有效样本）
        # 排除样本级overall，仅用5个核心指标计算全局overall
        core_metrics = [
            "text_edit",
            "code_edit",
            "table_edit",
            "table_TEDS",
            "formula_edit",
        ]
        core_scores = [overall_metrics[metric] for metric in core_metrics]
        overall_metrics["overall"] = sum(core_scores) / len(core_metrics)

        return overall_metrics

    def _calculate_category_metrics(
        self, sample_results: list[dict[str, t.Any]], samples: list[DataSample]
    ) -> dict[str, dict[str, float]] | None:
        """Calculate metrics by category."""
        # Group samples by content type
        category_samples = {}
        for i, sample in enumerate(samples):
            if i >= len(sample_results):
                break

            content_type = sample.content_type or "unknown"
            if content_type not in category_samples:
                category_samples[content_type] = []
            category_samples[content_type].append(sample_results[i])

        # Calculate metrics for each category
        category_metrics = {}
        for category, category_sample_results in category_samples.items():
            if len(category_sample_results) >= 3:  # Only calculate if enough samples
                category_metrics[category] = self._aggregate_metrics(
                    category_sample_results
                )

        return category_metrics if category_metrics else None

    def _analyze_errors(
        self,
        extraction_errors: list[dict[str, str]],
        sample_results: list[dict[str, t.Any]],
    ) -> dict[str, t.Any]:
        """Analyze extraction errors."""
        total_samples = len(sample_results)
        failed_samples = len(extraction_errors)
        success_rate = (
            (total_samples - failed_samples) / total_samples
            if total_samples > 0
            else 0.0
        )

        # Count error types
        error_types = {}
        for error in extraction_errors:
            error_msg = error["error"]
            # Simple error categorization
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                error_type = "network"
            elif "parse" in error_msg.lower() or "parsing" in error_msg.lower():
                error_type = "parsing"
            elif "empty" in error_msg.lower():
                error_type = "empty_input"
            else:
                error_type = "other"

            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_samples": total_samples,
            "failed_count": failed_samples,
            "success_rate": success_rate,
            "common_errors": error_types,
            "sample_errors": extraction_errors[:10],  # Keep first 10 for debugging
        }

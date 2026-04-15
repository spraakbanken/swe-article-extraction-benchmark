import typing as t

from swe_article_extraction_benchmark.metrics import BaseMetric, MetricResult
from swe_article_extraction_benchmark.metrics.text_metrics import (
    BleuMetric,
    RougeMetric,
)


class MetricCalculator:
    """"""

    def __init__(self, config: dict[str, t.Any] | None = None) -> None:
        self.config = config or {}
        self.metrics: dict[str, BaseMetric] = {}
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        self.add_metric(BleuMetric.create(self.config))
        self.add_metric(RougeMetric.create(self.config))

    def add_metric(self, metric: BaseMetric) -> None:
        self.metrics[metric.name] = metric

    def calculate_all(
        self,
        predicted_content: str,
        groundtruth_content: str,
        predicted_content_list: list[dict[str, t.Any]] | None = None,
        groundtruth_content_list: list[dict[str, t.Any]] | None = None,
        **kwargs,
    ) -> dict[str, MetricResult]:
        """
        Calculate all available metrics.

        Args:
            predicted_content: Predicted markdown content
            groundtruth_content: Ground truth markdown content
            predicted_content_list: Predicted content list
            groundtruth_content_list: Ground truth content list
            **kwargs: Additional arguments for specific metrics

        Returns:
            Dictionary mapping metric names to MetricResult instances
        """

        results: dict[str, MetricResult] = {}

        # 1. 先计算非表格指标（无依赖关系）
        for metric_name in list(self.metrics.keys()):
            if metric_name in ["table_edit", "table_TEDS"]:
                continue  # 表格相关指标单独处理

            metric = self.metrics[metric_name]
            result = metric.calculate(
                predicted=predicted_content,
                groundtruth=groundtruth_content,
                predicted_content_list=predicted_content_list,
                groundtruth_content_list=groundtruth_content_list,
                **kwargs,
            )
            results[metric_name] = result

        # 2. 处理表格相关指标（有依赖关系）
        # 2.1 计算 table_edit
        if "table_edit" in self.metrics:
            table_edit_result = self.metrics["table_edit"].calculate(
                predicted=predicted_content,
                groundtruth=groundtruth_content,
                predicted_content_list=predicted_content_list,
                groundtruth_content_list=groundtruth_content_list,
                **kwargs,
            )
            results["table_edit"] = table_edit_result

            # 2.2 计算 table_TEDS（依赖 table_edit 的结果）
            if "table_TEDS" in self.metrics:
                teds_result = self.metrics["table_TEDS"].calculate(
                    predicted=predicted_content,
                    groundtruth=groundtruth_content,
                    predicted_content_list=predicted_content_list,
                    groundtruth_content_list=groundtruth_content_list,
                    table_edit_result=table_edit_result,  # 传递依赖结果
                    **kwargs,
                )
                results["table_TEDS"] = teds_result

        # 3. 计算综合得分（所有成功指标的平均值）
        successful_scores = []
        failed_metrics = []

        for metric_name, result in results.items():
            if result.success:
                successful_scores.append(result.score)
            else:
                failed_metrics.append(metric_name)

        if successful_scores:
            overall_score = sum(successful_scores) / len(successful_scores)
            overall_result = MetricResult(
                metric_name="overall",
                score=overall_score,
                details={
                    "source": "average_of_all_metrics",
                    "description": "Overall score as average of all successful metrics",
                    "successful_metrics": len(successful_scores),
                    "failed_metrics": len(failed_metrics),
                    "individual_scores": {
                        name: result.score
                        for name, result in results.items()
                        if result.success
                    },
                },
            )
            results["overall"] = overall_result
        else:
            # 如果所有指标都失败了，overall分数为0
            overall_result = MetricResult.create_error_result(
                "overall", "All individual metrics failed"
            )
            results["overall"] = overall_result

        return results

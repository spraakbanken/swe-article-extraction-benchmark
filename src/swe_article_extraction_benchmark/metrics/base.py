import abc
import typing as t
from dataclasses import dataclass

from swe_article_extraction_benchmark.metrics.code_splitter import CodeSplitter
from swe_article_extraction_benchmark.metrics.formula_splitter import FormulaSplitter
from swe_article_extraction_benchmark.metrics.table_splitter import TableSplitter


@dataclass
class MetricResult:
    """Result of metric calculation."""

    metric_name: str
    score: float
    details: dict[str, t.Any] | None = None
    success: bool = True
    error_message: str | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "MetricResult":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def create_error_result(
        cls, metric_name: str, error_message: str
    ) -> "MetricResult":
        """Create an error result."""
        return cls(
            metric_name=metric_name,
            score=0.0,
            success=False,
            error_message=error_message,
        )


class BaseMetric(abc.ABC):
    def __init__(self, name: str, config: dict[str, t.Any] | None = None) -> None:
        self.name: str = name
        self.config = config or {}
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        """Setup the metric (load models, initialize components, etc)"""

    @abc.abstractmethod
    def _calculate_score(
        self, predicted: t.Any, groundtruth: t.Any, **kwargs
    ) -> MetricResult:
        """
        Calculate the metric score.

        Args:
            predicted: Predicted/extracted content
            groundtruth: Ground truth content
            **kwargs: Additional arguments

        Returns:
            MetricResult instance
        """
        pass

    def calculate(self, predicted: t.Any, groundtruth: t.Any, **kwargs) -> MetricResult:
        """
        Calculate metric with error handling.

        Args:
            predicted: Predicted/extracted content
            groundtruth: Ground truth content
            **kwargs: Additional arguments

        Returns:
            MetricResult instance
        """
        try:
            return self._calculate_score(predicted, groundtruth, **kwargs)
        except Exception as e:
            error_message = f"Metric calculation failed: {str(e)}"
            return MetricResult.create_error_result(self.name, error_message)

    def aggregate_results(self, results: list[MetricResult]) -> MetricResult:
        """
        Aggregate multiple metric results.

        Args:
            results: List of MetricResult instances

        Returns:
            Aggregated MetricResult
        """
        if not results:
            return MetricResult.create_error_result(
                self.name, "No results to aggregate"
            )

        # Filter successful results
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return MetricResult.create_error_result(
                self.name, "All calculations failed"
            )

        # Calculate aggregate score (mean by default)
        scores = [r.score for r in successful_results]
        avg_score = sum(scores) / len(scores)

        # Aggregate details
        aggregate_details = {
            "num_samples": len(results),
            "num_successful": len(successful_results),
            "num_failed": len(results) - len(successful_results),
            "scores": scores,
            "min_score": min(scores),
            "max_score": max(scores),
            "std_score": self._calculate_std(scores),
        }

        return MetricResult(
            metric_name=f"{self.name}_aggregate",
            score=avg_score,
            details=aggregate_details,
            success=True,
        )

    def _calculate_std(self, scores: list[float]) -> float:
        """Calculate standard deviation."""
        if len(scores) <= 1:
            return 0.0

        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance**0.5

    def get_config(self) -> dict[str, t.Any]:
        """Get metric configuration."""
        return self.config.copy()

    @staticmethod
    def extract_parts_from_markdown(text: str) -> dict[str, str]:
        if not text:
            return {"code": "", "formula": "", "table": "", "text": ""}

        code_extractor = CodeSplitter()
        formula_extractor = FormulaSplitter()
        table_extractor = TableSplitter()

        code_content = code_extractor.extract(text)
        formula_content = formula_extractor.extract(text)
        table_content = table_extractor.extract(text)

        return {
            "code": code_content,
            "formula": formula_content,
            "table": table_content,
            "text": text,
        }

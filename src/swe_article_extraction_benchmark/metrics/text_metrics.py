import typing as t

from nltk.translate import bleu_score
from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer

from swe_article_extraction_benchmark.metrics import BaseMetric, MetricResult


class EditDistanceMetric(BaseMetric):
    """Edit distance (Levenshtein distance) metric."""

    version = "1.0"
    description = "Character-level edit distance metric"

    def _setup(self) -> None:
        self.normalize = self.config.get("normalize", True)

    def _calculate_score(
        self, predicted: str, groundtruth: str, **kwargs
    ) -> MetricResult:
        """
        Calculate edit distance between predicted and ground truth text.

        Args:
            predicted: Predicted text
            groundtruth: Ground truth text

        Returns:
            MetricResult with edit distance score
        """
        if not isinstance(predicted, str) or not isinstance(groundtruth, str):
            return MetricResult.create_error_result(
                self.name, "Both inputs must be strings"
            )

        # Calculate edit distance using difflib
        distance = self._levenshtein_distance(predicted, groundtruth)

        # Normalize by the length of the longer string
        if self.normalize:
            max_len = max(len(predicted), len(groundtruth))
            # if max_len == 0:
            #     score = 1.0  # Both strings are empty
            # else:
            #     score = 1.0 - (distance / max_len)
            if max_len == 0:
                # 两者都为空时标记为失败
                return MetricResult.create_error_result(
                    self.name, "Both predicted and groundtruth are empty"
                )

            score = 1.0 - (distance / max_len)
            return MetricResult(
                metric_name=self.name,
                score=score,
                details={
                    "distance": distance,
                    "predicted_length": len(predicted),
                    "groundtruth_length": len(groundtruth),
                    "normalized": True,
                },
            )
        else:
            score = distance

        details = {
            "distance": distance,
            "predicted_length": len(predicted),
            "groundtruth_length": len(groundtruth),
            "normalized": self.normalize,
        }

        return MetricResult(metric_name=self.name, score=score, details=details)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""

        return Levenshtein.distance(s1, s2)


class BleuMetric(BaseMetric):
    """BLEU score metric for text similarity."""

    version = "1.0.0"
    description = "BLEU score for text similarity evaluation"

    @classmethod
    def create(cls, config: dict[str, t.Any] | None = None) -> "BleuMetric":
        return cls("bleu", config)

    def _setup(self) -> None:
        self._smoothing = bleu_score.SmoothingFunction()
        self.max_n = self.config.get("max_n", 4)
        self.smoothing_method = self.config.get("smoothing_method", "method1")

    def _calculate_score(
        self, predicted: str, groundtruth: str, **kwargs
    ) -> MetricResult:
        """
        Calculate BLEU score between predicted and ground truth text.

        Args:
            predicted: Predicted text
            groundtruth: Ground truth text

        Returns:
            MetricResult with BLEU score
        """
        if not isinstance(predicted, str) or not isinstance(groundtruth, str):
            return MetricResult.create_error_result(
                self.name, "Both inputs must be strings"
            )

        # Tokenize texts (simple whitespace tokenization)
        predicted_tokens = predicted.split()
        groundtruth_tokens = groundtruth.split()

        if not predicted_tokens and not groundtruth_tokens:
            score = 1.0  # Both are empty
        elif not predicted_tokens or not groundtruth_tokens:
            score = 0.0  # One is empty
        else:
            # Calculate BLEU score
            smoothing_func = getattr(self._smoothing, self.smoothing_method)
            score = bleu_score.sentence_bleu(
                [groundtruth_tokens],
                predicted_tokens,
                smoothing_function=smoothing_func,
            )

        details = {
            "predicted_tokens": len(predicted_tokens),
            "groundtruth_tokens": len(groundtruth_tokens),
            "max_n": self.max_n,
            "smoothing_method": self.smoothing_method,
        }

        return MetricResult(metric_name=self.name, score=score, details=details)


class RougeMetric(BaseMetric):
    """ROUGE score metric for text similarity."""

    version = "1.0.0"
    description = "ROUGE score for text similarity evaluation"

    @classmethod
    def create(cls, config: dict[str, t.Any] | None = None) -> "RougeMetric":
        return cls("rouge", config)

    def _setup(self) -> None:
        """Setup the ROUGE metric."""

        self.rouge_types = self.config.get(
            "rouge_types", ["rouge-1", "rouge-2", "rouge-l"]
        )
        self.use_stemmer = self.config.get("use_stemmer", True)
        self._rouge = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=self.use_stemmer
        )

    def _calculate_score(
        self, predicted: str, groundtruth: str, **kwargs
    ) -> MetricResult:
        """
        Calculate ROUGE score between predicted and ground truth text.

        Args:
            predicted: Predicted text
            groundtruth: Ground truth text

        Returns:
            MetricResult with ROUGE scores
        """
        if not isinstance(predicted, str) or not isinstance(groundtruth, str):
            return MetricResult.create_error_result(
                self.name, "Both inputs must be strings"
            )

        if not predicted.strip() and not groundtruth.strip():
            # Both are empty
            score = 1.0
            details = {
                "rouge-1": {"f": 1.0},
                "rouge-2": {"f": 1.0},
                "rouge-l": {"f": 1.0},
            }
        elif not predicted.strip() or not groundtruth.strip():
            # One is empty
            score = 0.0
            details = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }
        else:
            # Calculate ROUGE scores
            try:
                scores = self._rouge.score(predicted, groundtruth)[0]
                # Use ROUGE-L F1 as the main score
                score = scores["rouge-l"]["f"]
                details = scores
            except Exception as e:
                return MetricResult.create_error_result(
                    self.name, f"ROUGE calculation failed: {str(e)}"
                )

        return MetricResult(metric_name=self.name, score=score, details=details)


# class TextEditMetric(EditDistanceMetric):
#     version = "1.0"
#     description = "Pure text edit distance metric (excluding code, tables, formulas)"

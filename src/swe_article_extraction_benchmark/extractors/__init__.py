from swe_article_extraction_benchmark.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    ExtractorFactory,
)
from swe_article_extraction_benchmark.extractors.trafilatura_extractor import (
    TrafilaturaExtractor,
    TrafilaturaTxtExtractor,
)

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExtractorFactory",
    "TrafilaturaExtractor",
    "TrafilaturaTxtExtractor",
]

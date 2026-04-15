"""Extractor with resiliparse."""

import traceback
from dataclasses import dataclass

from resiliparse.extract.html2text import extract_plain_text

from swe_article_extraction_benchmark.extractors import (
    BaseExtractor,
    ExtractionResult,
    shared,
)


@dataclass
class ResiliparseInferenceConfig:
    """Configuration for Resiliparse extractor."""

    main_content: bool = True
    alt_texts: bool = True
    links: bool = False
    form_fields: bool = False
    noscript: bool = False
    list_bullets: bool = True
    preserve_formatting: bool = True
    comments: bool = True
    post_meta: bool = True
    hidden_elements: bool = False


class ResiliparseExtractor(BaseExtractor, name="resiliparse"):
    """Extractor using Resiliparse."""

    description: str = "Resiliparse based content extractor"

    def __init__(self, name: str, config: dict[str, t.Any] | None = None) -> None:
        super().__init__(name, config)
        self.inference_config = ResiliparseInferenceConfig()

        if config:
            for key, value in config.items():
                if hasattr(self.inference_config, key):
                    setattr(self.inference_config, key, value)

    def _setup(self) -> None:
        pass

    def _extract_content(self, html: str, url: str | None = None) -> ExtractionResult:
        try:
            content = extract_plain_text(
                html,
                main_content=self.inference_config.main_content,
                alt_texts=self.inference_config.alt_texts,
                links=self.inference_config.links,
                form_fields=self.inference_config.form_fields,
                noscript=self.inference_config.noscript,
                list_bullets=self.inference_config.list_bullets,
                preserve_formatting=self.inference_config.preserve_formatting,
                comments=self.inference_config.comments,
            )
            return ExtractionResult(
                content=content,
                title=shared.extract_title(html),
                language=shared.detect_language(content),
                success=True,
            )
        except Exception as exc:
            return ExtractionResult.create_error_result(
                f"Resiliparse extraction failed: {exc}",
                error_traceback="".join(traceback.format_exception(exc)),
            )

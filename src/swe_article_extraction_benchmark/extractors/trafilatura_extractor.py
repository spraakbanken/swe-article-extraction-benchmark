import traceback
import typing as t
from dataclasses import dataclass

import trafilatura

from swe_article_extraction_benchmark.extractors import (
    BaseExtractor,
    ExtractionResult,
    shared,
)


@dataclass
class TrafilaturaInferenceConfig:
    """Configuration for Trafilatura extractor."""

    favor_precision: bool = True
    favor_recall: bool = True
    include_comments: bool = False
    include_tables: bool = True
    include_images: bool = False
    include_links: bool = False
    with_metadata: bool = False
    skip_elements: bool = False
    output_format: str = (
        "markdown"  # one of: "csv", "json", "html", "markdown", "txt", "xml"
    )


class TrafilaturaExtractor(BaseExtractor, name="trafilatura"):
    """Extractor using Trafilatura."""

    description: str = "Trafilatura based content extractor in markdown format"

    def __init__(self, name: str, config: dict[str, t.Any] | None = None) -> None:
        super().__init__(name, config)
        self.inference_config = TrafilaturaInferenceConfig()

        # update config
        if config:
            for key, value in config.items():
                if hasattr(self.inference_config, key):
                    setattr(self.inference_config, key, value)

    def _setup(self) -> None:
        """Set up the Trafilatura extractor."""
        pass

    def _extract_content(self, html: str, url: str | None = None) -> ExtractionResult:
        try:
            content = trafilatura.extract(html, url=url)

            # content_list = []
            # if content:
            #     paragraphs = content.split("\n\n")
            #     for i, para in enumerate(paragraphs):
            #         if para_stripped := para.strip():
            #             content_list.append(
            #                 {"type": "paragraph", "content": para_stripped, "index": i}
            #             )
            return ExtractionResult(
                content=content or "",
                # content_list=content_list,
                title=shared.extract_title(html),
                language=shared.detect_language(content),
                success=True,
            )
        except Exception as exc:
            return ExtractionResult.create_error_result(
                f"Trafilatura extracion failed: {str(exc)}",
                error_traceback="".join(traceback.format_exception(exc)),
            )


class TrafilaturaTxtExtractor(TrafilaturaExtractor, name="trafilatura_txt"):
    """Extractor using Trafilatura."""

    description = "Trafilatura based content extractor in text format"

    def __init__(self, name: str, config: dict[str, t.Any] | None = None) -> None:
        super().__init__(name, config)

    def _extract_content(self, html: str, url: str | None = None) -> ExtractionResult:
        """
        Extract content using Trafilatura.

        Args:
            html: HTML content to extract from
            url: Optional URL of the page

        Returns:
            ExtractionResult instance
        """
        try:
            _postbody, content, _len_text = trafilatura.baseline(html)

            # # 创建 content_list（简单分割段落）
            # content_list = []
            # if content:
            #     paragraphs = content.split("\n\n")
            #     for i, para in enumerate(paragraphs):
            #         if para.strip():
            #             content_list.append(
            #                 {"type": "paragraph", "content": para.strip(), "index": i}
            #             )

            return ExtractionResult(
                content=content,
                # content_list=content_list,
                title=shared.extract_title(html),
                language=shared.detect_language(content),
                success=True,
            )

        except Exception as exc:
            return ExtractionResult.create_error_result(
                f"Trafilatura extraction failed: {str(exc)}",
                error_traceback="".join(traceback.format_exception(exc)),
            )

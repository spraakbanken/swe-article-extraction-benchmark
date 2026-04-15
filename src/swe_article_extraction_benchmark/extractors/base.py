import abc
import time
import traceback
import typing as t
from dataclasses import dataclass

ContentListItem = t.TypedDict(
    "ContentListItem", {"type": str, "content": str, "index": int}
)


@dataclass
class ExtractionResult:
    content: str = ""
    # content_list: list[ContentListItem] | None = None
    main_html: str = ""
    version: str | None = None

    # Metadata
    success: bool = True
    extraction_time_s: float = 0.0
    error_message: str | None = None
    error_traceback: str | None = None

    # Additional extracted information
    title: str | None = None
    author: str | None = None
    publish_date: str | None = None
    language: str | None = None

    # Quality indicators
    confidence_score: float | None = None  # 0.0 to 1.0

    # def __post_init__(self) -> None:
    #     if self.content_list is None:
    #         self.content_list = []

    @classmethod
    def create_error_result(
        cls,
        error_message: str,
        error_traceback: str | None = None,
        extraction_time_s: float = 0.0,
    ) -> "ExtractionResult":
        if error_traceback is None:
            error_traceback = ""
        return cls(
            success=False,
            error_message=error_message,
            error_traceback=error_traceback,
            extraction_time_s=extraction_time_s,
        )


class BaseExtractor(abc.ABC):
    """BaseExtractor."""

    description: str = ""

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ExtractorFactory.register(name=name, extractor_class=cls)

    def __init__(self, name: str, config: dict[str, t.Any] | None = None) -> None:
        self.name = name
        self.config: dict[str, t.Any] = config or {}
        self._setup()

    def get_config(self) -> dict[str, t.Any]:
        return self.config.copy()

    @abc.abstractmethod
    def _setup(self) -> None:
        """Setup the extractor."""

    @abc.abstractmethod
    def _extract_content(self, html: str, url: str | None = None) -> ExtractionResult:
        """Extract content from HTML."""

    def extract(self, html: str, url: str | None = None) -> ExtractionResult:
        """Extract content with error handling and timing."""

        start_time = time.perf_counter()
        if not html or not html.strip():
            return ExtractionResult.create_error_result(
                "Empty HTML input", extraction_time_s=time.perf_counter() - start_time
            )
        try:
            result = self._extract_content(html, url)
            result.extraction_time_s = time.perf_counter() - start_time

            return result
        except Exception as e:
            error_message = f"Extracion failed: {str(e)}"
            error_traceback = traceback.format_exc()
            return ExtractionResult.create_error_result(
                error_message,
                error_traceback,
                extraction_time_s=time.perf_counter() - start_time,
            )


class ExtractorFactory:
    """Factory for creating extractors."""

    _registry: dict[str, type[BaseExtractor]] = {}

    @classmethod
    def register(cls, name: str, extractor_class: type[BaseExtractor]) -> None:
        cls._registry[name] = extractor_class

    @classmethod
    def create(cls, name: str, config: dict[str, t.Any] | None = None) -> BaseExtractor:
        if name not in cls._registry:
            available_extractors = ",".join(cls._registry.keys())
            raise ValueError(
                f"Unknown extractor: {name}. Available: {available_extractors}"
            )

        extractor_class = cls._registry[name]
        return extractor_class(name=name, config=config)

    @classmethod
    def create_many_or_all(
        cls,
        names: list[str] | None = None,
        extractor_configs: dict[str, dict[str, t.Any] | None] | None = None,
        default_config: dict[str, t.Any] | None = None,
    ) -> list[BaseExtractor]:
        """Create many or all extractors.

        - If `extractor_configs` is given that is used.
        - If `names` is given that is used with optionally given `default_config`.
        - Otherwise all registered extractors are used with optionally given `default_config`.

        Args:
            names: list of extractors to create (default: None)
            extractor_configs: dictionary mapping extractor names to their configs (default: None)
            default_config: default extractor config (default: None)
        """
        extractors = []
        if extractor_configs is None:
            names_ = names or list(cls._registry.keys())
            extractor_configs = {name: default_config for name in names_}
        for name, config in extractor_configs.items():
            extractor = ExtractorFactory.create(name=name, config=config)
            extractors.append(extractor)
        return extractors

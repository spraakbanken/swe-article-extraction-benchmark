import enum
import typing as t
from dataclasses import dataclass
from enum import StrEnum


class DataDifficulty(StrEnum):
    Easy = enum.auto()
    Medium = enum.auto()
    Hard = enum.auto()


@dataclass
class DataSample:
    id: str
    html: str
    groundtruth_content: str

    url: str | None = None
    domain: str | None = None
    language: str | None = None
    content_type: str | None = None
    difficulty: DataDifficulty | None = None
    tags: list[str] | None = None
    main_html: str | None = None

    # Extracted results (populated during evaluation)
    extracted_result: dict[str, t.Any] | None = None

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "html": self.html,
            "groundtruth_content": self.groundtruth_content,
            "main_html": self.main_html,
            "url": self.url,
            "domain": self.domain,
            "language": self.language,
            "content_type": self.content_type,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "DataSample":
        return cls(**data)


class Dataset:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.samples: list[DataSample] = []
        self._metadata: dict[str, t.Any] = {}

    def add_sample(self, sample: DataSample) -> None:
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

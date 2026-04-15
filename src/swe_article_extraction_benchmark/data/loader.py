from pathlib import Path

import json_arrays

from swe_article_extraction_benchmark.data.dataset import DataSample, Dataset


class DataLoader:
    """Data loader."""

    @staticmethod
    def load_jsonl(file_path: Path, **kwargs) -> Dataset:
        dataset_name = kwargs.get("name", file_path.stem)
        dataset = Dataset(name=dataset_name)

        for idx, raw in enumerate(json_arrays.load_from_file(file_path), start=1):
            try:
                sample = DataSample.from_dict(raw)
                dataset.add_sample(sample)
            except Exception as e:
                print(f"ERROR Failed to load sample at line {idx}: {e}")

        return dataset

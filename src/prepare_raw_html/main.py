import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path

import json_arrays

from swe_article_extraction_benchmark.data.dataset import DataSample


def main() -> None:
    args = _parse_args()
    input_file = Path(args.input_file)
    dataset_path = Path(args.output_file)
    print(f"Reading raw html from '{input_file}' to '{dataset_path}' ...")

    _process(input_file, dataset_path)


def _process(input_file: Path, dataset_path: Path) -> None:
    input_dir = input_file.parent
    source = input_file.read_text()
    print(f"{source=}")
    with json_arrays.sink_from_file(dataset_path, file_mode="wb") as sink:
        for file_metadata in json_arrays.load_from_file(input_file):
            print(f"{file_metadata=}")
            file_id = uuid.uuid4().hex
            file_content = read_content(input_dir / file_metadata["file"])
            groundtruth_content = read_content(input_dir / file_metadata["groundtruth"])

            sample = DataSample(
                id=file_id,
                tags=file_metadata.get("tags"),
                domain=file_metadata.get("domain"),
                url=file_metadata["url"],
                html=file_content,
                groundtruth_content=groundtruth_content,
            )
            sink.send(sample.to_dict())


def read_content(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("iso-8859-1")


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="prepare all html files")
    parser.add_argument("input_file")
    parser.add_argument("output_file", default="data/dataset_sample.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    main()

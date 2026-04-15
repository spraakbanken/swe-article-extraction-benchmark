import argparse
from pathlib import Path

from swe_article_extraction_benchmark import runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=Path, default="data/dataset_sample.jsonl"
    )
    parser.add_argument("--formatters", type=str, nargs="*")
    args = parser.parse_args()

    dataset_path = args.dataset
    formatters = (
        args.formatters if args.formatters and len(args.formatters) > 0 else None
    )

    if formatters and len(formatters) == 1:
        runner.evaluate_extractor(dataset_path, formatters[0])
    else:
        runner.compare_extractors(dataset_path, formatters)

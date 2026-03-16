import gzip
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path("html")
    print(f"Reading raw html from '{input_dir}' to '{output_dir}' ...")

    _process(input_dir, output_dir)


def _process(input_dir: Path, output_dir: Path) -> None:

    for path in input_dir.iterdir():
        print(f"processing {path} ...")
        if path.is_dir():
            new_output_dir = output_dir / path.stem
            new_output_dir.mkdir(parents=True, exist_ok=True)
            _process(path, new_output_dir)
        elif path.is_file():
            filename = output_dir / f"{uuid.uuid4().hex}.html.gz"
            with gzip.open(filename, mode="wt") as zip_file:
                content = path.read_text()
                zip_file.write(content)
            path.unlink()


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="prepare all html files")
    parser.add_argument("input_dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download a streaming Persian Wikipedia sample for Graph-LM tests."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.fa"

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Persian Wikipedia from Hugging Face into a text file."
    )
    parser.add_argument(
        "--output",
        default="data/wiki_fa_50k.txt",
        help="Output text file path. One cleaned article per line.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Maximum number of rows to write after filtering.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum cleaned text length in characters.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Apply lightweight Persian-friendly cleanup and force one-line records."""
    text = text.replace("\u200c", " ")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_streaming_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing required package: datasets\n"
            "Install it with:\n"
            "  pip install datasets"
        ) from exc

    return load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )


def main() -> int:
    args = parse_args()
    if args.max_rows < 1:
        raise SystemExit("--max-rows must be greater than 0")
    if args.min_length < 0:
        raise SystemExit("--min-length must be 0 or greater")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_streaming_dataset()
    written = 0
    seen = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for row in dataset:
            seen += 1
            text = row.get("text", "")
            if not isinstance(text, str):
                continue

            text = clean_text(text)
            if len(text) < args.min_length:
                continue

            output_file.write(text + "\n")
            written += 1
            if written >= args.max_rows:
                break

            if written and written % 1_000 == 0:
                print(
                    f"written={written} seen={seen}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"Saved {written} rows to {output_path} "
        f"(seen={seen}, min_length={args.min_length})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

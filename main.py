"""Entry point for the multimodal invoice field extractor.

Usage:
    python main.py path/to/invoice.pdf
    python main.py path/to/invoice.png --output result.json
    python main.py path/to/invoice.png --pretty
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import Config, configure_logging
from src.pipeline import InvoiceExtractionPipeline

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured fields from an invoice image or PDF.")
    parser.add_argument("input", help="Path to the invoice file (PNG, JPG, or PDF).")
    parser.add_argument(
        "-o", "--output",
        help="Path to write the extracted JSON. Defaults to <OUTPUT_DIR>/<input_stem>.json",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_env()
    configure_logging(config.log_level)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.json"

    pipeline = InvoiceExtractionPipeline(config)
    result = pipeline.process(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote extraction result to %s", output_path)

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

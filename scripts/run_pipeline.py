"""
Command-line entry point.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow `python scripts/run_pipeline.py` from the project root without install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from microglia_annotator import load_config, run_pipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Microglia VAE annotation pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    print("\n--- summary ---")
    for k, v in result.items():
        if k == "enrichment_table":
            continue
        print(f"{k}: {v}")
    if result.get("enrichment_table") is not None:
        print("\nAD vs Control enrichment (top rows):")
        print(result["enrichment_table"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

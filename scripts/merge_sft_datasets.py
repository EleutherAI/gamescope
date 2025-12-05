"""Merge multiple Diplomacy SFT dataset JSONL files into a unified dataset.

Handles deduplication based on (game_id, phase, power, type) tuple.
Outputs merged dataset to a specified location.

Example:
    python scripts/merge_sft_datasets.py \
        --input_files results/diplomacy/sft/dataset_gen/*/sft_dataset.jsonl \
        --output_file results/diplomacy/sft/merged_dataset.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_key(record: Dict) -> Tuple:
    """Create a unique key for deduplication.

    Key is (game_id, phase, power, type, subround) where subround defaults to None.
    """
    return (
        record.get("game_id"),
        record.get("phase"),
        record.get("power"),
        record.get("type"),
        record.get("subround"),  # None for orders/journal, int for negotiation
    )


def merge_datasets(input_files: List[Path], output_file: Path) -> Dict:
    """Merge multiple JSONL files, deduplicating by key.

    Returns stats dict with counts.
    """
    seen_keys: Set[Tuple] = set()
    merged_records: List[Dict] = []
    stats = {
        "total_input": 0,
        "duplicates_skipped": 0,
        "final_count": 0,
        "by_type": {},
        "by_source": {},
    }

    for input_path in input_files:
        source_name = input_path.parent.name
        stats["by_source"][source_name] = {"read": 0, "kept": 0}

        logger.info(f"Processing {input_path}")

        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line in {input_path}: {e}")
                    continue

                stats["total_input"] += 1
                stats["by_source"][source_name]["read"] += 1

                key = make_key(record)

                if key in seen_keys:
                    stats["duplicates_skipped"] += 1
                    continue

                seen_keys.add(key)
                merged_records.append(record)
                stats["by_source"][source_name]["kept"] += 1

                # Track by type
                record_type = record.get("type", "unknown")
                stats["by_type"][record_type] = stats["by_type"].get(record_type, 0) + 1

    stats["final_count"] = len(merged_records)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for record in merged_records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Wrote {stats['final_count']} records to {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge Diplomacy SFT datasets")
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Input JSONL files (supports glob patterns via shell)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/diplomacy/sft/merged_dataset.jsonl",
        help="Output merged JSONL file",
    )

    args = parser.parse_args()

    input_paths = [Path(f) for f in args.input_files]
    output_path = Path(args.output_file)

    # Validate inputs exist
    for p in input_paths:
        if not p.exists():
            logger.error(f"Input file not found: {p}")
            return

    stats = merge_datasets(input_paths, output_path)

    # Print summary
    print("\n=== Merge Summary ===")
    print(f"Total input records: {stats['total_input']}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"Final merged count: {stats['final_count']}")
    print("\nBy type:")
    for t, count in sorted(stats["by_type"].items()):
        print(f"  {t}: {count}")
    print("\nBy source:")
    for src, counts in stats["by_source"].items():
        print(f"  {src}: {counts['read']} read, {counts['kept']} kept")


if __name__ == "__main__":
    main()

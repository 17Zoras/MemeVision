import argparse
import csv
import os
from collections import Counter

from feature_schema import EXPECTED_FEATURES, validate_features


def inspect_dataset(path):
    counts = Counter()
    sources = Counter()
    recordings = Counter()
    invalid = Counter()
    rows = 0
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows += 1
            try:
                label = row["label"]
                values = [float(value) for value in row["features"].split(",")]
                validate_features(values)
                counts[label] += 1
                sources[row.get("source") or "unknown"] += 1
                recordings[row.get("recording_id") or "unknown"] += 1
            except (KeyError, TypeError, ValueError) as error:
                invalid[str(error)] += 1

    return rows, counts, sources, recordings, invalid


def main():
    parser = argparse.ArgumentParser(description="Inspect MemeVision CSV data quality.")
    parser.add_argument("path", nargs="?", default="data/training_data.csv")
    args = parser.parse_args()
    rows, counts, sources, recordings, invalid = inspect_dataset(args.path)
    print(f"Dataset: {args.path}")
    print(f"Rows: {rows}")
    print(f"Expected features: {EXPECTED_FEATURES}")
    print("Class counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print("Source counts:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
    print(f"Recording IDs: {len(recordings)}")
    print(f"Invalid rows: {sum(invalid.values())}")
    for reason, count in invalid.items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
"""Create the balanced V6 training CSV without modifying the V5 CSV."""

import argparse
import csv
import os
import random
from collections import Counter

from feature_schema import EXPECTED_FEATURES, validate_features

LABELS = ("other", "shocked", "thinking", "thumbsup", "thumbsdown")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_samples(path):
    samples = []
    skipped = Counter()
    if not os.path.exists(path):
        return samples, skipped
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            try:
                label = row["label"]
                values = row["features"].split(",")
                if label not in LABELS or len(values) != EXPECTED_FEATURES:
                    raise ValueError("invalid label or feature count")
                validate_features([float(value) for value in values])
                samples.append((label, values))
            except (KeyError, TypeError, ValueError) as error:
                skipped[str(error)] += 1
    return samples, skipped


def write_samples(path, samples):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "features"])
        writer.writerows(
            (label, ",".join(values))
            for label, values in samples
        )


def print_counts(title, samples):
    counts = Counter(label for label, _ in samples)
    print(title)
    for label in LABELS:
        print(f"{label}: {counts[label]}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=os.path.join(BASE_DIR, "data", "training_data.csv"))
    parser.add_argument("--external", default=os.path.join(BASE_DIR, "data", "hf_training_data.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "data", "v6_training_data.csv"))
    parser.add_argument("--max-external-other", type=int, default=250)
    parser.add_argument("--max-external-thumb", type=int, default=120)
    args = parser.parse_args()

    local, local_skipped = read_samples(args.local)
    external, external_skipped = read_samples(args.external)
    print_counts("LOCAL DATA", local)
    print_counts("HUGGING FACE DATA", external)
    print(f"Skipped local rows: {sum(local_skipped.values())}")
    print(f"Skipped HF rows: {sum(external_skipped.values())}")

    rng = random.Random(42)
    selected_external = []
    per_class = {"other": args.max_external_other, "thumbsup": args.max_external_thumb,
                 "thumbsdown": args.max_external_thumb, "shocked": 0, "thinking": 0}
    for label in LABELS:
        candidates = [sample for sample in external if sample[0] == label]
        rng.shuffle(candidates)
        selected_external.extend(candidates[:per_class[label]])

    final_samples = local + selected_external
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_samples(args.output, final_samples)
    final_counts = print_counts("FINAL DATASET", final_samples)
    print(f"Total samples: {len(final_samples)}")
    print(f"Feature count: {EXPECTED_FEATURES}")
    print(f"External samples selected: {len(selected_external)}")
    print(f"Saved: {args.output}")
    if min(final_counts.values()) == 0:
        raise SystemExit("Every class must have at least one sample")


if __name__ == "__main__":
    main()

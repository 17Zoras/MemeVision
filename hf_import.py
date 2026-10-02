"""Import a small, conservative subset of HaGRID as 60-feature samples."""

import csv
import os
import argparse

import cv2
import numpy as np
from datasets import load_dataset

from features import extract_features
from feature_schema import EXPECTED_FEATURES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "hf_training_data.csv")
DATASET_NAME = "s17660101713/hagrid-subset"
MAX_PER_CLASS = 250
TARGET_LABELS = {
    "like": "thumbsup",
    "dislike": "thumbsdown",
    "fist": "other",
    "point": "other",
    "no_gesture": "other",
}


def cv2_from_image(image):
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def label_names_for(dataset):
    label_feature = dataset.features.get("label")
    if hasattr(label_feature, "names"):
        return label_feature.names
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-other", type=int, default=MAX_PER_CLASS)
    parser.add_argument("--max-thumb", type=int, default=120)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Loading {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as error:
        print(f"Hugging Face import unavailable: {error}")
        print("Keeping the existing local/Hugging Face CSV unchanged.")
        return
    try:
        label_names = label_names_for(dataset)
        if label_names is None:
            raise ValueError("Expected a ClassLabel 'label' column in the dataset")
    except Exception as error:
        print(f"Hugging Face schema unavailable: {error}")
        print("Keeping the existing local/Hugging Face CSV unchanged.")
        return

    print("Available labels:", ", ".join(label_names))
    counts = {label: 0 for label in set(TARGET_LABELS.values())}
    limits = {
        "other": args.max_other,
        "thumbsup": args.max_thumb,
        "thumbsdown": args.max_thumb,
    }
    skipped = {}
    written = 0

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "features"])
        for index, sample in enumerate(dataset):
            label_id = sample.get("label")
            if not isinstance(label_id, int) or label_id >= len(label_names):
                skipped["invalid label"] = skipped.get("invalid label", 0) + 1
                continue
            source_label = label_names[label_id]
            target_label = TARGET_LABELS.get(source_label)
            if target_label is None or counts[target_label] >= limits[target_label]:
                continue
            image = sample.get("image")
            try:
                if image is None:
                    raise ValueError("missing image")
                features = extract_features(cv2_from_image(image))
                if features.shape != (EXPECTED_FEATURES,):
                    raise ValueError(f"expected {EXPECTED_FEATURES} features, got {features.shape}")
                # Zero vectors mean neither detector found useful landmarks.
                if not np.any(features):
                    raise ValueError("no face or hand landmarks detected")
                writer.writerow([target_label, ",".join(map(str, features.tolist()))])
                counts[target_label] += 1
                written += 1
                if written % 25 == 0:
                    print(f"Processed {written}: {counts}")
                if all(counts[label] >= limits[label] for label in limits):
                    print("Requested Hugging Face sample limits reached.")
                    break
            except Exception as error:
                reason = str(error)
                skipped[reason] = skipped.get(reason, 0) + 1
                print(f"Skipped image {index}: {reason}")

    print("HF IMPORT COMPLETE")
    print("Counts:", counts)
    print("Skipped:", sum(skipped.values()))
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
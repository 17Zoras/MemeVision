"""Train the V6 landmark model; validation on similar recordings can overestimate reality."""

import argparse
import csv
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from feature_schema import EXPECTED_FEATURES, validate_features

REQUIRED_LABELS = {"other", "shocked", "thinking", "thumbsup", "thumbsdown"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path):
    features, labels = [], []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            values = validate_features([float(value) for value in row["features"].split(",")])
            features.append(values)
            labels.append(row["label"])
    if not features:
        raise ValueError(f"No training samples found in {path}")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "data", "v6_training_data.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "model_v6.pkl"))
    args = parser.parse_args()

    X, y = load_dataset(args.input)
    missing_labels = REQUIRED_LABELS - set(y)
    if missing_labels:
        raise ValueError(
            "Training data is missing required labels: "
            + ", ".join(sorted(missing_labels))
        )
    print("FINAL DATASET")
    for label, count in sorted(Counter(y).items()):
        print(f"{label}: {count}")
    print(f"Total samples: {len(X)}")
    print(f"Feature count: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=120, max_depth=18, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
    )
    print("Training...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Validation accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))
    with open(args.output, "wb") as file:
        pickle.dump(model, file)
    print(f"V6 model saved: {args.output}")


if __name__ == "__main__":
    main()

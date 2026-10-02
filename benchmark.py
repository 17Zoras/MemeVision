import argparse
import os
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from train_v6 import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Benchmark lightweight MemeVision models.")
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "data", "v6_training_data.csv"))
    args = parser.parse_args()
    X, y = load_dataset(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=18, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        predictions = model.predict(X_test)
        predict_ms = (time.perf_counter() - started) * 1000 / len(X_test)
        print(f"{name}: accuracy={accuracy_score(y_test, predictions):.4f} "
              f"macro_f1={f1_score(y_test, predictions, average='macro'):.4f} "
              f"train_ms={train_ms:.1f} predict_ms_per_row={predict_ms:.3f}")
        print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
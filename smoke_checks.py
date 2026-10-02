import csv
import os
import pickle
from pathlib import Path

from PIL import Image, ImageSequence

from feature_schema import EXPECTED_FEATURES, validate_features


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_LABELS = {"other", "shocked", "thinking", "thumbsup", "thumbsdown"}


def check_csv(path):
    rows = 0
    labels = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            validate_features([float(value) for value in row["features"].split(",")])
            labels.add(row["label"])
            rows += 1
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows, labels


def main():
    for filename in ("model.pkl", "model_v6.pkl"):
        with (BASE_DIR / filename).open("rb") as file:
            model = pickle.load(file)
        if model.n_features_in_ != EXPECTED_FEATURES:
            raise ValueError(f"{filename} has {model.n_features_in_} features")
        if not REQUIRED_LABELS.issubset(model.classes_):
            raise ValueError(f"{filename} is missing required labels")
        print(f"model ok: {filename}")

    for filename in ("training_data.csv", "v6_training_data.csv"):
        rows, labels = check_csv(BASE_DIR / "data" / filename)
        print(f"dataset ok: {filename} rows={rows} labels={','.join(sorted(labels))}")

    for path in sorted((BASE_DIR / "memes").iterdir()):
        with Image.open(path) as image:
            frames = sum(1 for _ in ImageSequence.Iterator(image))
            if frames < 1:
                raise ValueError(f"No frames in {path}")
            print(f"meme ok: {path.name} frames={frames} size={image.size}")

    for filename in ("face_landmarker.task", "hand_landmarker.task"):
        path = BASE_DIR / "models" / filename
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        print(f"mediapipe model ok: {filename}")

    print("SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
import os
import csv
import numpy as np

from feature_schema import EXPECTED_FEATURES, validate_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(
    DATA_DIR,
    "training_data.csv"
)


MAX_PER_CLASS = {

    "thinking": 1000,

    "shocked": 1000,

    "thumbsup": 1000,

    "thumbsdown": 1000,

    "other": 2000
}


# =========================================================
# CREATE DATABASE
# =========================================================

def initialize_database():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )


    if not os.path.exists(DATA_FILE):

        with open(
            DATA_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "label",
                "features"
            ])


# =========================================================
# ADD SAMPLE
# =========================================================

def add_sample(label, features):

    initialize_database()
    features = validate_features(features)


    with open(
        DATA_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            label,
            ",".join(
                map(str, features)
            )
        ])


    enforce_limit(label)


# =========================================================
# READ DATA
# =========================================================

def load_data():

    initialize_database()


    X = []
    y = []


    with open(
        DATA_FILE,
        "r"
    ) as file:

        reader = csv.DictReader(file)


        for row in reader:

            y.append(
                row["label"]
            )

            values = validate_features(
                [float(x) for x in row["features"].split(",")]
            )
            X.append(values)


    if len(X) == 0:

        return (
            np.empty((0, 0)),
            np.array([])
        )


    return (
        np.array(X),
        np.array(y)
    )


# =========================================================
# ROLLING LIMIT
# =========================================================

def enforce_limit(label):

    X, y = load_data()


    indexes = np.where(
        y == label
    )[0]


    maximum = MAX_PER_CLASS.get(
        label,
        1000
    )


    if len(indexes) <= maximum:

        return


    # Keep the newest samples
    keep = indexes[-maximum:]


    other_indexes = np.where(
        y != label
    )[0]


    final_indexes = np.concatenate([
        other_indexes,
        keep
    ])


    final_indexes.sort()


    with open(
        DATA_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "label",
            "features"
        ])


        for index in final_indexes:

            writer.writerow([
                y[index],
                ",".join(
                    map(str, X[index])
                )
            ])
from database import load_data

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import pickle


print()
print("======================================")
print("       MEMEVISION V5 TRAINER")
print("======================================")


X, y = load_data()


if len(X) == 0:

    print("Database is empty.")

    print("Run collect.py first.")

    exit()


print(
    "Total samples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)


# =========================================================
# CLASS COUNTS
# =========================================================

print()
print("Samples per class:")

for label in sorted(set(y)):

    print(
        f"{label}: {(y == label).sum()}"
    )


# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=18,

    min_samples_leaf=3,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


print()
print("Training...")

model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST
# =========================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()

print(
    f"Validation accuracy: {accuracy * 100:.2f}%"
)


print()

print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================================================
# SAVE
# =========================================================

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print()
print("======================================")
print("V5 MODEL SAVED")
print("======================================")
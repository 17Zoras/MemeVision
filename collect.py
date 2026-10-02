import cv2
import numpy as np
import time

from features import extract_features
from database import add_sample
from feature_schema import validate_features


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

MIN_SAMPLE_DISTANCE = 0.015

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()


labels = {
    "1": "thinking",
    "2": "shocked",
    "3": "thumbsup",
    "4": "thumbsdown",
    "5": "other"
}


print()
print("======================================")
print("       MEMEVISION DATA COLLECTOR")
print("======================================")
print()
print("1 = Thinking")
print("2 = Shocked")
print("3 = Thumbs Up")
print("4 = Thumbs Down")
print("5 = Other")
print("Q = Quit")
print()


while True:

    choice = input("Choose class: ").strip().lower()

    if choice == "q":
        break

    if choice not in labels:
        print("Invalid choice.")
        continue

    label = labels[choice]

    print()
    print("--------------------------------------")
    print(f"Recording: {label.upper()}")
    print("--------------------------------------")

    print("Get into the pose.")

    print("Starting in 3...")
    time.sleep(1)

    print("2...")
    time.sleep(1)

    print("1...")
    time.sleep(1)

    print("RECORDING!")
    print()

    # =====================================================
    # TEMPORARY SAMPLES
    # =====================================================

    temporary_samples = []
    last_saved_features = None

    start = time.time()

    while time.time() - start < 5:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (640, 480)
        )

        # Extract features
        features = validate_features(extract_features(frame))

        # DON'T save yet
        if (
            last_saved_features is None
            or np.linalg.norm(features - last_saved_features) >= MIN_SAMPLE_DISTANCE
        ):
            temporary_samples.append(features)
            last_saved_features = features

        # Display
        cv2.putText(
            frame,
            f"RECORDING: {label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {len(temporary_samples)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = stop early",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "MemeVision Collector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =====================================================
    # ASK WHETHER TO KEEP
    # =====================================================

    print()
    print("--------------------------------------")
    print(f"Recorded {len(temporary_samples)} frames for {label}")
    print("--------------------------------------")

    while True:

        confirm = input(
            "Keep these samples? (Y/N): "
        ).strip().lower()

        if confirm == "y":

            for features in temporary_samples:

                add_sample(
                    label,
                    features
                )

            print()
            print(
                f"✓ Added {len(temporary_samples)} samples "
                f"for {label}"
            )

            break

        elif confirm == "n":

            # Simply throw away the temporary list
            temporary_samples.clear()

            print()
            print("✗ Samples discarded.")

            break

        else:

            print("Please enter Y or N.")


cap.release()

cv2.destroyAllWindows()

print()
print("======================================")
print("Collector closed.")
print("======================================")
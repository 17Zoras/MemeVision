import cv2
import pickle
import os
import sys
import time
import tkinter as tk
import threading
import queue

import numpy as np
from collections import deque, Counter
from PIL import Image, ImageTk, ImageSequence

from features import extract_features
from feature_schema import EXPECTED_FEATURES, validate_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# SETTINGS
# =========================================================

CAMERA_WIDTH = 600
CAMERA_HEIGHT = 450

MEME_WIDTH = 760
MEME_HEIGHT = 560

# Run MediaPipe/model roughly this many times per second.
# The camera itself can still display smoothly.
INFERENCE_INTERVAL = 0.07


CONFIDENCE_THRESHOLD = 0.55
MIN_MARGIN = 0.12
CANDIDATE_DURATION = 0.55
RELEASE_DURATION = 0.80

HISTORY_SIZE = 8

MEME_COOLDOWN = 0.60


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_FILE = next(
    (
        os.path.join(BASE_DIR, filename)
        for filename in ("model_v6.pkl", "model.pkl")
        if os.path.exists(os.path.join(BASE_DIR, filename))
    ),
    None,
)

if MODEL_FILE is None:

    print("No MemeVision model found!")
    print("Run prepare_v6.py and train_v6.py first.")
    sys.exit(1)


with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


if getattr(model, "n_features_in_", None) != 60:
    raise ValueError("The selected model must accept exactly 60 features")

required_labels = {"thinking", "shocked", "thumbsup", "thumbsdown"}
missing_labels = required_labels - set(model.classes_)
if missing_labels:
    raise ValueError(
        "The selected model is missing meme labels: "
        + ", ".join(sorted(missing_labels))
    )

print(f"MemeVision model loaded: {MODEL_FILE}")


# =========================================================
# MEME FILES
# =========================================================

MEME_CONFIG = {
    "thinking": {
        "path": os.path.join(BASE_DIR, "memes", "thinking.jpg"),
        "minimum_confidence": 0.72,
        "description": "Hand near chin or mouth with compatible face state",
    },
    "shocked": {
        "path": os.path.join(BASE_DIR, "memes", "shocked.jpg"),
        "minimum_confidence": 0.78,
        "description": "Wide-eyed, open-mouth expression",
    },
    "thumbsup": {
        "path": os.path.join(BASE_DIR, "memes", "thumbsup.jpg"),
        "minimum_confidence": 0.76,
        "description": "Thumb-up hand configuration",
    },
    "thumbsdown": {
        "path": os.path.join(BASE_DIR, "memes", "thumbsdown.gif"),
        "minimum_confidence": 0.76,
        "description": "Thumb-down hand configuration",
    },
}

MEMES = {label: config["path"] for label, config in MEME_CONFIG.items()}


# =========================================================
# LOAD MEMES
# =========================================================

loaded_memes = {}


for label, path in MEMES.items():

    if not os.path.exists(path):

        print(
            f"WARNING: {path} not found"
        )

        continue


    try:

        image = Image.open(path)

        frames = []


        for frame in ImageSequence.Iterator(image):

            frames.append(
                frame.convert("RGB").copy()
            )


        duration = image.info.get(
            "duration",
            100
        )


        loaded_memes[label] = {

            "frames": frames,

            "duration": duration
        }


        print(
            f"Loaded meme: {label}"
        )


    except Exception as e:

        print(
            f"Could not load {label}: {e}"
        )


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Camera could not be opened.")

    sys.exit(1)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


# =========================================================
# TKINTER
# =========================================================

root = tk.Tk()

root.title(
    "MemeVision V6"
)

root.geometry(
    "1400x650"
)

root.configure(
    bg="black"
)


# =========================================================
# CAMERA PANEL
# =========================================================

camera_frame = tk.Frame(
    root,
    bg="black"
)

camera_frame.pack(
    side="left",
    padx=10,
    pady=10
)


camera_label = tk.Label(
    camera_frame,
    bg="black"
)

camera_label.pack()


status_label = tk.Label(
    camera_frame,
    text="Waiting for expression...",
    bg="black",
    fg="#bbbbbb",
    font=("Segoe UI", 12)
)

status_label.pack(
    pady=(8, 0)
)


# =========================================================
# MEME PANEL
# =========================================================

meme_frame = tk.Frame(
    root,
    width=MEME_WIDTH,
    height=MEME_HEIGHT,
    bg="black"
)

meme_frame.pack(
    side="right",
    padx=10,
    pady=10
)

meme_frame.pack_propagate(False)


meme_label = tk.Label(
    meme_frame,
    bg="black"
)

meme_label.pack(
    expand=True
)


# =========================================================
# STATE
# =========================================================

prediction_history = deque(
    maxlen=HISTORY_SIZE
)

state_lock = threading.Lock()

last_inference_time = 0

last_prediction = "other"

last_confidence = 0.0

last_top_predictions = []

stable_prediction = "other"

other_since = time.time()

candidate_prediction = "other"

candidate_since = 0.0

release_since = None

current_meme = None

last_meme_time = 0

gif_index = 0

last_gif_time = 0

inference_queue = queue.Queue(maxsize=1)
stop_inference = threading.Event()

debug_enabled = False

camera_frame_count = 0

camera_fps_started = time.time()

camera_fps = 0.0

inference_count = 0

inference_fps_started = time.time()

inference_fps = 0.0

last_inference_ms = 0.0


# =========================================================
# RESIZE MEME
# =========================================================

def resize_meme(image):

    image = image.copy()

    image.thumbnail(
        (
            MEME_WIDTH - 20,
            MEME_HEIGHT - 20
        ),
        Image.Resampling.LANCZOS
    )

    return image


# =========================================================
# SHOW MEME
# =========================================================

def show_meme(label):

    global current_meme
    global gif_index
    global last_gif_time


    if label not in loaded_memes:

        return


    current_meme = label

    gif_index = 0

    last_gif_time = 0


# =========================================================
# HIDE MEME
# =========================================================

def hide_meme():

    global current_meme

    current_meme = None

    meme_label.configure(
        image=""
    )

    meme_label.image = None


# =========================================================
# UPDATE MEME
# =========================================================

def update_meme():

    global gif_index
    global last_gif_time


    if current_meme is None:

        return


    if current_meme not in loaded_memes:

        return


    data = loaded_memes[
        current_meme
    ]


    frames = data["frames"]

    duration = data["duration"]


    if len(frames) == 0:

        return


    now = time.time()


    if (
        now - last_gif_time
        >= duration / 1000
    ):

        image = resize_meme(
            frames[gif_index]
        )


        photo = ImageTk.PhotoImage(
            image
        )


        meme_label.configure(
            image=photo
        )

        meme_label.image = photo


        gif_index = (
            gif_index + 1
        ) % len(frames)


        last_gif_time = now


# =========================================================
# INFERENCE
# =========================================================

def run_inference(frame):

    global last_prediction, last_confidence, last_top_predictions, stable_prediction
    global candidate_prediction, candidate_since, release_since
    global inference_count, inference_fps, inference_fps_started, last_inference_ms

    started = time.perf_counter()

    features = validate_features(extract_features(frame))


    probabilities = model.predict_proba(
        [features]
    )[0]

    probabilities = apply_pose_guards(
        features,
        probabilities
    )


    best_index = probabilities.argmax()


    prediction = model.classes_[
        best_index
    ]


    confidence = probabilities[
        best_index
    ]


    sorted_probabilities = np.sort(probabilities)
    margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
    now = time.time()

    with state_lock:
        last_prediction = prediction
        last_confidence = confidence
        last_top_predictions = [
            (str(model.classes_[index]), float(probabilities[index]))
            for index in np.argsort(probabilities)[-3:][::-1]
        ]
        last_inference_ms = (time.perf_counter() - started) * 1000
        inference_count += 1
        elapsed = now - inference_fps_started
        if elapsed >= 1.0:
            inference_fps = inference_count / elapsed
            inference_count = 0
            inference_fps_started = now

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_history.append(prediction)
        else:
            prediction_history.append("uncertain")

        if len(prediction_history) >= HISTORY_SIZE:
            counts = Counter(prediction_history)
            label, count = counts.most_common(1)[0]
            if label != "uncertain" and count >= 5:
                stable_prediction = label


def apply_pose_guards(features, probabilities):
    """Reject a few high-risk confusions without replacing the trained model."""
    adjusted = probabilities.copy()
    classes = list(model.classes_)
    index_by_class = {label: index for index, label in enumerate(classes)}

    eye_open = max(float(features[0]), float(features[1]))
    mouth_open = float(features[2])
    expressive_face = eye_open >= 0.035 or mouth_open >= 0.075
    hand_slots = (features[8:34], features[34:60])

    shocked_index = index_by_class.get("shocked")
    if shocked_index is not None and not expressive_face:
        adjusted[shocked_index] *= 0.45

    thinking_index = index_by_class.get("thinking")
    if thinking_index is not None:
        # A hand near the face is not enough for thinking when the index finger
        # is visibly extended, which is a common hand-in-front-of-face case.
        for hand in hand_slots:
            if np.any(hand) and hand[22] < 0.30 and hand[3] > 0.90:
                adjusted[thinking_index] *= 0.55
                break

    for label in ("thumbsup", "thumbsdown"):
        thumb_index = index_by_class.get(label)
        if thumb_index is None:
            continue
        for hand in hand_slots:
            finger_lengths = hand[2:7]
            clear_fist = np.any(hand) and np.max(finger_lengths) < 0.85
            if clear_fist:
                adjusted[thumb_index] *= 0.35
                break

    total = adjusted.sum()
    return adjusted / total if total else probabilities


def inference_worker():
    while not stop_inference.is_set():
        try:
            frame = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            run_inference(frame)
        except Exception as error:
            print(f"Inference error: {error}")


# =========================================================
# CAMERA UPDATE
# =========================================================

def update_camera():

    global last_inference_time, camera_frame_count, camera_fps, camera_fps_started
    global other_since
    global last_meme_time


    ret, frame = cap.read()


    if not ret:

        root.after(
            10,
            update_camera
        )

        return


    frame = cv2.flip(
        frame,
        1
    )


    frame = cv2.resize(
        frame,
        (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )
    )


    now = time.time()
    camera_frame_count += 1
    camera_elapsed = now - camera_fps_started
    if camera_elapsed >= 1.0:
        camera_fps = camera_frame_count / camera_elapsed
        camera_frame_count = 0
        camera_fps_started = now

    with state_lock:
        current_prediction = last_prediction
        current_confidence = last_confidence
        current_top_predictions = list(last_top_predictions)
        current_stable = stable_prediction
        current_inference_ms = last_inference_ms
        current_inference_fps = inference_fps

    if current_stable == "other":
        if other_since is None:
            other_since = now
        if now - other_since >= 5.0:
            status_label.configure(
                text="Waiting for expression...",
                fg="#bbbbbb"
            )
        else:
            status_label.configure(
                text="Expression not detected yet...",
                fg="#888888"
            )
        if now - other_since >= 5.0 and current_meme is not None:
            hide_meme()
    else:
        other_since = None
        status_label.configure(
            text=f"Expression detected: {current_stable}",
            fg="#7ee787"
        )


    # =====================================================
    # RUN AI ONLY WHEN NEEDED
    # =====================================================

    if (
        now - last_inference_time
        >= INFERENCE_INTERVAL
    ):

        try:
            inference_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        last_inference_time = now


    # =====================================================
    # TRIGGER MEME
    # =====================================================

    if (
        current_stable in loaded_memes
        and
        current_stable != current_meme
        and
        now - last_meme_time >= MEME_COOLDOWN
    ):

        show_meme(
            current_stable
        )

        last_meme_time = now


    # =====================================================
    # CAMERA TEXT
    # =====================================================

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {current_confidence * 100:.1f}%",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Stable: {current_stable}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    if debug_enabled:
        cv2.putText(frame, f"Camera FPS: {camera_fps:.1f}  AI FPS: {current_inference_fps:.1f}",
                    (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, f"Features: {EXPECTED_FEATURES}  AI ms: {current_inference_ms:.1f}",
                    (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, "Top: " + ", ".join(
            f"{label} {score:.2f}" for label, score in current_top_predictions
        ), (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)


    # =====================================================
    # CAMERA → TKINTER
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        rgb
    )


    photo = ImageTk.PhotoImage(
        image
    )


    camera_label.configure(
        image=photo
    )

    camera_label.image = photo


    # =====================================================
    # MEME
    # =====================================================

    update_meme()


    # =====================================================
    # NEXT FRAME
    # =====================================================

    root.after(
        10,
        update_camera
    )


# =========================================================
# CLOSE
# =========================================================

def close_app():

    stop_inference.set()
    cap.release()

    root.destroy()


def toggle_debug(_event=None):
    global debug_enabled
    debug_enabled = not debug_enabled


root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)
root.bind("<F3>", toggle_debug)


# =========================================================
# START
# =========================================================

print()
print("======================================")
print("MemeVision V6 running!")
print("Press the window X to quit.")
print("======================================")
print()


threading.Thread(target=inference_worker, daemon=True).start()
update_camera()

root.mainloop()
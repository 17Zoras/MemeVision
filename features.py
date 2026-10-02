import cv2
import mediapipe as mp
import numpy as np
import os

from feature_schema import EXPECTED_FEATURES, validate_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(BASE_DIR, "models", "hand_landmarker.task")
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

hand_detector = HandLandmarker.create_from_options(
    hand_options
)


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(BASE_DIR, "models", "face_landmarker.task")
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

face_detector = FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# DISTANCE
# =========================================================

def dist(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


# =========================================================
# NORMALIZE HAND
# =========================================================

def hand_features(hand, face):

    wrist = hand[0]

    # Palm size
    palm_size = dist(
        hand[0],
        hand[9]
    )

    if palm_size < 0.001:
        palm_size = 0.001


    features = []


    # =====================================================
    # PALM POSITION
    # =====================================================

    features.extend([
        hand[9].x,
        hand[9].y
    ])


    # =====================================================
    # FINGER LENGTHS
    # =====================================================

    fingers = [
        (4, 2),
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]

    for tip, base in fingers:

        features.append(
            dist(
                hand[tip],
                hand[base]
            ) / palm_size
        )


    # =====================================================
    # TIP → WRIST DISTANCE
    # =====================================================

    for tip in [4, 8, 12, 16, 20]:

        features.append(
            dist(
                hand[tip],
                wrist
            ) / palm_size
        )


    # =====================================================
    # FINGER TIP RELATIONSHIPS
    # =====================================================

    tips = [4, 8, 12, 16, 20]

    for i in range(len(tips)):

        for j in range(i + 1, len(tips)):

            features.append(
                dist(
                    hand[tips[i]],
                    hand[tips[j]]
                ) / palm_size
            )


    # =====================================================
    # HAND → FACE
    # =====================================================

    if face is not None:

        nose = face[1]
        mouth = face[13]

        features.extend([

            dist(
                hand[9],
                nose
            ),

            dist(
                hand[8],
                mouth
            ),

            dist(
                hand[9],
                face[234]
            ),

            dist(
                hand[9],
                face[454]
            )

        ])

    else:

        features.extend(
            [0.0] * 4
        )


    return features


# =========================================================
# FACE FEATURES
# =========================================================

def face_features(face):

    face_width = dist(
        face[234],
        face[454]
    )

    if face_width < 0.001:
        face_width = 0.001


    features = []


    # Eyes

    features.extend([

        dist(
            face[159],
            face[145]
        ) / face_width,

        dist(
            face[386],
            face[374]
        ) / face_width

    ])


    # Mouth

    features.extend([

        dist(
            face[13],
            face[14]
        ) / face_width,

        dist(
            face[61],
            face[291]
        ) / face_width

    ])


    # Eyebrows

    features.extend([

        dist(
            face[105],
            face[159]
        ) / face_width,

        dist(
            face[334],
            face[386]
        ) / face_width

    ])


    # Nose position

    center_x = (
        face[234].x +
        face[454].x
    ) / 2

    center_y = (
        face[234].y +
        face[454].y
    ) / 2


    features.extend([

        face[1].x - center_x,

        face[1].y - center_y

    ])


    return features


# =========================================================
# MAIN FEATURE EXTRACTOR
# =========================================================

def extract_features(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    face_result = face_detector.detect(
        image
    )

    hand_result = hand_detector.detect(
        image
    )


    features = []


    # =====================================================
    # FACE
    # =====================================================

    if face_result.face_landmarks:

        face = face_result.face_landmarks[0]

        features.extend(
            face_features(face)
        )

    else:

        features.extend(
            [0.0] * 8
        )

        face = None


    # =====================================================
    # HANDS
    # =====================================================

    hands = hand_result.hand_landmarks


    # Always reserve space for 2 hands

    for i in range(2):

        if i < len(hands):

            features.extend(
                hand_features(
                    hands[i],
                    face
                )
            )

        else:

            # 2 palm
            # 5 finger lengths
            # 5 wrist distances
            # 10 tip relationships
            # 4 face relationships

            features.extend(
                [0.0] * 26
            )


    return validate_features(features)
import numpy as np


EXPECTED_FEATURES = 60

FEATURE_NAMES = (
    "left_eye_open",
    "right_eye_open",
    "mouth_open",
    "mouth_width",
    "left_brow_eye_distance",
    "right_brow_eye_distance",
    "nose_offset_x",
    "nose_offset_y",
    "hand1_palm_x",
    "hand1_palm_y",
    "hand1_thumb_length",
    "hand1_index_length",
    "hand1_middle_length",
    "hand1_ring_length",
    "hand1_pinky_length",
    "hand1_thumb_wrist_distance",
    "hand1_index_wrist_distance",
    "hand1_middle_wrist_distance",
    "hand1_ring_wrist_distance",
    "hand1_pinky_wrist_distance",
    "hand1_thumb_index_distance",
    "hand1_thumb_middle_distance",
    "hand1_thumb_ring_distance",
    "hand1_thumb_pinky_distance",
    "hand1_index_middle_distance",
    "hand1_index_ring_distance",
    "hand1_index_pinky_distance",
    "hand1_middle_ring_distance",
    "hand1_middle_pinky_distance",
    "hand1_ring_pinky_distance",
    "hand1_palm_nose_distance",
    "hand1_index_mouth_distance",
    "hand1_palm_left_face_distance",
    "hand1_palm_right_face_distance",
    "hand2_palm_x",
    "hand2_palm_y",
    "hand2_thumb_length",
    "hand2_index_length",
    "hand2_middle_length",
    "hand2_ring_length",
    "hand2_pinky_length",
    "hand2_thumb_wrist_distance",
    "hand2_index_wrist_distance",
    "hand2_middle_wrist_distance",
    "hand2_ring_wrist_distance",
    "hand2_pinky_wrist_distance",
    "hand2_thumb_index_distance",
    "hand2_thumb_middle_distance",
    "hand2_thumb_ring_distance",
    "hand2_thumb_pinky_distance",
    "hand2_index_middle_distance",
    "hand2_index_ring_distance",
    "hand2_index_pinky_distance",
    "hand2_middle_ring_distance",
    "hand2_middle_pinky_distance",
    "hand2_ring_pinky_distance",
    "hand2_palm_nose_distance",
    "hand2_index_mouth_distance",
    "hand2_palm_left_face_distance",
    "hand2_palm_right_face_distance",
)


def validate_features(features):
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature vector contains non-finite values")
    return values
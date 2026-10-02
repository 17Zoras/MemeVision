# 😂 MemeVision

> Real-time gesture recognition that turns your webcam input into meme reactions.

MemeVision is a computer-vision and machine-learning project that detects hand gestures and facial expressions from a live camera feed and maps them to meme reactions.

The project started as a fun idea: **what if your webcam could understand your reaction and instantly show the appropriate meme?**

Under the hood, it became a fairly complete ML pipeline involving landmark detection, feature engineering, dataset collection, model training, validation, inference, and automated checks.

---

## 🎯 What It Does

MemeVision uses a webcam to recognize user gestures/reactions such as:

- 👍 Thumb Up
- 👎 Thumb Down
- 🤔 Thinking
- 😲 Shocked
- 😐 Other

The detected class is then mapped to a corresponding meme.

### Basic pipeline

```text
Webcam
   ↓
MediaPipe Landmark Detection
   ↓
Face + Hand Landmarks
   ↓
Feature Extraction
   ↓
Feature Vector
   ↓
ML Classifier
   ↓
Gesture / Expression Class
   ↓
Meme Selection
   ↓
Display Reaction

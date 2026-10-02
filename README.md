# MemeVision

### Real-Time Gesture Recognition & Context-Aware Meme Retrieval

MemeVision is a real-time computer vision system that performs webcam-based human gesture recognition and maps recognized gestures to predefined visual reactions.

The system uses **MediaPipe Face/Hand Landmarks**, custom feature engineering, a structured feature schema, and supervised machine learning to classify user gestures from low-dimensional landmark representations rather than raw image pixels.

The project was developed through multiple dataset and model iterations, with the latest recorded validation run achieving **97.79% validation accuracy**.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │       Webcam        │
                    │    RGB Video Feed   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   OpenCV Capture    │
                    │   Frame Acquisition │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       MediaPipe Pipeline       │
              │                                │
              │   Face Landmarker              │
              │   Hand Landmarker              │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Landmark Extraction │
                    │                     │
                    │ Face Coordinates    │
                    │ Hand Coordinates    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ Normalization       │
                    │ Coordinate Features │
                    │ Structured Schema   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ML Classification  │
                    │                     │
                    │ Serialized Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gesture Prediction  │
                    │ + Confidence        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Meme Mapping Layer  │
                    │                     │
                    │  Gesture → Meme     │
                    └─────────────────────┘

Core Technical Approach
1. Real-Time Video Acquisition
The application continuously captures frames from the user's webcam using OpenCV.
Each frame is processed through the landmark extraction pipeline before being converted into the numerical representation expected by the classifier.
The pipeline is designed around real-time inference rather than offline image classification.
2. Landmark-Based Computer Vision
Instead of training directly on RGB pixels, MemeVision extracts semantic landmarks from the user's face and hands.
Face
MediaPipe Face Landmarker provides structured facial landmark coordinates.
Hands
MediaPipe Hand Landmarker provides hand keypoints that encode the spatial configuration of the user's hand.
The resulting representation can be viewed as:
Frame
  ↓
Face Landmarks
  +
Hand Landmarks
  ↓
Structured Numerical Representation
This reduces the problem from high-dimensional image classification to classification over engineered numerical features.
3. Feature Engineering
The landmark coordinates are transformed into a deterministic feature vector.
The project maintains an explicit feature schema through:
feature_schema.py

while feature generation is handled by:
features.py

This separation is important because the exact feature ordering and dimensionality must remain consistent between:
Training
   ↓
Validation
   ↓
Model Serialization
   ↓
Inference

A mismatch between training and inference features can result in incorrect predictions or incompatible model input.
4. Feature Schema
MemeVision treats the feature representation as an explicit interface between computer vision and machine learning.
MediaPipe Output
       ↓
Feature Extraction
       ↓
Feature Schema
       ↓
ML Model

The schema provides a consistent contract for:
- Feature dimensionality
- Feature ordering
- Landmark-derived values
- Training/inference compatibility
This makes experimentation with different model versions easier and safer.
5. Supervised Classification
The extracted feature vectors are used to train a supervised classification model.
The project maintains multiple model generations:
model.pkl
model_v6.pkl

with corresponding training pipelines:
train_v5.py
train_v6.py

The trained model is serialized after training and loaded during real-time inference.
6. Dataset Collection Pipeline
Training data is collected using:
collect.py

The collection pipeline captures webcam samples associated with predefined gesture classes.
Current classes include:
thumbsup
thumbsdown
thinking
shocked
other

The collected samples are then prepared and validated before being used for model training.
7. Dataset Preparation
Dataset preparation is separated from model training.
The repository includes:
prepare_v6.py
dataset_stats.py

This allows the dataset to be inspected and transformed independently from the classifier.
Dataset analysis can be used to identify:
- Class distribution
- Malformed samples
- Missing features
- Feature dimensionality problems
- Inconsistencies between samples
8. Model Training
Training is versioned rather than modifying a single training script.
train_v5.py
train_v6.py

This allows different feature representations, datasets, and model configurations to be evaluated without destroying previous experiments.
The latest recorded validation result is:
Validation Accuracy: 97.79%

This metric represents the validation configuration used during development and should not be interpreted as a guarantee of real-world accuracy across unseen users, environments, lighting conditions, or camera hardware.
9. Model Versioning
The project intentionally maintains multiple serialized model artifacts:
model.pkl
model_v6.pkl

This enables comparison between model generations and makes it possible to reproduce or revert to earlier experiments.
The model files are small enough to remain inside the repository rather than requiring external model storage.
10. Real-Time Inference
The inference pipeline follows approximately:
while webcam_is_running:

    frame = capture_frame()

    landmarks = detect_landmarks(frame)

    features = extract_features(landmarks)

    prediction = model.predict(features)

    meme = gesture_to_meme(prediction)

    display(frame, meme)

The classifier operates on the engineered landmark representation rather than directly on the original image.
11. Model ↔ Feature Compatibility
One of the main engineering concerns in the project is maintaining compatibility between:
Feature Generator
        ↕
Feature Schema
        ↕
Trained Model

A model trained on one feature representation cannot safely consume an incompatible representation during inference.
For this reason, the repository includes explicit feature-schema and smoke-check tooling.
12. Validation & Smoke Testing
The repository contains:
smoke_checks.py

for lightweight validation of the application pipeline.
The purpose is to catch issues such as:
- Missing model artifacts
- Invalid feature dimensions
- Incompatible inputs
- Missing dependencies
- Broken project components
before running the full real-time application.
13. Benchmarking
Performance-related experimentation is handled through:
benchmark.py

This separates different aspects of system evaluation:
Correctness
    ↓
Smoke Tests

Model Quality
    ↓
Validation Metrics

Runtime Performance
    ↓
Benchmarking

A model can have high validation accuracy while still being unsuitable for real-time inference if the processing pipeline introduces excessive latency.
14. Model & Runtime Artifacts
The repository contains the trained classifier models:
model.pkl
model_v6.pkl

and MediaPipe task models:
models/
├── face_landmarker.task
└── hand_landmarker.task

These artifacts allow the application to run without requiring the user to retrain the classifier from scratch.
Technology Stack
Computer Vision
- OpenCV
- MediaPipe
- MediaPipe Face Landmarker
- MediaPipe Hand Landmarker
Machine Learning
- scikit-learn
- NumPy
- SciPy
Data Processing
- Python
- CSV-based dataset pipeline
- Custom feature engineering
Model Persistence
- Python Pickle
Development
- Git
- GitHub
- Python virtual environments
Repository Architecture
MemeVision/
│
├── models/
│   ├── face_landmarker.task
│   └── hand_landmarker.task
│
├── memes/
│   ├── blehh.jpg
│   ├── clockit.jpg
│   ├── cooked.jpg
│   ├── happy.gif
│   ├── plotting.jpg
│   ├── sad.png
│   ├── shocked.jpg
│   ├── surprised.jpg
│   ├── suspicious.jpg
│   ├── thinking.jpg
│   ├── thumbsdown.gif
│   └── thumbsup.jpg
│
├── benchmark.py
├── collect.py
├── database.py
├── dataset_stats.py
├── feature_schema.py
├── features.py
├── hf_import.py
├── main.py
├── prepare_v6.py
├── smoke_checks.py
│
├── train_v5.py
├── train_v6.py
│
├── model.pkl
├── model_v6.pkl
│
├── requirements.txt
├── AUDIT.md
├── .gitignore
└── README.md

Installation
Clone the repository:
git clone https://github.com/17Zoras/MemeVision.git
cd MemeVision

Create a virtual environment:
python -m venv .venv

Activate it on Windows:
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Application
Start the real-time application:
python main.py

The application requires:
- A working webcam
- MediaPipe task models
- A trained classifier
- Compatible feature extraction code
Training From Scratch
To reproduce the machine-learning pipeline:
1. Collect samples
python collect.py

2. Inspect the dataset
python dataset_stats.py

3. Prepare the dataset
python prepare_v6.py

4. Train the model
python train_v6.py

The resulting model can then be used by the inference pipeline.
Dataset
The raw training dataset is intentionally not included in the repository.
The dataset contains webcam-derived samples for the project's gesture classes.
Keeping the raw dataset outside Git prevents the repository from becoming unnecessarily large while retaining the complete collection and processing pipeline.
Dataset collection is handled through:
collect.py

and dataset preparation is handled through the associated preparation scripts.
Dataset Inspection
Run:
python dataset_stats.py

This can be used to inspect:
- Class distribution
- Dataset structure
- Feature consistency
- Sample counts
- Invalid or incomplete samples
Validation
Run the project's smoke checks with:
python smoke_checks.py

These checks provide a lightweight way to verify that the core components of the pipeline remain compatible.
Engineering Challenges
Feature Consistency
The same feature representation must be maintained across:
Dataset Creation
       ↓
Feature Extraction
       ↓
Training
       ↓
Model Serialization
       ↓
Inference

Any mismatch can invalidate model predictions.
Dataset Quality
Gesture recognition performance depends heavily on:
- Sample distribution
- Gesture consistency
- Camera angle
- Lighting
- Background
- User variation
Real-Time Constraints
The system continuously performs:
Frame Capture
      +
Landmark Detection
      +
Feature Extraction
      +
Model Inference
      +
Meme Rendering

The entire pipeline therefore has to operate within the time constraints of an interactive webcam application.
Environment Compatibility
Computer-vision and scientific Python packages frequently contain native binaries and platform-specific dependencies.
This makes dependency versions, virtual environments, and runtime compatibility an important part of the development process.
Current Results
Metric	Result
Gesture Classes	5
Latest Recorded Validation Accuracy	97.79%
Classifier Input	Landmark-derived features
Inference	Real-time
Face Processing	MediaPipe
Hand Processing	MediaPipe
Model Persistence	Pickle


Future Work
Potential improvements include:
- Temporal gesture modeling
- Sliding-window feature sequences
- Gesture smoothing
- Confidence-based prediction thresholds
- User-specific calibration
- Dataset balancing
- Cross-user validation
- Robustness testing under different lighting conditions
- FPS profiling
- Inference-latency profiling
- Neural-network comparison
- Automatic meme ranking
- More gesture classes
- Larger and more diverse datasets
A particularly interesting direction is temporal modeling.
The current system primarily operates on a spatial representation:
Single Frame
    ↓
Landmarks
    ↓
Feature Vector
    ↓
Classifier

A future version could model gestures as sequences:
Frame Sequence
      ↓
Landmark Sequence
      ↓
Temporal Feature Extraction
      ↓
Sequence Model
      ↓
Gesture Prediction

This could allow the system to recognize motion-based gestures rather than relying primarily on a single-frame pose.
Project Motivation
MemeVision started as a simple experiment:
Can a webcam understand what reaction I'm making and automatically choose the meme for it?

The project evolved into a practical exploration of:
- Computer vision
- Landmark-based representation
- Feature engineering
- Supervised classification
- Dataset collection
- Dataset validation
- Model versioning
- Real-time inference
- Runtime performance
- ML pipeline reproducibility
The meme layer is the interface.
The underlying engineering problem is:
Can structured human pose and facial information be converted into a compact numerical representation that is reliable enough for real-time classification?

MemeVision is an exploration of that problem.
Author
17Zoras
GitHub: https://github.com/17Zoras
License
No explicit open-source license is currently provided.
```

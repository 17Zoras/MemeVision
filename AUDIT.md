# MemeVision Current Project Audit

Audit performed against the files currently present in the workspace on 2026-09-03.

## Architecture

MemeVision is a Python desktop application. `main.py` opens a webcam with OpenCV, creates a Tkinter window, captures and displays mirrored frames, and sends occasional frame copies to a daemon inference thread. The inference thread calls the shared MediaPipe feature extractor and a pickled scikit-learn classifier. Tkinter owns meme rendering and the UI event loop.

The project has three data paths:

1. `collect.py` records temporary webcam frames for a selected class and asks for confirmation before appending feature rows.
2. `database.py` stores confirmed rows in `data/training_data.csv` and retains only the newest rows after a per-class limit is exceeded.
3. `hf_import.py` attempts to convert a Hugging Face image dataset into the same numerical representation. `prepare_v6.py` merges local and external rows into `data/v6_training_data.csv`; `train_v6.py` trains the V6 model.

## Current Data Flow

```text
webcam -> MediaPipe face/hand landmarks -> 60 float features
                                      -> CSV local rolling data
                                      -> RandomForestClassifier
webcam -> same extractor -> model probabilities -> count-based history -> meme GIF/JPG/PNG
```

The collector and live application both import `extract_features` from `features.py`, so they currently share the implementation. The Hugging Face importer also uses that function.

## Current Feature Contract

The current feature vector has exactly 60 values:

- 8 face values: two eye distances, two mouth distances, two eyebrow distances, and two nose offsets.
- 26 values for each of two hand slots: palm x/y, five finger lengths, five fingertip-to-wrist distances, ten fingertip relationships, and four hand-to-face distances.
- Missing face or hand slots are zero-filled.

The values are numerical MediaPipe landmark-derived features. Most face and hand shape values are normalized by face width or palm size, but palm x/y and four hand-to-face values are not consistently normalized. There is no explicit feature dimension assertion in `features.py` or `collect.py`.

## Current Models And Training

Both `model.pkl` and `model_v6.pkl` are `RandomForestClassifier` instances with 60 input features and classes `other`, `shocked`, `thinking`, `thumbsdown`, and `thumbsup`. `main.py` prefers `model_v6.pkl` and falls back to `model.pkl`; both original model files remain present.

`train_v5.py` trains from the rolling local CSV and writes `model.pkl`. `train_v6.py` trains from the merged V6 CSV and writes `model_v6.pkl`. Both use a random stratified 80/20 frame split. That split can overestimate real-world performance because adjacent frames from the same recording can appear in both partitions. V6 currently reports 97.79% accuracy on 136 random held-out rows; this is not session-independent validation.

Measured current dataset counts:

| Dataset | other | shocked | thinking | thumbsdown | thumbsup | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `training_data.csv` | 130 | 132 | 182 | 116 | 117 | 677 |
| `v6_training_data.csv` | 130 | 132 | 182 | 116 | 117 | 677 |

There is currently no source, recording ID, timestamp, or session metadata in either CSV. The V6 merged file currently contains no external rows because the configured Hugging Face import has not succeeded.

## Current Inference And Confidence

The model runs asynchronously approximately every 0.10 seconds. `predict_proba` is used, then hand-written pose guards reduce some `shocked`, `thinking`, and thumb probabilities. A prediction is added to a 12-item history only when confidence is at least 0.72; lower confidence becomes `uncertain`. A label becomes stable when it appears at least 8 times in the history.

This is unweighted majority voting. It does not use the probability margin, exponential smoothing, candidate duration, release hysteresis, or per-class thresholds. The history is mutated by the worker while the Tkinter thread reads it and related state, which is a small thread-safety risk.

## Current Meme Behavior

Active mappings are `thinking.jpg`, `shocked.jpg`, `thumbsup.jpg`, and `thumbsdown.gif`. The other class intentionally has no meme. The folder also contains `blehh.jpg`, `clockit.jpg`, `cooked.jpg`, `happy.gif`, `plotting.jpg`, `sad.png`, `surprised.jpg`, and `suspicious.jpg`, but there is no training data or semantic class configuration for them, so they are correctly inactive for now.

JPG, PNG, and GIF files are loaded through Pillow. All GIF frames are decoded at startup and resized on display. A new meme resets the GIF to frame zero. The current trigger has a 1.5 second cooldown but no candidate duration or sustained release delay. The same active label is not restarted while it remains active.

## Hugging Face Integration

`hf_import.py` targets `s17660101713/hagrid-subset` and maps `like` to `thumbsup`, `dislike` to `thumbsdown`, and `fist`, `point`, and `no_gesture` to `other`. It does not provide valid evidence for the combined `thinking` or `shocked` classes, so local data must supply those classes.

The repository currently exposes a very large image manifest and its Xet-backed files fail with reconstruction 404 errors. Streaming metadata also did not expose the expected `ClassLabel` schema. The importer now catches these failures, but it does not yet use a smaller stable dataset or record provenance.

## Known Bugs And Risks

- `main.py`, `features.py`, and model loading execute at import time, making unit testing and missing-resource checks awkward.
- The application opens camera index 0 only; there is no retry or selectable camera path.
- An unavailable camera exits without a Tkinter diagnostic.
- The inference worker writes shared prediction state directly while Tkinter reads it.
- The UI updates every 10 ms even when camera or inference work is slower.
- MediaPipe performs face and hand detection for every submitted inference frame with no timing or FPS instrumentation.
- The camera frame is resized before display, so display resolution cannot be independent of the AI input.
- Feature vectors are not validated at extraction, collection, database insertion, or model prediction boundaries.
- The local CSV has no recording/session metadata and the collector stores every captured frame, including near-duplicates.
- Training cannot perform recording-aware validation because recording IDs are absent.
- Random Forest probabilities are treated as calibrated confidence without calibration or margin rejection.
- V5 uses `exit()` and relative output behavior inherited from its older script style.
- The Hugging Face import is supplementary only and currently unavailable for the configured repository.
- The runtime title says V5 even when V6 is selected.
- `other` has no visual output, which is appropriate as a rejection class but should be explicit in configuration.

## Performance Bottlenecks

The expensive path is MediaPipe face plus hand detection followed by feature extraction, followed by model prediction. The Tkinter thread also performs camera capture, OpenCV drawing, BGR-to-RGB conversion, PIL conversion, and meme frame creation. GIF frames are decoded once, but `ImageTk.PhotoImage` is recreated each display update. The one-item inference queue avoids a large backlog, but `put_nowait` drops frames without reporting latency or throughput.

## Recommended Improvements

1. Add a shared versioned feature schema with named constants and strict 60-value validation.
2. Preserve `training_data.csv`, add source/timestamp/recording metadata in a versioned dataset, and keep V5 and V6 models recoverable.
3. Add collector deduplication and recording IDs so whole sessions can be audited or removed.
4. Build a dataset statistics/check utility and reject malformed or non-finite rows.
5. Add semantic relational features only through a versioned extractor; retrain and compare against the existing 60-feature baseline rather than mixing schemas.
6. Use recording-aware validation when metadata exists and report macro F1, per-class recall, OTHER precision, and inference timing.
7. Replace count-only smoothing with confidence-weighted candidate/stable/release state and configurable per-class thresholds.
8. Move all camera and inference resource setup behind controlled startup functions and expose missing-resource errors clearly.
9. Add debug-mode timing/status output without cluttering normal mode.
10. Keep extra meme assets inactive until matching data and a measurable classifier class exist.
11. Replace the unstable Hugging Face source or allow a local cached export; never map generic expression labels to combined poses without evidence.

## Implementation Status After Audit

The following incremental changes were applied after the baseline audit:

- Added `feature_schema.py` with the shared 60-feature contract, semantic names, shape checks, and finite-value checks.
- Applied that validation to feature extraction, local storage, Hugging Face import, dataset merge, training, and runtime inference.
- Added near-duplicate rejection for consecutive collector samples while retaining the Y/N confirmation gate.
- Added `dataset_stats.py` for class, source, recording, and invalid-row reporting.
- Added `benchmark.py` for Random Forest and Extra Trees comparison.
- Added `smoke_checks.py` for model, dataset, media, GIF, and MediaPipe resource checks.
- Added configurable metadata for the four active, trained meme classes without activating unsupported extra meme assets.
- Replaced live count-only smoothing with confidence-weighted history, probability-margin rejection, candidate duration, and release hysteresis.
- Added F3 debug mode with top-three predictions, feature count, camera FPS, AI FPS, and AI processing time.
- Preserved `data/training_data.csv`, `model.pkl`, and the existing V6 artifacts.

Measured after implementation: both datasets still contain 677 valid rows; the V6 model remains a 60-feature Random Forest covering all five labels; random-split accuracy is 0.9779 with macro F1 0.9773; one live extraction measured approximately 37.2 ms on the current machine; and the GUI remained alive during a bounded launch check. The validation split is still frame-random rather than recording-aware because the legacy CSV has no recording IDs.

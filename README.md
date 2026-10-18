# ECG Rhythm Classification Using Machine Learning

A small-scale prototype that classifies ECG heartbeats as **Normal** or **Abnormal** with a Random Forest, and serves predictions through a FastAPI backend. Built as a combined AI/ML and Software Engineering course project.

> **Status:** working prototype on a subset of the MIT-BIH Arrhythmia Database. It is not a medical device and must not be used for clinical decisions.

---

## 1. Problem

Manual ECG review is slow, needs trained specialists, and does not scale with growing monitoring data. Rare but dangerous rhythm abnormalities are the easiest to miss. This project explores whether a machine learning model can classify heartbeats automatically from ECG features to support faster, more consistent screening.

## 2. Approach

- **AI approach:** supervised learning (labeled beats, classification).
- **Model:** Random Forest (100 trees, `class_weight="balanced"`, `random_state=42`).
- **Task:** binary classification, **Normal (0)** vs **Abnormal (1)**.
- **Backend:** FastAPI service that loads the saved model and returns a class with a confidence score.

## 3. Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet): https://physionet.org/content/mitdb/

- 48 half-hour, two-channel ambulatory ECG recordings from 47 subjects, sampled at 360 Hz.
- Beat-by-beat annotations by cardiologists; recordings collected 1975-1979, database released in 1980.
- Data is downloaded automatically by the `wfdb` library when `train.py` runs. No data files are stored in this repo.

### Label mapping (2 classes)

| Class | MIT-BIH beat symbols used |
|---|---|
| Normal (0) | `N`, `L`, `R`, `e`, `j` |
| Abnormal (1) | `A`, `a`, `J`, `S`, `V`, `E`, `F` |

Beats with other symbols (for example paced or unclassifiable beats) are skipped. This is a 2-class grouping of the AAMI EC57 categories (Normal vs. supraventricular, ventricular and fusion beats), not the full 5-class scheme.

## 4. Train/Test Split

The split is **by record (patient), not by beat**. Beats from one recording never appear in both sets, which avoids patient leakage that would inflate results.

| Set | Records | Beats | Normal | Abnormal |
|---|---|---|---|---|
| **Train** | 101, 106, 119, 201, 203, 208 | 13,754 | 10,647 | 3,107 |
| **Test** | 200, 210, 221 | 7,669 | 6,189 | 1,480 |

Per-record breakdown:

| Split | Record | Beats | Normal | Abnormal |
|---|---|---|---|---|
| Train | 101 | 1,860 | 1,857 | 3 |
| Train | 106 | 2,025 | 1,506 | 519 |
| Train | 119 | 1,984 | 1,540 | 444 |
| Train | 201 | 1,961 | 1,633 | 328 |
| Train | 203 | 2,974 | 2,527 | 447 |
| Train | 208 | 2,950 | 1,584 | 1,366 |
| Test | 200 | 2,598 | 1,741 | 857 |
| Test | 210 | 2,647 | 2,420 | 227 |
| Test | 221 | 2,424 | 2,028 | 396 |

Abnormal beats are about 22.6% of the training set and 19.3% of the test set. No separate validation set was used and no hyperparameter tuning was done in this version.

## 5. Preprocessing and Features

For every labeled beat (anchored on the annotated R-peak):

1. **Window:** 180 samples around the R-peak (90 before, 90 after; 0.5 s at 360 Hz), from the first ECG channel.
2. **Normalization:** each beat window is z-scored (zero mean, unit standard deviation).
3. **Features (34 total):**
   - 4 rhythm features: previous RR interval, next RR interval, and each divided by the local average RR interval
   - 30 waveform samples (every 6th sample of the normalized beat window)
4. **Class imbalance:** handled with `class_weight="balanced"` in the Random Forest.

Not implemented in this version: bandpass filtering and automated R-peak detection (the dataset's annotated R-peak positions are used instead).

## 6. Results

Evaluated on the held-out test records (7,669 beats from patients not seen in training). All numbers below come from `python train.py` and are saved in `metrics.json`.

### Overall

| Metric | Value |
|---|---|
| **Accuracy** | **0.952** |
| Macro avg precision / recall / F1 | 0.909 / 0.947 / 0.926 |
| Weighted avg precision / recall / F1 | 0.956 / 0.952 / 0.953 |

### Per class

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal | 0.985 | 0.955 | 0.970 | 6,189 |
| Abnormal | 0.833 | 0.940 | 0.883 | 1,480 |

### Confusion matrix (rows = true, columns = predicted)

| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **True Normal** | 5,910 | 279 |
| **True Abnormal** | 89 | 1,391 |

![Confusion matrix](confusion_matrix.png)

### How to read these results

- **Recall (Abnormal) = 0.940:** the model caught 1,391 of 1,480 abnormal beats and missed 89.
- **Precision (Abnormal) = 0.833:** when the model says Abnormal it is right 83.3% of the time; there were 279 false alarms on normal beats.
- The model favors catching abnormal beats at the cost of some false alarms, which suits a screening-assist setting.

### Metric definitions

| Metric | Definition |
|---|---|
| Accuracy | (TP + TN) / all beats |
| Precision | TP / (TP + FP): of beats predicted as a class, how many were correct |
| Recall | TP / (TP + FN): of beats truly in a class, how many were found |
| F1-score | 2 x Precision x Recall / (Precision + Recall) |

## 7. Limitations

- Small subset: 9 of the 48 records, with only 3 test records. Results may change with different records.
- Single train/test run, no cross-validation and no confidence intervals.
- Binary task only; the 5 AAMI classes are not yet separated.
- One model only; no comparison with SVM or a CNN yet.
- Uses annotated R-peak positions rather than an automatic detector, which makes the task easier than a fully automatic pipeline.
- MIT-BIH is an old, single-source dataset; generalization to other databases has not been tested.

## 8. Project Structure

```
.
├── train.py             # data loading, preprocessing, training, evaluation, saves artifacts
├── api.py               # FastAPI backend: /predict, /metrics, /sample
├── requirements.txt     # Python dependencies
├── model.pkl            # trained Random Forest (created by train.py)
├── metrics.json         # saved evaluation report (created by train.py)
├── confusion_matrix.png # confusion matrix image (created by train.py)
├── sample_beats.json    # example normal/abnormal test beats for the API demo
├── LICENSE
└── README.md
```

## 9. How to Run

Requires Python 3.9+ and an internet connection (to download MIT-BIH records).

```bash
pip install -r requirements.txt
python train.py          # trains and prints metrics; writes model.pkl, metrics.json, etc.
uvicorn api:app          # starts the backend
```

Then open http://127.0.0.1:8000/docs.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check; returns the expected feature count (34) |
| POST | `/predict` | Body: `{"features": [34 numbers]}`; returns `{"class": "Normal" or "Abnormal", "confidence": 0.xx}` |
| GET | `/metrics` | Returns the saved evaluation report |
| GET | `/sample` | Returns one real normal and one real abnormal test beat to paste into `/predict` |

## 10. Software Engineering Overview

- **Modules:** data loading and preprocessing, feature extraction, model training and evaluation, prediction API.
- **Interface:** the model is a replaceable module behind one function-like contract (34 features in, class and confidence out).
- **Validation:** the API rejects requests with the wrong number of features (HTTP 400).
- **Version control:** Git/GitHub, with changes committed in small steps.

## 11. Roadmap

Planned, not yet done:

- Use all usable MIT-BIH records and the 5-class AAMI scheme
- Add SVM and a 1D-CNN for comparison
- Bandpass filtering and automatic R-peak detection
- Validation split, hyperparameter tuning, and cross-validation
- Minority-class handling comparison (class weights vs. SMOTE) and per-class reporting
- Unit and integration tests, defect log
- Simple web front end
- Cross-dataset testing on other PhysioNet databases

## 12. References

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3):45-50, 2001.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23):e215-e220, 2000.
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/

## Disclaimer

This is an academic proof-of-concept. It is not a certified medical device and is not intended for diagnosis or treatment.

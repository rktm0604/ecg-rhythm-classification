# ECG Rhythm Classification — Corrected Phase 2

This project evaluates five-class AAMI heartbeat classification on MIT-BIH Arrhythmia Database records. The primary experiment is **offline/delayed**: reference annotations locate beats, RR features include future heartbeat intervals, and each waveform window contains 90 samples before and 110 samples after the annotated beat.

## Setup and commands

Use the existing virtual environment and locally available MIT-BIH files under `data/mitdb`. The corrected pipeline reads local records; it does not download them.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe artifacts\phase2_corrected\train_eval_phase2.py
.\.venv\Scripts\streamlit.exe run app.py
```

The Phase 2 training command writes only to `artifacts/phase2_corrected/`. Running it again will replace files in that corrected experiment directory. Existing legacy and earlier experiment artifacts are separate and remain available.

## Annotation mapping and features

- N: `N, L, R, e, j`
- S: `A, a, J, S`
- V: `V, E`
- F: `F`
- Q: `/, f, Q`

RR features use consecutive annotation positions whose symbols map to these heartbeat classes; unmapped/non-beat annotation positions are not interval endpoints. `rr_prev` and `rr_next` are the previous and following valid-heartbeat intervals. `rr_local` is the mean of up to two preceding and two following valid-heartbeat intervals. Beats lacking a previous or next valid heartbeat and beats whose 200-sample window crosses a signal boundary are excluded with explicit reasons in the extraction/audit logic.

## Experiment design

The existing record-wise DS1/DS2 split is retained; record 202 is excluded because it shares a subject with DS1 record 201. A deterministic 80/20 record-grouped split within DS1 provides the same fit and validation records for Random Forest, 1D-CNN, and the majority/RR-only baselines. RF candidate selection and CNN early stopping use DS1 validation only. DS2 is reserved for final reporting and is not used for feature selection, class-weight selection, thresholds, or model selection.

The RF uses four RR features plus 30 sampled waveform values. The CNN uses the complete 200-sample waveform. The majority baseline and RR-only Logistic Regression use the same fit/validation split.

Phase 4 is a separate secondary causal experiment. It predicts at the annotated R-peak using only the current/past waveform and past RR intervals. Run it with:

```powershell
.\.venv\Scripts\python.exe artifacts\phase4_causal\train_eval_phase4.py
```

Its configuration, predictions, metrics, feature timing, and comparison with Phase 2 are stored in `artifacts/phase4_causal/`. The causal and offline/delayed experiments have different information cut-offs and should not be interpreted as identical tasks.

## Corrected Phase 2 outputs

All outputs are stored together in `artifacts/phase2_corrected/`:

- `experiment_config.json` — labels, split, counts, protocol, model selection settings
- `results.json` — validation and final test metrics, reports, and confusion matrices
- `predictions.npz` — DS1 validation predictions and held-out DS2 predictions/probabilities
- `random_forest.joblib`, `rr_logistic_baseline.joblib`, `cnn.pt` — saved checkpoints
- `train_eval_phase2.py` — experiment runner

The five-class dashboard reads these Phase 2 files by default. Run it with `streamlit run app.py` from the project root.

## Results and limitations

Corrected extraction retained 98,511 beats: N 87,983; S 2,724; V 6,988; F 801; Q 15. DS2 contains 45,673 beats. RF DS2 macro-F1 is 0.3860 and accuracy is 0.9444; CNN macro-F1 is 0.2855 and accuracy is 0.7335. Macro-F1 and per-class results should be considered alongside accuracy because N dominates the dataset.

Minority class results are weak. DS1 fit contains only 31 F and 4 Q examples; DS2 has 387 F and 7 Q. RF has zero recall on F and Q in DS2, and CNN has zero recall on Q. These supports are too small for strong conclusions about rare classes; their individual supports and metrics are included in the results file and dashboard.

Additional limitations: MIT-BIH is not sampled to represent population prevalence; annotations provide beat locations, so this does not evaluate automatic beat detection; the offline/delayed design uses future RR intervals and post-peak samples and is not causal at the R peak. This research prototype is not a medical device and must not be used for diagnosis or treatment.

## Legacy binary application

`train.py`, root-level `model.pkl`, `metrics.json`, `sample_beats.json`, `confusion_matrix.png`, `api.py`, `ui.py`, and `static/` are preserved as the **legacy two-class Normal/Abnormal experiment**. They do not use the corrected five-class Phase 2 model or metrics. The five-class app is `app.py`.

## Data citation

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50. MIT-BIH Arrhythmia Database, PhysioNet, DOI: 10.13026/C2F305.

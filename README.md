# ECG Rhythm Classification

This repository contains a corrected, five-class AAMI heartbeat-classification research project using the **MIT-BIH Arrhythmia Database**. It includes a primary offline/delayed experiment (Phase 2), a secondary strictly causal experiment (Phase 4), saved model outputs, automated tests, and academic documentation. A Streamlit results dashboard is present; UI and live-demo ownership is separate from this ML project work.

This is a research prototype, not a medical device. It does not detect beats automatically: reference annotations provide the beat locations and labels.

## Repository map

```text
pipeline.py                         Shared annotation handling and offline/delayed extraction
dataset_audit.py                    Per-record eligibility/count audit using the shared extractor
app.py                              Five-class Streamlit dashboard
tests/test_pipeline_rr.py           RR, annotation, boundary and audit parity tests
tests/test_phase2_results.py        Phase 2 metrics/checkpoint/split consistency tests
tests/test_phase2_rf_improved.py    DS1 candidate and final RF metric verification
tests/test_phase4_results.py        Phase 4 metrics/checkpoint/comparison consistency tests
tests/test_causality.py             Causal feature perturbation tests
scripts/phase2_eda.py               DS1-only descriptive EDA; does not fit or tune models
scripts/phase2_error_analysis.py    Post-hoc DS2 analysis from frozen predictions; no training
docs/phase2_final_report.md          Final report with the requested 12 sections
docs/phase2_error_analysis.md        Detailed DS2 error and record-wise analysis
docs/phase2_rf_improvement.md        DS1-only RF investigation and final DS2 comparison
docs/phase2_compliance_audit.md      Requirement-level audit and remaining gaps
docs/phase2_presentation_outline.md  Presentation outline and speaker prompts
docs/ml_final_handover.md             Model input/output and artifact handover for UI teammate
artifacts/phase2_corrected/          Primary offline/delayed experiment and saved outputs
artifacts/phase2_rf_improved/       Separate DS1 RF investigation and one-shot DS2 evaluation
artifacts/phase4_causal/             Secondary causal experiment and saved outputs
```

The Phase 2 and Phase 4 directories each contain their training script, experiment configuration, predictions, metrics and RF/CNN/baseline checkpoints. Phase 2's reproducible error-analysis tables and figures are in `artifacts/phase2_corrected/error_analysis/`; see [the analysis report](docs/phase2_error_analysis.md). Phase 4 also contains `feature_timing.md` and `comparison_report.md`. The causal test lives in `tests/` so the documented test command discovers it. The [ML handover](docs/ml_final_handover.md) records checkpoint paths, feature order, tensor shapes, timing, and output interpretation.

## AAMI annotation mapping

The shared mapping in `pipeline.py` groups beat symbols into these five labels:

| Class | Annotation symbols | Meaning (summary) |
|---|---|---|
| N | `N`, `L`, `R`, `e`, `j` | Normal and bundle-branch beats |
| S | `A`, `a`, `J`, `S` | Supraventricular ectopic beats |
| V | `V`, `E` | Ventricular ectopic beats |
| F | `F` | Fusion of ventricular and normal beat |
| Q | `/`, `f`, `Q` | Paced, fusion-paced, or unclassifiable beats |

Only annotations in this mapping count as heartbeat positions for interval endpoints. Unmapped rhythm and other non-beat annotations remain in the annotation stream but cannot be used as RR endpoints.

## Data preparation and corrected extraction

MIT-BIH data is **not included** in this repository. Obtain it separately from the [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) and place the extracted files under `data/mitdb/`, beside `pipeline.py`. Each record needs the WFDB header, signal and annotation files (normally `.hea`, `.dat`, and `.atr`). The pipeline reads local files and does not download the database. `.gitignore` excludes `data/mitdb/` so raw recordings are not committed.

`pipeline.py` selects the MLII channel when its header label is available (otherwise it reports a warning and uses channel 0). Phase 2 extracts a 200-sample window from 90 samples before through 109 samples after the annotated R-peak. It excludes beats without both a previous and a following valid heartbeat, beats with an out-of-record or non-finite waveform, and beats with invalid RR intervals. The RR values are measured in samples at 360 Hz.

- `rr_prev`: interval from the previous valid heartbeat to the current beat.
- `rr_next`: interval from the current beat to the next valid heartbeat.
- `rr_local`: mean of up to two preceding and two following intervals in the valid-heartbeat sequence. This includes future intervals.

`dataset_audit.py` calls the same `extract_record_beats` eligibility function as model-data extraction. It can write per-record, per-class counts and exclusion reasons to a chosen output directory:

```powershell
python dataset_audit.py --out artifacts/phase1_corrected
```

Run the audit only when the local MIT-BIH data is available. The output directory is created if needed; choose a new directory if you need to preserve an existing audit report.

## Splits and model methodology

The record-wise split is stored with each experiment. DS1 contains 23 development records; record 202 is excluded because it shares a subject with DS1 record 201, and paced records 102, 104, 107, and 217 are excluded. Record 121 is included in this project's allocation, which is not claimed to be an exact standard benchmark split. A deterministic, record-grouped 80/20 split within DS1 produces 18 fit records and 5 validation records. **Both phases use the same records**:

- DS1 fit: `106, 108, 109, 112, 114, 115, 116, 121, 122, 124, 201, 203, 207, 209, 215, 220, 223, 230`
- DS1 validation: `101, 118, 119, 205, 208`
- DS2 final test: `100, 103, 105, 111, 113, 117, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234`

RF candidates and CNN early stopping are selected using DS1 validation only. Class weights are derived from DS1 fit data or are fixed candidate values. DS2 is reserved for final evaluation and is not used for model selection, feature selection, class-weight selection, or thresholds. The majority and RR-only Logistic Regression baselines use the same fit/validation records. The RF consumes engineered RR values and 30 evenly spaced waveform values; the 1D-CNN consumes the complete 200-sample waveform. They share the same examples and partitions, while their input representations differ.

The **Random Forest is presented as the primary selected model** because it had the higher DS1 validation macro-F1 (0.3992 vs 0.3717 for the CNN). The 1D-CNN is retained as an additional architecture comparison. The RF's bounded validation search uses 300 trees, `max_depth` in `{None, 24}`, `min_samples_leaf` in `{1, 3}`, and three class-weight choices. The CNN uses three Conv1D blocks (32/64/128 channels; kernel sizes 9/7/5), batch normalization, ReLU, max pooling after the first two blocks, adaptive average pooling, dropout 0.35, and a five-class linear output. Training uses AdamW (lr 0.001, weight decay 0.0001), weighted cross-entropy, batch size 256, up to 35 epochs, and patience 7 on DS1 validation macro-F1. The saved best epoch is 18. Full details and limitations are in [the final report](docs/phase2_final_report.md).

### DS1-only Random Forest investigation

The S-class investigation and controlled RF comparison are documented in [the RF improvement report](docs/phase2_rf_improvement.md). It compares the existing model with weight, depth/leaf and RR-only/waveform-only alternatives using only the existing DS1 fit and validation records. No candidate met the prespecified improvement rule, so the current RF was retained. The selected checkpoint and final DS2 comparison are saved separately under `artifacts/phase2_rf_improved/`; Phase 2, CNN and Phase 4 artifacts remain unchanged. Record 232 was not used for selection.

To reproduce the experiment in a **new empty** output folder after placing MIT-BIH locally:

```powershell
$studyOut = "artifacts\phase2_rf_improved_reproduction"
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\train_rf_improved.py --data-dir data\mitdb --out $studyOut
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\evaluate_rf_improved_ds2.py --data-dir data\mitdb --out $studyOut
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\verify_rf_improved_ds2.py --out $studyOut
```

The DS1 script never loads DS2 data. The evaluator runs once after the frozen DS1 configuration exists and refuses to overwrite outputs; the verification script recomputes metrics from saved predictions without model inference.

## Reports and assessment materials

The [final report](docs/phase2_final_report.md) presents Phase 2 as the primary experiment and Phase 4 as a separate causal experiment. The [compliance audit](docs/phase2_compliance_audit.md) maps requirements to repository evidence and records what remains unverified. The [RF investigation](docs/phase2_rf_improvement.md) documents the DS1-only search and why the existing RF was retained. The [presentation outline](docs/phase2_presentation_outline.md) contains a 12-slide plan with speaker prompts. An actual PowerPoint deck is not included: COM activation returned access denied, UI automation could not activate the PowerPoint window, and no PowerPoint authoring library was available. No placeholder `.pptx` was created.

The supplied rubric screenshots cover Phase 2 Sections 21–26. Section 27 and the Phase 1 approval/success-criterion wording are not shown; the [compliance audit](docs/phase2_compliance_audit.md) maps the visible ML criteria and states what remains unverifiable. Presentation, UI, and live demo work is deferred for now per the project owner's direction.

### EDA and assessment materials

Run the DS1-only exploratory analysis when MIT-BIH is available locally:

```powershell
python scripts/phase2_eda.py --data-dir data/mitdb
```

It saves five plots and an `eda_summary.json` under `artifacts/phase2_corrected/eda/`. It loads DS1 only and does not change the saved experiment results. The [compliance audit](docs/phase2_compliance_audit.md) records requirement status and unresolved gaps. The [presentation outline](docs/phase2_presentation_outline.md) is speaker-ready content, not a rendered slide deck.

### Phase 2 — primary offline/delayed

Phase 2 uses `rr_prev`, `rr_next`, ratios to `rr_local`, and the 90-before/110-after waveform. Because it uses following-beat timing, future RR intervals, and post-R samples, the prediction is delayed relative to the annotated R-peak and is **not causal at R**.

Stored results in `artifacts/phase2_corrected/results.json` (macro-F1 / accuracy):

| Model | DS1 validation | DS2 final test |
|---|---:|---:|
| Random Forest | 0.3992 / 0.9558 | 0.3860 / 0.9444 |
| 1D-CNN | 0.3717 / 0.7516 | 0.2855 / 0.7335 |
| Majority baseline | 0.1812 / 0.8283 | 0.1875 / 0.8824 |
| RR-only Logistic Regression | 0.2739 / 0.5017 | 0.2588 / 0.4670 |

Phase 2 DS2 per-class metrics (precision / recall / F1):

| Class | Support | RF P / R / F1 | CNN P / R / F1 |
|---|---:|---:|---:|
| N | 40,300 | 0.9587 / 0.9961 / 0.9770 | 0.9506 / 0.7626 / 0.8463 |
| S | 1,780 | 0.5508 / 0.0579 / 0.1047 | 0.0398 / 0.0539 / 0.0458 |
| V | 3,199 | 0.7996 / 0.9028 / 0.8480 | 0.3838 / 0.8274 / 0.5244 |
| F | 387 | 0 / 0 / 0 | 0.0060 / 0.0620 / 0.0110 |
| Q | 7 | 0 / 0 / 0 | 0 / 0 / 0 |

The 98,511 corrected Phase 2 beats are N=87,983, S=2,724, V=6,988, F=801 and Q=15 across DS1 and DS2. The validation and final-test confusion matrices and full classification reports are saved in `results.json`.

### Phase 4 — secondary causal

Phase 4 predicts at the annotated R-peak. Its 200-sample waveform is `signal[R-199:R+1]`, including R but no later sample. `rr_prev` uses the previous valid heartbeat; causal `rr_local` is the mean of up to four preceding intervals ending at the current beat. There is no `rr_next`. The first valid beat is excluded when no previous heartbeat is available. See `artifacts/phase4_causal/feature_timing.md` for exact timing and exclusions.

Stored DS2 results (macro-F1 / accuracy):

| Model | Phase 2 offline/delayed | Phase 4 causal | Causal minus Phase 2 macro-F1 |
|---|---:|---:|---:|
| Random Forest | 0.3860 / 0.9444 | 0.3784 / 0.9326 | -0.0075 |
| 1D-CNN | 0.2855 / 0.7335 | 0.3460 / 0.8088 | +0.0604 |
| Majority baseline | 0.1875 / 0.8824 | 0.1875 / 0.8824 | +0.0000 |
| RR-only Logistic Regression | 0.2588 / 0.4670 | 0.1638 / 0.2148 | -0.0950 |

These are not identical tasks: the models receive different information before prediction, and the causal extraction retains some beats that Phase 2 excludes. Phase 4 DS2 has 40,319 N, 1,781 S, 3,199 V, 387 F and 7 Q beats. The RF has zero DS2 recall for F and Q; the CNN has near-zero F recall and zero Q recall. Only 31 F and 4 Q examples are in DS1 fit, so rare-class estimates are highly unstable and should not be treated as reliable population performance. Full class-by-class deltas are in `artifacts/phase4_causal/comparison_report.md` and are shown in the dashboard.

## Installation and commands

Python 3.11 is the tested interpreter. The direct dependencies are pinned in `requirements.txt`; the checked project environment reported versions matching those pins. From the repository root in PowerShell, create a fresh environment and install those dependencies, or use an already prepared environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit.exe run app.py
```

The test suite uses synthetic data and saved artifacts; it does not require MIT-BIH raw data. Dataset-backed runs expect WFDB `.hea`, `.dat`, and `.atr` files for each record under `data/mitdb/`. No raw ECG files are tracked.

To reproduce the Phase 2 error analysis from existing DS2 predictions without retraining (and regenerate the representative waveform figure), run:

```powershell
.\.venv\Scripts\python.exe scripts\phase2_error_analysis.py --data-dir data\mitdb
```

The script recomputes aggregate and record-wise metrics from `artifacts/phase2_corrected/predictions.npz` and verifies them against the saved `results.json`. Raw MIT-BIH files are needed only for the representative waveform figure. Outputs go to `artifacts/phase2_corrected/error_analysis/`; model artifacts and predictions are read-only inputs. To preserve existing analysis outputs, select a new output directory with `--out`.

To run the training scripts again when the dataset is present:

```powershell
.\.venv\Scripts\python.exe artifacts\phase2_corrected\train_eval_phase2.py
.\.venv\Scripts\python.exe artifacts\phase4_causal\train_eval_phase4.py
```

Training is not needed to reproduce reported metrics and the scripts write into their existing artifact directories. Running them will replace files there. Use a copy or a separate checkout and a separate output-aware script before any retraining; do not use DS2 for model selection.

Each training script writes checkpoints, predictions, configuration and results into its own artifact directory. **Running a training script overwrites that phase's existing output files.** Training is optional and is not needed to reproduce the documented metric tables or error analysis. Do not use DS2 to choose model settings.

## Dashboard and saved artifacts

The existing Streamlit dashboard loads Phase 2 predictions and metrics from `artifacts/phase2_corrected/` and loads the Phase 4 DS2 comparison from `artifacts/phase4_causal/`. It presents the five labels in `N, S, V, F, Q` order and does not use legacy binary artifacts. The UI/live-demo workflow is maintained separately; see the [ML handover](docs/ml_final_handover.md) for model contracts and limitations.

`experiment_config.json` records the data split, extraction design, selected RF settings and CNN epoch/weights. `predictions.npz` stores validation and DS2 labels, predictions and scores. `results.json` stores per-class reports and confusion matrices. Checkpoint files are `random_forest.joblib`, `rr_logistic_baseline.joblib`, and `cnn.pt` in each experiment directory. The RF `predict_proba` values are vote fractions, not calibrated clinical probabilities.

## Limitations and citation

MIT-BIH has substantial class imbalance and does not represent population prevalence. In particular, the very small F and Q fit/test supports make those estimates unreliable. Annotated beat locations are assumed, so automatic beat detection is outside this experiment. Phase 2 uses future information; Phase 4 is causal only with respect to the annotated R-peak and still assumes that R-peak annotation is available at prediction time. Neither experiment is intended for diagnosis or treatment.

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50. MIT-BIH Arrhythmia Database, PhysioNet, DOI: 10.13026/C2F305.


# Phase 2 Compliance Audit

## Assessment basis and scope

This audit checks the user's supplied Phase 2 rubric screenshots (visible Sections 21–26, including the 24.1 assessment table) against source code, saved Phase 2 artifacts, tests, and documentation. The screenshots do not include Section 27 or the Phase 1 approval/success-criterion wording referenced by the Phase 2 rubric. Conclusions below cover only the visible ML criteria. Per the user's latest instruction, presentation quality, UI, and live demonstration are out of scope for this pass; those rubric items are explicitly deferred rather than assessed.

The corrected checkout is `D:\Downloads\ecg project\cleanup-checkout`. Historical Phase 1 baseline evidence was read only from the original local project's `D:\Downloads\ecg project\artifacts\results.json`, `test_data.npz`, `rr_logistic_baseline.joblib`, and `train_eval.py`. Those historical artifacts are not tracked in the cleaned checkout.

Status labels: **IMPLEMENTED AND VERIFIED**, **PARTIALLY IMPLEMENTED**, **NOT IMPLEMENTED**, and **CANNOT VERIFY**.

## A. Requirement-by-requirement findings

### 1. Data understanding and preparation

| Requirement | Status | Evidence and finding |
|---|---|---|
| Dataset description and source | IMPLEMENTED AND VERIFIED | `docs/phase2_final_report.md` §2/§12 and `README.md` identify MIT-BIH/PhysioNet. Local `data/mitdb` contains 48 headers. Raw data remains outside the checkout and is not committed. |
| Data loading and annotation handling | IMPLEMENTED AND VERIFIED | `pipeline.py::_read_record` checks local `.hea/.dat/.atr`, reads WFDB record and annotations, chooses MLII where possible, and returns samples/symbols. `dataset_audit.py::audit_record` uses the shared extractor. |
| AAMI five-class mapping | IMPLEMENTED AND VERIFIED | `pipeline.py::LABEL_MAP` maps N=`N,L,R,e,j`; S=`A,a,J,S`; V=`V,E`; F=`F`; Q=`/,f,Q`. Tests check the approved mappings. |
| Missing, invalid, and inconsistent data | PARTIALLY IMPLEMENTED | Missing files raise explicitly; `extract_record_beats` excludes incomplete neighbor context, out-of-bounds windows, non-finite waveforms, and nonpositive/nonfinite RR. Tests cover these. It does not perform general signal repair/imputation or validate ordering of all annotations as a separate input check. |
| Signal preprocessing and segmentation | IMPLEMENTED AND VERIFIED | `pipeline.py::extract_record_beats` slices 200 samples, `pipeline.py::_safe_zscore` normalizes each window, and the EDA waveform plot confirms R position and window extent. There is no filtering or resampling. |
| RR interval extraction | IMPLEMENTED AND VERIFIED | `extract_record_beats` filters endpoints through `LABEL_MAP`; uses adjacent valid heartbeat positions for `rr_prev/rr_next` and up to two preceding and two following intervals for `rr_local`. Audit/extraction parity and non-beat tests pass. |
| Explanation and justification of preprocessing | PARTIALLY IMPLEMENTED | Added exact methods and limitations in `docs/phase2_final_report.md` §3. The code does not document a tested reason for per-window z-scoring; no ablation measures the effect of normalization or lack of filtering. |
| Class imbalance analysis | IMPLEMENTED AND VERIFIED | DS1 and DS2 counts in `experiment_config.json`; new DS1 log-scale plot and analysis in `artifacts/phase2_corrected/eda/`; rare-class supports/metrics in `results.json` and report. |
| Train/validation/test splitting | IMPLEMENTED AND VERIFIED | `pipeline.py::check_record_split` plus Phase 2 `main` use record-wise DS1/DS2, then `GroupShuffleSplit` by DS1 record. Exact records/counts are in `experiment_config.json`; tests check disjointness and saved prediction records. |
| Leakage prevention | IMPLEMENTED AND VERIFIED | `check_record_split` rejects record overlap and known subject overlap and excludes record 202 because it shares a subject with 201. Training script does not pass record IDs or metadata to model inputs. |
| Approved information cut-off | IMPLEMENTED AND VERIFIED | Phase 2 is the approved offline/delayed task: `rr_next`, following intervals in `rr_local`, and 110 samples from R onward are used. This is accurately disclosed in `pipeline.py`, experiment config, README, dashboard, and report. It must not be presented as causal-at-R. |
| Excluded variables do not enter the model | IMPLEMENTED AND VERIFIED | `build_dataset` returns feature arrays separately from `record` and `beat_number`; training uses only `X_rf` or `X_wave`. Annotation index, record ID, sample, and raw symbol are not model features. |

### 2. Exploratory data analysis

| Requirement | Status | Evidence and finding |
|---|---|---|
| Five-class distribution visualisation | IMPLEMENTED AND VERIFIED | `scripts/phase2_eda.py`; `artifacts/phase2_corrected/eda/class_distribution_ds1.png` uses DS1 only, log counts and labels. |
| ECG waveform examples | IMPLEMENTED AND VERIFIED | `representative_waveforms_ds1.png` contains one normalized annotated example for each class with the R location marked. The report warns these single examples do not establish class-wide separability. |
| Relevant feature distributions | IMPLEMENTED AND VERIFIED | `rr_distributions_ds1.png` shows previous, following, and local RR distributions by class. |
| Relationships/patterns | IMPLEMENTED AND VERIFIED | `rr_relationship_ds1.png` plots previous versus following RR, with sample counts and explicit 99.5th-percentile display clipping. |
| Visualisations support the analysis | IMPLEMENTED AND VERIFIED | `docs/phase2_final_report.md` §4 embeds/links the generated DS1 figures and describes their limits. |
| Written interpretation | IMPLEMENTED AND VERIFIED | Report §4 interprets class prevalence, RR medians/overlap, representative traces, and model feature importance. |

### 3. Model implementation

| Requirement | Status | Evidence and finding |
|---|---|---|
| Identify a primary model | IMPLEMENTED AND VERIFIED | `docs/phase2_final_report.md` §1 and README now identify RF as primary based on higher DS1 validation macro-F1; CNN is secondary comparison. |
| Match the rubric's single-model requirement (22.3) | PARTIALLY IMPLEMENTED | The supplied rubric explicitly asks for a single ML model. RF is the selected primary model, but the approved project also retains CNN as a comparison. The CNN remains by the user's explicit instruction; this is a known rubric mismatch, not a hidden assumption. |
| Verify RF implementation | IMPLEMENTED AND VERIFIED | `artifacts/phase2_corrected/train_eval_phase2.py::main` fits `RandomForestClassifier`; saved checkpoint and parameters are checked by `tests/test_phase2_results.py`. |
| Inputs and output classes | IMPLEMENTED AND VERIFIED | RF input width 34 and CNN waveform width 200 are tested; class order is N,S,V,F,Q. Details in report §5. |
| Architecture/algorithm configuration | IMPLEMENTED AND VERIFIED | RF checkpoint plus experiment config; CNN `ECGNet` implementation and saved checkpoint checked by tests. |
| Training procedure | IMPLEMENTED AND VERIFIED | Phase 2 `main` fits RF/CNN/baselines on shared fit indices; `fit_cnn` early-stops on DS1 validation; config stores split, best epoch, and fit class weights. |
| Hyperparameters | IMPLEMENTED AND VERIFIED | RF and CNN settings are present in `train_eval_phase2.py` and config; documented in report §5–6. |
| Important hyperparameters tuned | PARTIALLY IMPLEMENTED | RF depth, leaf size, and class-weight candidates are compared on DS1 validation; CNN uses a fixed architecture/optimizer/rate and early stopping but no broad hyperparameter search. |
| Tuning decisions justified | PARTIALLY IMPLEMENTED | The report explains the bounded search and macro-F1 selection. It correctly notes that the chosen ranges were not broadly justified or proven optimal; no DS2 tuning occurred. |

### 4. Evaluation

| Requirement | Status | Evidence and finding |
|---|---|---|
| Primary metric | IMPLEMENTED AND VERIFIED | Macro-F1 is used for RF selection and CNN early stopping, and reported on validation/test. |
| Secondary metrics | IMPLEMENTED AND VERIFIED | Accuracy and full classification reports are saved. |
| Per-class precision/recall/F1 | IMPLEMENTED AND VERIFIED | `results.json` includes all five classes; `tests/test_phase2_results.py` recomputes and compares them. |
| Confusion matrices | IMPLEMENTED AND VERIFIED | `results.json` stores ordered validation/test matrices; tests recompute final DS2 matrices. |
| Phase 1 baselines under same evaluation protocol | PARTIALLY IMPLEMENTED | The historical Phase 1 baselines are majority and RR-only Logistic Regression in the original local `train_eval.py`/`artifacts/results.json`. Phase 2 also evaluates those baseline types on its own shared corrected split. Historical and corrected extraction retain 45,687 vs 45,673 DS2 beats, so historical scores are not exactly beat-paired. |
| Improvement margin over each approved Phase 1 baseline | PARTIALLY IMPLEMENTED | Using historical saved scores, Phase 2 RF exceeds majority by +0.198456 and RR-only Logistic Regression by +0.109924 macro-F1. The historical output records a 0.05 margin. Both pass arithmetically, but exact official threshold wording is unavailable and beat supports differ by 14. The matched current Phase 2 margins are +0.198458 and +0.127129. |
| Same held-out DS2 records | IMPLEMENTED AND VERIFIED | Phase 1 and Phase 2 configs use the same 20 DS2 record IDs. The corrected Phase 2 tests verify model partitions and test prediction record IDs. Exact eligible beats differ by 14 historically. |
| Reproduce reported values from saved predictions | IMPLEMENTED AND VERIFIED | Phase 2 artifact tests recompute validation/test metrics and matrices from `predictions.npz`. The original local Phase 1 baseline logistic model and `test_data.npz` also independently reproduce historical scores, but those reference artifacts are not in the cleaned checkout. |

### 5. Critical insights and interpretation

| Requirement | Status | Evidence and finding |
|---|---|---|
| Explain selected-model performance | IMPLEMENTED AND VERIFIED | Report §9 interprets RF/CNN validation and DS2 metrics and confusion counts. |
| Explain class imbalance | IMPLEMENTED AND VERIFIED | DS1/DS2 counts, prevalence plot, and N-dominated accuracy interpretation are documented. |
| Analyse minority-class performance | IMPLEMENTED AND VERIFIED | Per-class tables show RF F/Q recall zero, CNN Q recall zero, fit supports F=31/Q=4. |
| Identify important features/patterns | IMPLEMENTED AND VERIFIED | EDA saves impurity-based importance for the saved RF and RR class distributions. No causal interpretation is claimed. |
| Analyse incorrect predictions | PARTIALLY IMPLEMENTED | `docs/phase2_error_analysis.md` and `scripts/phase2_error_analysis.py` add saved-prediction confusion/error counts, one-versus-rest metrics, per-record metrics, and eight label-verified waveform examples. There is still no signal-quality or systematic blinded waveform review, so causes are not established. |
| Discuss preprocessing effect | PARTIALLY IMPLEMENTED | Report documents z-scoring and notes amplitude information may be lost; no ablation establishes effect. |
| Discuss hyperparameter choices | PARTIALLY IMPLEMENTED | Search and validation results are described, but no extensive sensitivity study supports the ranges or fixed CNN settings. |
| Identify unexpected results | IMPLEMENTED AND VERIFIED | Report describes CNN's lower DS2 macro-F1 than validation and the divergence between accuracy and macro-F1. |
| Relate results to Phase 1 | PARTIALLY IMPLEMENTED | Historical RF/CNN and baseline values are compared with exact support and extraction caveats; archival files are outside checkout. |

### 6. Conclusion

| Requirement | Status | Evidence and finding |
|---|---|---|
| State whether task addressed | IMPLEMENTED AND VERIFIED | Report §11 states that the known-annotated-beat, offline/delayed five-class task was addressed and explicitly excludes beat detection/clinical claims. |
| Verify Phase 1 success criterion referenced in the Phase 2 rubric | PARTIALLY IMPLEMENTED | The original local Phase 1 result records a 0.05 macro-F1 margin and `success=true`; Phase 2 RF exceeds both historical saved Phase 1 baseline macro-F1 values by more than 0.05. The Phase 1 approval wording defining the criterion was not supplied, and historical Phase 1 vs corrected Phase 2 eligible DS2 counts differ by 14, so exact criterion compliance and exact beat-paired comparison cannot be certified from the available evidence. |
| Compare final result with actual Phase 1 baselines | PARTIALLY IMPLEMENTED | Report §8 compares against the actual locally saved Phase 1 baselines and separately gives corrected matched Phase 2 baselines; historical beat count differs and reference artifacts are not in checkout. |
| Discuss limitations | IMPLEMENTED AND VERIFIED | Report §10 covers class support, information cut-off, preprocessing, model fairness, historical comparison, and dataset access. |
| Suggest future improvements | IMPLEMENTED AND VERIFIED | Report §11 recommends additional F/Q data, DS1-only ablations, stratified error analysis, and preserving historical baselines. |

### 7. Deliverables

| Deliverable | Status | Evidence and finding |
|---|---|---|
| Final report with all 12 sections | IMPLEMENTED AND VERIFIED | `docs/phase2_final_report.md` has the requested 12 numbered sections. |
| Organized executable source and tests | IMPLEMENTED AND VERIFIED | Root pipeline, training scripts, `scripts/phase2_eda.py`, tests, and artifact directories are organized. Test suite and compilation checks are run below. |
| Dataset source/access instructions | IMPLEMENTED AND VERIFIED | README and report §2/§12 give PhysioNet source and local file placement; raw data is not committed. |
| Presentation, UI, live demo and presentation-quality rubric items | DEFERRED BY USER | The user asked to leave presentation, UI and live-demo work for now. These items are outside this ML-only audit update and are not counted as ML evidence. |

## B. Missing or partially implemented items

1. The supplied screenshots cover Sections 21–26 but omit Section 27 and the Phase 1 approval/success-criterion wording. Those unseen requirements cannot be certified.
2. Historical Phase 1 baseline output is stored only in the original local project, not in the cleaned checkout. The 14-beat DS2 support difference also prevents exact per-beat historical pairing.
3. The repository includes two algorithms even though the brief reportedly asks for one; RF is designated primary and CNN an additional comparison, but a literal single-implementation rule remains a scope mismatch.
4. No ablation isolates effects of per-beat normalization, signal filtering, window size, RR features, or fixed CNN settings.
5. The error analysis now includes record metrics and representative saved ECG windows, but it is not a blinded clinical review or a signal-quality/subject-stratified causal analysis.
6. Presentation quality, UI, and live-demo deliverables are deferred at the user's direction and were not assessed in this ML-only pass.

## C. Changes made

- Added `scripts/phase2_eda.py`, which loads DS1 only and writes five EDA figures plus `eda_summary.json` under `artifacts/phase2_corrected/eda/`.
- Added `docs/phase2_final_report.md` with all 12 requested sections, saved Phase 2 metrics, historical/current baseline comparisons, error counts, EDA interpretation, limitations, and references.
- Added this audit, `docs/phase2_presentation_outline.md`, and `docs/phase2_error_analysis.md`.
- Added `scripts/phase2_error_analysis.py` and reproducible DS2 output tables/figures. The script checks aggregate predictions-derived metrics against saved results and verifies each plotted window's true label against shared extraction.
- Updated README with EDA/error-analysis commands, report links, RF primary-model designation, and reproduction instructions that do not retrain.
- No saved Phase 2 or Phase 4 experiment output was overwritten; no model was retrained; no package was installed; no GitHub branch was changed.

## D. Verification

Verification performed with the existing environment at `D:\Downloads\ecg project\.venv\Scripts\python.exe`:

- `python -m unittest discover -s tests -v`: **29 tests passed** in the current final ML verification.
- `python -m compileall -q pipeline.py dataset_audit.py app.py artifacts\phase2_corrected\train_eval_phase2.py artifacts\phase4_causal\train_eval_phase4.py scripts tests`: **passed**.
- Streamlit `AppTest.from_file('app.py').run()`: **0 exceptions**; Streamlit emitted only its expected bare-mode `ScriptRunContext` warning.
- `git diff --check`: **passed**.
- EDA was generated successfully from the local MIT-BIH data using DS1 only; five figures were visually inspected. The DS1 count summary is N=47,683, S=944, V=3,789, F=414, Q=8 (52,838 total).
- `scripts/phase2_error_analysis.py --data-dir <local MIT-BIH directory>` completed from saved DS2 predictions and ECG data; its assertions confirm exact agreement with `results.json` and label/record/beat alignment for all eight plotted windows. Confusion and recordwise figures were visually inspected.
- Phase 1 baseline scores were independently recalculated from the original local baseline checkpoint/test data and match the saved baseline results.

No model training or package installation occurred. Phase 2/4 model checkpoints, predictions, metrics, and configurations were not changed.

## E. Remaining submission checks

- Obtain the complete official Section 27 and the Phase 1 approval/success-criterion wording if exact certification is required.
- Decide whether the grader accepts a primary RF plus a comparative CNN, or requires only one implemented model.
- Include or otherwise make accessible the historical Phase 1 baseline reference outputs if the instructor expects fresh-clone reproduction.
- Turn the outline into a rendered deck if a slide deck is a required artifact.
- Do not call the experiments real-time or clinical; Phase 2 uses future information and reference annotations supply beat locations.

## F. Final ML documentation and verification update

This final pass checked the current clean branch and existing saved experiments. It did not train or tune a model, change predictions, or modify Phase 2/Phase 4 experiment artifacts.

### Additional documentation

- The Phase 2 final report retains its 12 numbered sections and now includes an abstract and explicit project objectives. Its content covers dataset/mapping, exclusions and record split, extraction/features, DS1 EDA, models/training, baselines, validation/model selection, Phase 2 and Phase 4 metrics, error analysis, discussion, limitations and references.
- `docs/ml_final_handover.md` documents model paths, ordered feature vectors, tensor shapes, label order, timing, output interpretation and constraints for the teammate's separately maintained UI/live demo.
- `docs/phase2_presentation_outline.md` contains a 12-slide outline tied to verified metrics. An editable PowerPoint was not produced: PowerPoint COM launch returned `E_ACCESSDENIED`, native UI automation could not activate the window, and `python-pptx` is unavailable in the existing environment. No package was installed and no placeholder `.pptx` was created.
- The README distinguishes verified reproduction of saved Phase 2/4 metrics and error analysis from historical Phase 1 comparisons that cannot be recreated from the cleaned checkout. MIT-BIH raw data is not included and must be acquired separately.

### Current verification

Using the existing interpreter at `D:\Downloads\ecg project\.venv\Scripts\python.exe` from the cleanup checkout:

- `python -m unittest discover -s tests -v`: **29 tests passed**.
- `python -m compileall -q pipeline.py dataset_audit.py app.py scripts artifacts tests`: passed in the final pass. `app.py` was compiled for syntax verification but was not edited.
- `scripts/phase2_error_analysis.py --data-dir "D:\Downloads\ecg project\data\mitdb" --out "$env:TEMP\ecg_phase2_verification"`: completed without touching the checked-in analysis output folder. It independently recomputed DS2 metrics from saved Phase 2 predictions and labels and matched `results.json` exactly. RF macro-F1/accuracy were 0.385958/0.944409; CNN values were 0.285512/0.733497. The representative waveform examples were matched to actual annotations.
- `artifacts/phase2_rf_improved/verify_rf_improved_ds2.py --out artifacts/phase2_rf_improved`: passed its independent recomputation for the frozen RF; record 232 accuracy was 0.219910 and present-class macro-F1 was 0.251609. This is evaluation reporting only; record 232 was not used for selection.
- The test suite independently checks Phase 2 and Phase 4 aggregate/per-class metrics, confusion matrices, checkpoint/config consistency, prediction labels, and split separation. The saved outputs are therefore reproducible from the tracked predictions without retraining.
- Compilation found no syntax errors. Python source scans found no hard-coded machine-specific absolute paths; scripts construct paths relative to their source or accept data/output arguments. `artifacts/phase2_corrected/eda/eda_summary.json` does retain the original machine's absolute source-data/checkpoint provenance strings; they are descriptive metadata and are not used by the pipeline.
- Required Python dependencies are pinned in `requirements.txt`; the existing environment reports Python 3.11.9 and installed versions matching the listed pins. No package was installed or upgraded.

### Remaining ML rubric findings from the supplied screenshots

| Visible rubric item | Status | Evidence and limitation |
|---|---|---|
| 21.1 — demonstrate the complete ML problem-solving pipeline | IMPLEMENTED AND VERIFIED | `pipeline.py`, Phase 2/4 training scripts, saved predictions/results, tests, and the 12-section report cover problem/data/preparation/model/training/evaluation/analysis/conclusion for the corrected experiments. Historical Phase 1 artifacts are not in this checkout. |
| 22.1 — dataset understanding, missing/inconsistent data, preprocessing, imbalance and information cut-off | PARTIALLY IMPLEMENTED | Mapping, validity/exclusion reasons, record split/leakage tests, counts, and both timing designs are documented and tested. There is no general signal repair/imputation; the exact Clause 4.4 wording is not in the screenshots, so exact clause-level cutoff compliance cannot be checked. |
| 22.2 — EDA, patterns and visuals | IMPLEMENTED AND VERIFIED | `scripts/phase2_eda.py` and saved DS1-only EDA figures are cited and interpreted in report §4. The figures are descriptive and do not establish class separability. |
| 22.3 — implement a single suitable ML model, train, tune and select | PARTIALLY IMPLEMENTED | RF is the selected primary model; its bounded settings are selected using DS1 validation. CNN is retained as a comparison because the user explicitly approved it. The literal one-model clause is therefore not met; the RF remains the sole primary selection. No DS2 selection was performed. |
| 22.4 — approved evaluation, primary/secondary metrics, both Phase 1 baselines, margins and fair comparison | PARTIALLY IMPLEMENTED | Macro-F1, accuracy, per-class reports, matrices, and matched majority/RR-only baselines are verified from the corrected saved predictions. Historical Phase 1 baseline figures and margins are reported, but historical extraction retains 14 more DS2 beats; Phase 1 reference files are outside this checkout. |
| 22.5 — explain performance, patterns, feature/preprocessing effects and limitations | PARTIALLY IMPLEMENTED | Report §9 and the DS2 error analysis interpret actual metrics, errors, class supports, and feature importance. A preprocessing ablation and signal-quality/blinded waveform review were not performed, so those effects/causes are not established. |
| 22.6 — address problem, Phase 1 criterion, limitations and future work | PARTIALLY IMPLEMENTED | The report states the task, metrics, limitations and future work. It reports the stored 0.05 margin and historical baseline results, but the unseen Phase 1 approval wording prevents an exact success-criterion certification. |
| 23.2 — organized executable source for loading through evaluation | IMPLEMENTED AND VERIFIED for corrected Phase 2/4 | Shared extraction, training/evaluation scripts, saved configurations/results, and 29 automated tests are present. Historical Phase 1 baseline training source/artifacts are not part of this cleaned checkout. |
| 23.3 — dataset source, access and preparation | IMPLEMENTED AND VERIFIED | README and final report identify PhysioNet, expected `.hea/.dat/.atr` files under `data/mitdb/`, and required preparation. Raw MIT-BIH data is not committed. |
| 24.1 — Data Preparation and Preprocessing (3 marks) | PARTIALLY IMPLEMENTED | Annotation filtering, record-wise split, leakage checks and cutoff designs are implemented; see 22.1. General repair and exact unseen Clause 4.4 check remain unavailable. |
| 24.1 — Model Implementation and Experimental Setup (3 marks) | PARTIALLY IMPLEMENTED | Training/checkpoints/configs are verified, but the implementation includes a CNN comparison despite the literal single-model wording. This is retained per user approval. |
| 24.1 — Model Evaluation (3 marks) | PARTIALLY IMPLEMENTED | Current corrected metrics/baselines are reproducible; historical Phase 1 figures are not exactly beat-matched and are not reproducible from a fresh clone. |
| 24.1 — Critical Insights and Interpretation (3 marks) | PARTIALLY IMPLEMENTED | Actual imbalance/error patterns are analyzed; causal reasons and preprocessing effects are not established. |
| 24.1 — Technical Implementation and Reproducibility (1 mark) | PARTIALLY IMPLEMENTED | Corrected experiments pass the current suite and metrics reproduce from saved outputs. Fresh full regeneration requires separately acquired raw data; original Phase 1 historical artifacts are unavailable in this checkout. |
| 24.1 — Report Quality (1 mark) | IMPLEMENTED AND VERIFIED as ML documentation | `docs/phase2_final_report.md` contains the requested 12 numbered sections, actual metrics, analysis and limitations. Formal grading remains the instructor's decision. |
| 25.1 — evidence-supported critical interpretation | PARTIALLY IMPLEMENTED | Report and error analysis support interpretations with saved metrics and counts and avoid claiming causes. No signal-quality review or preprocessing ablation supports causal explanations. |
| 26.1 — problem formulation, preprocessing, model choice, fair comparison, errors, criterion and limitations | PARTIALLY IMPLEMENTED | The report covers these areas; the remaining constraints are the single-model clause, non-identical historical beat sets, and unavailable Phase 1 success-criterion wording described above. |

### Remaining constraints

- Section 27 and the Phase 1 approval/success-criterion wording were not included in the screenshots, so requirements appearing only there remain unverified.
- Presentation quality, UI, and live demonstration were intentionally not addressed in this ML-only pass, per the user's instruction.
- The historical Phase 1 artifacts remain outside the cleaned checkout; historical comparison numbers are documented but cannot be independently reproduced from a fresh clone.
- F and Q remain severely underrepresented in DS1 fit (31 and 4 beats); RF misses both classes on DS2, and RF S recall is only 0.0579. These limitations must accompany presentation of aggregate accuracy.


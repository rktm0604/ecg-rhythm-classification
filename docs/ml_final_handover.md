# ML Handover for the UI and Demonstration Teammate

## Purpose and selected models

This handover documents the saved machine-learning interfaces. The primary selected model is the Phase 2 Random Forest. The Phase 2 1D-CNN is an additional comparison model and should remain available for comparison. Phase 4 contains separate causal RF and CNN models; it is a different experiment and must not be presented as interchangeable with Phase 2.

The project classifies a beat only after its location and reference label have been supplied by MIT-BIH annotations. It does not include an automatic R-peak detector. A demo that accepts a raw continuous ECG therefore needs a separately designed and validated beat-detection component before these classifiers can be used.

## Artifacts

| Experiment/model | Checkpoint | Configuration and label order |
|---|---|---|
| Phase 2 RF (primary) | `artifacts/phase2_corrected/random_forest.joblib` | `artifacts/phase2_corrected/experiment_config.json`; labels `N,S,V,F,Q` |
| Phase 2 CNN (comparison) | `artifacts/phase2_corrected/cnn.pt` | Same Phase 2 configuration; PyTorch state dictionary and metadata |
| Phase 4 RF (causal) | `artifacts/phase4_causal/random_forest.joblib` | `artifacts/phase4_causal/experiment_config.json`; labels `N,S,V,F,Q` |
| Phase 4 CNN (causal) | `artifacts/phase4_causal/cnn.pt` | Same Phase 4 configuration; PyTorch state dictionary and metadata |

The checkpoints are preserved experimental artifacts. Do not overwrite them by running a training script. `predictions.npz` and `results.json` in each phase are the saved evaluation evidence, not inputs needed to classify a new beat.

## Input requirements and order

All waveform windows are single-lead MLII, sampled at 360 Hz, per-beat z-scored, and represented as 32-bit floating point values.

### Phase 2 RF

Provide one 34-element feature vector in this exact order:

1. `rr_prev`: samples between the previous valid mapped heartbeat and current beat.
2. `rr_next`: samples between current beat and the next valid mapped heartbeat.
3. `rr_prev / rr_local`.
4. `rr_next / rr_local`.
5. Thirty waveform values selected at rounded, evenly spaced indices from 0 through 199 of the normalized Phase 2 window.

The window is `signal[R-90:R+110]`: index 90 is the annotated R-peak and indices 90–199 include 110 samples from R through R+109. The `rr_local` denominator is the mean of up to two preceding and two following intervals between valid AAMI heartbeat positions. The model expects shape `(n_beats, 34)`; for one beat, use `(1, 34)`. Keep the order fixed. The serialized sklearn estimator accepts these features and returns labels from `N,S,V,F,Q`.

Phase 2 is **offline/delayed**. `rr_next` and the following intervals in `rr_local` require future beat annotations, and the waveform includes future samples. A prediction cannot be shown as an instantaneous output at R.

### Phase 2 CNN

Provide the same normalized 200-sample waveform, without engineered RR features, with batch shape `(n_beats, 1, 200)` as a `float32` tensor. The CNN checkpoint stores a state dictionary and metadata; instantiate the `ECGNet` class in `artifacts/phase2_corrected/train_eval_phase2.py`, load the checkpoint, switch to evaluation mode, and map the argmax index using the saved label order. The CNN output is a five-class logit vector per beat.

### Phase 4 causal models

Phase 4 uses a different input contract. RF receives a 33-element vector ordered as `rr_prev`, `rr_local`, `rr_prev / rr_local`, then 30 evenly spaced waveform values. Its waveform is `signal[R-199:R+1]`, with the annotated R at index 199 and no later samples. `rr_local` is the mean of up to four preceding valid heartbeat intervals. The CNN receives shape `(n_beats, 1, 200)` using that same causal waveform. There is no `rr_next`. Phase 4 models and preprocessing must never be loaded with Phase 2 feature vectors.

Unavailable previous-beat context is explicitly excluded in Phase 4 extraction. In either phase, non-beat rhythm annotations are not RR endpoints; only symbols in the approved five-class heartbeat mapping define heartbeat positions.

## Output interpretation

Use the checkpoint's class order, not an independently assumed order. For RF, `predict_proba` returns the forest's class vote fractions. These are **not calibrated clinical probabilities** and should not be described as a subject's probability of disease. Do not apply an undocumented threshold or reinterpret a label as a diagnosis. CNN outputs are logits; a softmax can be used to display normalized model scores, but those scores are not clinically calibrated probabilities either.

## Demo guidance

For an annotated-record demonstration, Phase 2 is appropriate when the UI clearly labels results as offline/delayed and makes the future information cut-off clear. Phase 4 better illustrates a causal-at-R feature design, but it still assumes a valid annotated R-peak is already available. Neither phase supports a raw-stream/live ECG claim without a separately implemented and validated detector. The UI/live demonstration is owned by the teammate; this repository change adds no UI code.

## Limits to show with the result

- Primary RF Phase 2 DS2 macro-F1 is 0.3860 and accuracy is 0.9444; RF S recall is 0.0579 and F/Q recall are zero.
- CNN Phase 2 DS2 macro-F1 is 0.2855 and accuracy is 0.7335; Q recall is zero.
- Only 31 F and 4 Q beats were present in DS1 fit. Rare-class estimates are weak; DS2 Q support is 7.
- Record 232 has unusually poor observed Phase 2 performance, but it was held out and was not used for selection.
- The dataset is not population-representative; the project is not a clinically validated diagnostic device.

See `docs/phase2_final_report.md`, `docs/phase2_error_analysis.md`, and `artifacts/phase4_causal/feature_timing.md` for methods and evidence.


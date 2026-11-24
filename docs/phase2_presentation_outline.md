# Phase 2 Presentation Outline

This is a slide-by-slide outline with speaker prompts, not a rendered slide deck.
Use it to prepare a short presentation of the corrected Phase 2 experiment.

## Slide 1 — Question and scope

- Classify an already annotated heartbeat into AAMI groups N, S, V, F, Q.
- Primary experiment: offline/delayed; Phase 4 causal is a separate experiment.
- Reference annotations supply beat positions; this project does not detect beats.

**Say:** “The question is how well a known annotated beat can be classified, not how to find beats in a continuous ECG.”

## Slide 2 — Dataset and split

- MIT-BIH Arrhythmia Database: 48 recordings, 47 subjects, 360 Hz.
- Record-wise DS1 fit/validation and held-out DS2 test.
- Explain shared-subject exclusion for records 201/202.

**Say:** “Splitting by recording and subject reduces leakage from the same person appearing on both sides.”

## Slide 3 — Annotation mapping and beat extraction

- Show the five AAMI groups and their approved symbols.
- Only mapped heartbeat annotations are RR endpoints.
- 200-sample window: 90 samples before and 110 from the annotated R location onward.
- Per-window z-score; RR context uses previous and following valid beats.

**Say:** “The extraction is annotation-assisted and the future context makes this a delayed task.”

## Slide 4 — Class balance and EDA

- Show `artifacts/phase2_corrected/eda/class_distribution_ds1.png`.
- Point out the very small DS1 fit supports for F (31) and Q (4).
- Briefly show representative waveforms and RR distributions.

**Say:** “Overall accuracy can look high because N dominates; the minority supports are too small for stable F/Q estimates.”

## Slide 5 — Models and inputs

- Primary: Random Forest on 34 engineered RR/waveform features.
- Comparison: 1D-CNN on the full 200-sample waveform.
- Both use identical fit/validation examples; baselines are majority and RR-only Logistic Regression.

**Say:** “The models share examples and partitions, but not the representation, so this compares two pipelines rather than isolating architecture alone.”

## Slide 6 — Training and selection

- RF candidate settings are compared using DS1 validation macro-F1.
- CNN training uses weighted cross-entropy and early stopping on DS1 validation macro-F1.
- DS2 is used only for final reporting.

**Say:** “No test-set score was used to select models, tune settings, choose weights, or thresholds.”

## Slide 7 — Final results

- RF DS2: macro-F1 0.3860, accuracy 0.9444.
- CNN DS2: macro-F1 0.2855, accuracy 0.7335.
- Show per-class table or confusion matrices from `docs/phase2_final_report.md` §7.

**Say:** “High RF accuracy coexists with poor minority-class recognition; macro-F1 exposes that imbalance.”

## Slide 8 — Baseline comparison

- Current matched Phase 2 baselines: majority macro-F1 0.1875; RR-only LR 0.2588.
- RF gains: +0.1985 and +0.1271 macro-F1, respectively.
- Historical Phase 1 scores are also listed in report §8, but DS2 retained 14 more beats under old extraction.

**Say:** “The corrected matched comparison is the cleanest within-Phase-2 comparison. Historical Phase 1 comparison is descriptive because retained examples differ.”

## Slide 9 — Errors and limitations

- RF F and Q recall are zero; CNN F recall is 0.0620 and Q recall is zero.
- RF commonly maps S and F to N; CNN confuses many N and S beats with other classes.
- Small F/Q fit support limits conclusions; annotated beat locations are assumed.

**Say:** “These scores do not establish clinical usefulness or robust rare-class performance.”

## Slide 10 — Conclusion and next steps

- The corrected offline/delayed five-class task is implemented and reproducible from saved artifacts.
- RF is the primary validation-selected model; CNN is an additional comparison.
- Obtain the official Sections 21–27 rubric and create a rendered deck if required.
- Future work: more rare-class data, DS1-only ablations, and stratified error review.

**Say:** “The project is ready for a technical review, subject to confirming the exact rubric and presentation format.”

## Likely questions to prepare for

1. Why is this offline/delayed rather than causal? It uses `rr_next`, future RR context, and samples after R.
2. Why report macro-F1 as well as accuracy? Macro-F1 weights each class equally and reveals minority-class failures.
3. Why are F and Q results weak? DS1 fit has only 31 F and 4 Q examples.
4. Did DS2 affect selection? No; selection is recorded as DS1 validation only.
5. Does the system detect heartbeats? No; it classifies reference-annotated beats.
6. Why compare to Phase 1 cautiously? Historical DS2 retained 14 more beats than corrected extraction.

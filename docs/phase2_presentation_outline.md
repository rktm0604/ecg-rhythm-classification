# ECG Machine Learning — Final Presentation Outline

**Target:** 12 editable slides, approximately 8–10 minutes. This is a slide-by-slide outline with titles, key visual, and concise speaker prompts; it is not a rendered PowerPoint file.

## Slide 1 — Five-class ECG beat classification

- MIT-BIH Arrhythmia Database; annotation-assisted beat classification.
- Primary: Phase 2 offline/delayed Random Forest.
- Secondary comparisons: Phase 2 CNN and separate Phase 4 causal models.
- State immediately: not automatic beat detection and not a diagnostic device.

**Visual:** simple pipeline: annotated beat → features/model → AAMI label.

**Say:** “The project classifies reference-annotated beats; it does not find beats in a continuous signal.”

## Slide 2 — Problem and objectives

- Predict one of N, S, V, F, Q for an annotated beat.
- Compare RF and 1D-CNN with shared record partitions; report baselines and per-class metrics.
- Macro-F1 is primary because class support is highly imbalanced.

**Visual:** five output class chips plus a small macro-F1 versus accuracy note.

**Say:** “Accuracy alone overstates performance when normal beats dominate.”

## Slide 3 — Dataset and AAMI mapping

- MIT-BIH: 48 recordings, 47 subjects, 360 Hz.
- N: `N,L,R,e,j`; S: `A,a,J,S`; V: `V,E`; F: `F`; Q: `/,f,Q`.
- Reference annotations define both beat positions and target labels.

**Visual:** five-row mapping table; small class-support bar plot from `artifacts/phase2_corrected/eda/class_distribution_ds1.png`.

**Say:** “Non-beat rhythm annotations remain metadata and are never RR endpoints.”

## Slide 4 — Extraction and feature design

- Corrected filtering uses valid AAMI heartbeat positions for RR intervals.
- Phase 2 window: 200 samples, `R−90` through `R+109`; 34 RF features, full waveform for CNN.
- Phase 2 RR features include following-beat context.
- Phase 4 window: `R−199` through `R`; previous-only RR context.

**Visual:** two aligned timelines with R marker and shaded future samples/context.

**Say:** “Phase 2 is delayed; Phase 4 is causal relative to the annotated R-peak.”

## Slide 5 — Record split and evaluation design

- DS1 fit (18): 106, 108, 109, 112, 114, 115, 116, 121, 122, 124, 201, 203, 207, 209, 215, 220, 223, 230.
- DS1 validation (5): 101, 118, 119, 205, 208.
- DS2 final evaluation (20): 100, 103, 105, 111, 113, 117, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.
- Record 202 excluded due to shared subject with 201; paced records 102,104,107,217 excluded.

**Visual:** split diagram with DS1 selection arrows and one-way DS2 evaluation arrow.

**Say:** “No DS2 data informed feature, weight, threshold, or model selection.”

## Slide 6 — Models and training

- Primary RF: 34 engineered inputs; bounded DS1 validation comparison.
- Additional 1D-CNN: three Conv1D blocks on the 200-sample waveform; early stopping on DS1 validation.
- Baselines: majority class and RR-only logistic regression.
- Same examples and fit/validation records, different model input representations.

**Visual:** compact RF/CNN/baseline comparison table.

**Say:** “The comparison evaluates complete pipelines; input representation differs, so architecture alone is not isolated.”

## Slide 7 — Phase 2 results

| Model | DS2 macro-F1 | DS2 accuracy |
|---|---:|---:|
| Random Forest | 0.3860 | 0.9444 |
| 1D-CNN | 0.2855 | 0.7335 |
| Majority baseline | 0.1875 | 0.8824 |
| RR-only logistic regression | 0.2588 | 0.4670 |

- RF selected using higher DS1 validation macro-F1 (0.3992 versus CNN 0.3717).

**Visual:** paired bars for accuracy and macro-F1; label axes clearly.

**Say:** “RF has the highest aggregate macro-F1, but its minority-class performance remains weak.”

## Slide 8 — Per-class performance and imbalance

- RF DS2 F1: N 0.9770, S 0.1047, V 0.8480, F 0, Q 0.
- DS1 fit has only F=31 and Q=4; DS2 has F=387 and Q=7.
- RF S recall is 0.0579; CNN S recall is 0.0539; both Q recalls are zero.

**Visual:** per-class F1 dot/bar chart with support labels.

**Say:** “Rare-class supports limit what can be concluded; do not present the estimates as stable population performance.”

## Slide 9 — Error analysis

- RF common errors: S→N 1,041; S→V 636; F→N 378; V→N 306.
- CNN common errors: N→V 3,645; N→F 3,560; N→S 2,297; S→N 1,305.
- Record 232: RF classified 0/1,381 S beats correctly and produced 1,387 total errors; this is post-hoc held-out observation only.
- CNN errors are more distributed across records.

**Visual:** confusion matrices from `artifacts/phase2_corrected/error_analysis/` and one traceable example from the report.

**Say:** “The counts describe these held-out records and do not establish physiological or clinical causes.”

## Slide 10 — Phase 4 causal comparison

| Model | Phase 2 macro-F1 | Phase 4 macro-F1 | Difference |
|---|---:|---:|---:|
| Random Forest | 0.3860 | 0.3784 | −0.0075 |
| 1D-CNN | 0.2855 | 0.3460 | +0.0604 |

- Phase 4 uses only waveform through R and previous RR intervals.
- Eligible sample sets differ, so this is not a controlled paired comparison of identical tasks.

**Visual:** two annotated timing diagrams plus the comparison table.

**Say:** “The CNN score is higher in the causal run, but task timing and eligibility differ, so interpret it as a separate experiment.”

## Slide 11 — Limitations and responsible interpretation

- No automatic R-peak detection; annotation locations are assumed.
- Severe class imbalance, particularly F and Q; S recall is low.
- No confidence intervals, calibration study, preprocessing ablation, or external dataset evaluation.
- MIT-BIH is not population-representative; no clinical validation.

**Visual:** concise limitation checklist.

**Say:** “This is an academic ML result, not a clinical diagnostic system.”

## Slide 12 — Conclusion and future work

- Corrected five-class pipeline and saved evaluation artifacts are verified by tests and independent recomputation.
- RF remains the primary model; CNN is retained as comparison.
- Next: more independent rare-class data, prespecified DS1-only analyses, external validation, and a separately validated beat detector if a raw-signal demo is needed.

**Visual:** three takeaway statements and future-work arrows.

**Say:** “The main contribution is a reproducible, carefully bounded experiment with explicit limitations.”

## Likely questions

1. Why macro-F1? It weights each class equally and exposes minority-class failures hidden by accuracy.
2. Did DS2 or record 232 guide tuning? No; it was reserved for frozen final evaluation and post-hoc error analysis.
3. Why is Phase 2 delayed? It uses following-beat RR context and post-R waveform samples.
4. Is the RF output probability calibrated? No; `predict_proba` gives vote fractions, not calibrated clinical probabilities.
5. Can this classify raw continuous ECG? Not by itself; reference beat locations are assumed.

## Delivery status

The editable `.pptx` requested in the accompanying task could not be generated in the current environment: PowerPoint COM activation returned access denied; native UI automation also failed because the PowerPoint window could not be activated; and `python-pptx` is unavailable. No package was installed and no placeholder file was created. This outline contains the verified slide content for later rendering in PowerPoint.


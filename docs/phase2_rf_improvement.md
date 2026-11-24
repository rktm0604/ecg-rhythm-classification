# Phase 2 Random Forest DS1 Investigation and Final Evaluation

**Primary result:** Existing Phase 2 Random Forest retained.
**Additional model:** The Phase 2 1D-CNN and all Phase 4 files are unchanged.
**Task:** Corrected five-class AAMI beat classification, offline/delayed.
**Selection data:** DS1 fit and validation only.
**Final evaluation:** The frozen RF was evaluated once on DS2; DS2 did not select or tune it.

## 1. Original problem

The Phase 2 experiment classifies a known, expert-annotated heartbeat into N, S, V, F, or Q. It is not automatic heartbeat detection. The offline/delayed design uses a 200-sample window (90 before and 110 starting at R) and following-beat RR features. The same corrected extraction, AAMI mapping, and record split were kept in this study.

## 2. DS1 split and method

The experiment loaded only DS1 records while investigating and fitting candidates. The same record-grouped split as Phase 2 was verified:

- **Fit (18):** 106, 108, 109, 112, 114, 115, 116, 121, 122, 124, 201, 203, 207, 209, 215, 220, 223, 230
- **Validation (5):** 101, 118, 119, 205, 208
- The separate final test remains the Phase 2 DS2 record set; no DS2 features, labels, predictions, scores, or record-level findings were accessed until after the selection config and checkpoint were frozen.

The current Phase 2 checkpoint was used as the DS1 validation baseline and its saved validation predictions were reproduced exactly. The bounded candidate study then varied class weights, depth, leaf size, and feature families. Every forest used the same DS1 fit rows, seed 42, 300 trees, `max_features="sqrt"`, and the fixed validation records. Macro-F1 across all five configured classes was the primary ranking metric.

Before fitting candidate variants, the selection rule was fixed: require at least +0.01 absolute validation macro-F1, at least +0.02 S-F1, no more than 0.02 absolute F1 decrease for N or V, and no more than 0.02 accuracy decrease. F and Q metrics were reported but not made hard selection gates because validation supports were only 383 and 4. If no candidate passed, retain the existing RF.

## 3. S-class distribution across DS1

Corrected DS1 contains **944 S beats**. The class is concentrated in a few recordings; seven of the 23 DS1 records have no eligible S beats, and record 209 alone has 383.

| DS1 record | S support | Total eligible beats |
|---:|---:|---:|
| 101 | 3 | 1,863 |
| 106 | 0 | 2,025 |
| 108 | 4 | 1,761 |
| 109 | 0 | 2,530 |
| 112 | 2 | 2,537 |
| 114 | 12 | 1,877 |
| 115 | 0 | 1,951 |
| 116 | 1 | 2,410 |
| 118 | 96 | 2,276 |
| 119 | 0 | 1,985 |
| 121 | 1 | 1,861 |
| 122 | 0 | 2,474 |
| 124 | 31 | 1,617 |
| 201 | 128 | 1,961 |
| 203 | 2 | 2,978 |
| 205 | 3 | 2,654 |
| 207 | 106 | 1,858 |
| 208 | 2 | 2,953 |
| 209 | 383 | 3,003 |
| 215 | 3 | 3,361 |
| 220 | 94 | 2,046 |
| 223 | 73 | 2,603 |
| 230 | 0 | 2,254 |

Fit contains 840 S beats; validation has 104. Most validation S examples are in record 118 (96/104), with only 3 in 101, 3 in 205, 2 in 208, and none in 119. This makes the validation S estimate heavily dependent on one record.

## 4. Current RF DS1 validation diagnosis

The current model reproduced validation macro-F1 **0.39924**, accuracy **0.95584**, and S precision/recall/F1 **1.000 / 0.0385 / 0.0741** (support 104). It correctly labels 4 S beats, predicts S only 4 times, sends 86 S beats to V and 14 to N. Its N F1 is 0.9819 and V F1 0.9402; it identifies none of validation F (support 383) or Q (support 4).

The N/S/V portion of the DS1 validation confusion matrix (rows true; columns predicted, ordered N/S/V) is:

```text
[[9707,    0,   10],
 [  14,    4,   86],
 [  21,    0, 1502]]
```

| Validation record | N | S | V | F | Q | Accuracy | Five-class macro-F1 | S precision | S recall | S F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 1,858 | 3 | 0 | 0 | 2 | 0.9979 | 0.2998 | 1.000 | 0.333 | 0.500 |
| 118 | 2,164 | 96 | 16 | 0 | 0 | 0.9534 | 0.2467 | 0 | 0 | 0 |
| 119 | 1,541 | 0 | 444 | 0 | 0 | 1.0000 | 0.4000 | — | — | — |
| 205 | 2,569 | 3 | 71 | 11 | 0 | 0.9932 | 0.5889 | 1.000 | 1.000 | 1.000 |
| 208 | 1,585 | 2 | 992 | 372 | 2 | 0.8679 | 0.3733 | 0 | 0 | 0 |

Macro-F1 is fixed across all five classes here, so a missing class in a record contributes zero. “—” means that record has no true S support; its S precision/recall is not a meaningful within-record estimate. Record 205's perfect S result is only 3 examples and is not a stable estimate. Record 118 is the informative validation case: 0/96 S correct, with 86 called V and 10 called N.

The current RF's validation S-probability distribution also overlaps heavily with V. For true S (n=104), S-probability median is 0.090 and 75th percentile 0.114; for true V (n=1,523), median is 0.147 and 25th percentile 0.087. True S mean S-probability is 0.114 versus 0.147 for true V. These are RF vote fractions, not calibrated probabilities, and no threshold was changed.

## 5. Feature investigation

The saved RF's impurity importance totals **0.4201 for the four RR-derived features** and **0.5799 for the 30 sampled waveform values**. The highest individual features are `rr_prev / rr_local` (0.1480), `rr_prev` (0.1473), `rr_next` (0.0684), and `rr_next / rr_local` (0.0563). Importance is global and can be redistributed among correlated features; it is not class-specific evidence or causal attribution.

Small DS1-validation feature-family ablations used the original RF settings:

| Feature family | Macro-F1 | Accuracy | S precision | S recall | S F1 | N F1 | V F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 34 (current model) | 0.3992 | 0.9558 | 1.000 | 0.0385 | 0.0741 | 0.9819 | 0.9402 |
| RR only (4) | 0.3476 | 0.8859 | 0.1369 | 0.2212 | 0.1691 | 0.9415 | 0.6274 |
| Waveform only (30) | 0.3700 | 0.9333 | 0.1250 | 0.0096 | 0.0179 | 0.9661 | 0.8659 |

RR-only improves S recall/F1 on this validation split, but substantially lowers S precision, overall macro-F1, accuracy, and V performance. Waveform-only alone performs poorly for S. The full-feature model remains stronger overall. This single ablation does not prove which features cause errors, and the validation split is too small and S-skewed to make reliable general claims.

## 6. Controlled RF candidate experiments

All candidate outcomes below are DS1-validation-only. The core set included six weight variants at the original tree structure; six depth/leaf alternatives with the existing custom weights; and four S-upweight/regularization interactions. Two feature-family ablations were added. All used 300 trees and the unchanged five classes.

| Candidate | Macro-F1 | Accuracy | S precision | S recall | S F1 | N F1 | V F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing custom weights, depth=None, leaf=1 | 0.3992 | 0.9558 | 1.000 | 0.0385 | 0.0741 | 0.9819 | 0.9402 |
| Balanced, depth=None, leaf=1 | 0.3970 | 0.9517 | 1.000 | 0.0385 | 0.0741 | 0.9778 | 0.9334 |
| Balanced-subsample, depth=None, leaf=1 | 0.3953 | 0.9505 | 1.000 | 0.0385 | 0.0741 | 0.9778 | 0.9252 |
| Unweighted, depth=None, leaf=1 | 0.3941 | 0.9544 | 0.1364 | 0.0288 | 0.0476 | 0.9810 | 0.9424 |
| S=10, depth=None, leaf=1 | 0.3997 | 0.9558 | 0.800 | 0.0385 | 0.0734 | 0.9810 | 0.9440 |
| S=20, depth=None, leaf=1 | 0.3989 | 0.9558 | 0.800 | 0.0385 | 0.0734 | 0.9819 | 0.9394 |
| Existing weights, depth=24, leaf=1 | 0.3992 | 0.9559 | 1.000 | 0.0385 | 0.0741 | 0.9819 | 0.9402 |
| Existing weights, depth=24, leaf=3 | 0.3983 | 0.9549 | 0.250 | 0.0385 | 0.0667 | 0.9810 | 0.9442 |
| Existing weights, depth=24, leaf=5 | 0.3972 | 0.9541 | 0.1818 | 0.0385 | 0.0635 | 0.9801 | 0.9424 |
| Existing weights, depth=12, leaf=1 | **0.3999** | 0.9557 | 0.5714 | 0.0385 | 0.0721 | 0.9810 | 0.9457 |
| Existing weights, depth=12, leaf=3 | 0.3967 | 0.9538 | 0.1739 | 0.0385 | 0.0630 | 0.9801 | 0.9403 |
| Existing weights, depth=12, leaf=5 | 0.3968 | 0.9529 | 0.1600 | 0.0385 | 0.0620 | 0.9792 | 0.9432 |
| S=10, depth=None, leaf=3 | 0.3983 | 0.9551 | 0.3333 | 0.0385 | 0.0690 | 0.9810 | 0.9417 |
| S=10, depth=12, leaf=3 | 0.3956 | 0.9530 | 0.1143 | 0.0385 | 0.0576 | 0.9801 | 0.9403 |
| S=20, depth=None, leaf=3 | 0.3974 | 0.9547 | 0.2667 | 0.0385 | 0.0672 | 0.9810 | 0.9394 |
| S=20, depth=12, leaf=3 | 0.3797 | 0.9417 | 0.0237 | 0.0385 | 0.0293 | 0.9810 | 0.8884 |
| RR-only (4 features) | 0.3476 | 0.8859 | 0.1369 | 0.2212 | 0.1691 | 0.9415 | 0.6274 |
| Waveform-only (30 features) | 0.3700 | 0.9333 | 0.1250 | 0.0096 | 0.0179 | 0.9661 | 0.8659 |

The top candidate, depth 12/leaf 1, adds only **0.00062** macro-F1 and lowers S-F1 from 0.0741 to 0.0721. S=10 adds only 0.00044 macro-F1 and leaves S recall unchanged while reducing S precision to 0.80. RR-only meaningfully raises S F1 but loses 0.0516 macro-F1 and 0.0700 accuracy. **No candidate passed the predeclared selection rule**, so the current RF was retained. The complete metrics, supports, confusion matrices, parameters, and validation predictions for all candidates are preserved in `artifacts/phase2_rf_improved/`.

## 7. Frozen decision

`frozen_model_config.json` and `random_forest_frozen.joblib` were written after the DS1-only study. The selected model is a copy of the original Phase 2 RF: 300 trees, unlimited depth, leaf size 1, custom class weights N=1, S=5, V=1, F=8, Q=1, seed 42, full 34 features. It was retained because the alternatives did not deliver a convincing S improvement and acceptable aggregate performance. The selected model does not promise better behavior on any particular test record.

## 8. Final DS2 evaluation after freeze

Only after the DS1 configuration was frozen, the selected checkpoint was run once on DS2. The final DS2 predictions exactly match the existing Phase 2 RF predictions, confirming the selected model is unchanged. The final evaluation is not a new model improvement.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| N | 40,300 | 0.9587 | 0.9961 | 0.9770 |
| S | 1,780 | 0.5508 | 0.0579 | 0.1047 |
| V | 3,199 | 0.7996 | 0.9028 | 0.8480 |
| F | 387 | 0 | 0 | 0 |
| Q | 7 | 0 | 0 | 0 |

**Selected RF:** macro-F1 0.385958, accuracy 0.944409.
**Existing RF:** macro-F1 0.385958, accuracy 0.944409.
**Difference:** 0.000000 in macro-F1 and accuracy; predictions identical. All per-class F1 and recall differences are zero.

DS2 confusion matrix (rows true; columns predicted, order N/S/V/F/Q):

```text
[[40143,   79,   78,    0, 0],
 [ 1041,  103,  636,    0, 0],
 [  306,    5, 2888,    0, 0],
 [  378,    0,    9,    0, 0],
 [    6,    0,    1,    0, 0]]
```

The one-shot outputs are `final_ds2_results.json` and `predictions_ds2.npz`. `verify_rf_improved_ds2.py` independently recomputes all aggregate and per-record metrics from those saved predictions without model inference; its output is `final_ds2_metrics_verified.json`.

## 9. Record 232 final results

Record 232 is reported here as part of the final DS2 evaluation; it was not used in the candidate study or selection. It contains **1,778** eligible beats: 397 N and 1,381 S. The retained RF correctly classifies 391/397 N beats but recognizes **0/1,381 S beats**. It classifies 766 S as N and 615 S as V. Record accuracy is **0.2199**. Macro-F1 across the two classes present (N and S) is **0.2516**; a fixed five-class per-record macro-F1 would include unsupported V/F/Q classes and is less meaningful here. This outcome is unchanged from the original RF and is not evidence that the DS1 experiment should have been tuned to this record.

## 10. Limitations and overfitting risk

1. The fixed DS1 validation set has only five records, and 96 of 104 S validation beats are in one record. Candidate selection against it can overfit record-specific patterns even though no DS2 data was used.
2. The S recall is 4/104 for the existing model; estimates are highly sensitive to a few beats and the validation record composition.
3. F and Q validation supports (383 and 4) are small or extremely small; Q precision/recall estimates are not dependable.
4. The grid was intentionally bounded but still compared 18 candidates against the same validation set. Small score differences (under 0.001) are not convincing evidence of a general improvement.
5. Impurity feature importance and group ablations do not establish causation. RR and waveform inputs may be correlated; no class-conditional permutation or independent external cohort was used.
6. Record 232's failure may reflect distribution or morphology differences, but this study does not determine the cause. No signal-quality or annotation review was performed.
7. Since the current RF was retained, this phase establishes no improvement on DS2 or record 232. The original DS2 result is exactly reproduced.

## 11. Reproduction and artifacts

Run using the existing environment and local MIT-BIH files:

```powershell
$studyOut = "artifacts\phase2_rf_improved_reproduction"
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\train_rf_improved.py --data-dir "D:\path\to\mitdb" --out $studyOut
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\evaluate_rf_improved_ds2.py --data-dir "D:\path\to\mitdb" --out $studyOut
.\.venv\Scripts\python.exe artifacts\phase2_rf_improved\verify_rf_improved_ds2.py --out $studyOut
```

Use a new empty output folder for a fresh run. The training script refuses to overwrite a frozen study; the evaluator is intentionally one-shot and refuses to overwrite final DS2 files. The first script never reads DS2 rows or predictions; the second loads DS2 only after the frozen DS1 config exists; the verifier reads saved outputs only. Original Phase 2 results, checkpoints, predictions, CNN, and Phase 4 artifacts remain intact.

Key files: `train_rf_improved.py`, `evaluate_rf_improved_ds2.py`, `verify_rf_improved_ds2.py`, `frozen_model_config.json`, `ds1_results.json`, `ds1_candidate_metrics.csv`, `ds1_validation_predictions.npz`, `random_forest_frozen.joblib`, `predictions_ds2.npz`, `final_ds2_results.json`, and `final_ds2_metrics_verified.json`.


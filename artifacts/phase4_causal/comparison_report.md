# Phase 2 offline/delayed vs Phase 4 causal comparison

These are related but different prediction tasks. Phase 2 uses a 200-sample waveform extending 110 samples after R and future RR intervals. Phase 4 predicts at R using only samples R-199 through R and past RR context. Metrics are descriptive comparisons on the same DS2 records; DS2 was not used for selection.

| Model | P2 macro-F1 | P4 macro-F1 | Difference | P2 accuracy | P4 accuracy | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.3860 | 0.3784 | -0.0075 | 0.9444 | 0.9326 | -0.0118 |
| 1D-CNN | 0.2855 | 0.3460 | +0.0604 | 0.7335 | 0.8088 | +0.0753 |
| Majority baseline | 0.1875 | 0.1875 | +0.0000 | 0.8824 | 0.8824 | +0.0000 |
| RR-only Logistic Regression | 0.2588 | 0.1638 | -0.0950 | 0.4670 | 0.2148 | -0.2523 |

## Per-class differences

F1 and recall differences are Phase 4 minus Phase 2. Supports are the same DS2 annotations when both experiments retain the same beats; any support difference is reported below.

### Random Forest

| Class | Support P2 / P4 | F1 P2 | F1 P4 | ΔF1 | Recall P2 | Recall P4 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|
| N | 40300 / 40319 | 0.9770 | 0.9644 | -0.0127 | 0.9961 | 0.9926 | -0.0035 |
| S | 1780 / 1781 | 0.1047 | 0.0903 | -0.0144 | 0.0579 | 0.0511 | -0.0068 |
| V | 3199 / 3199 | 0.8480 | 0.8375 | -0.0106 | 0.9028 | 0.7827 | -0.1200 |
| F | 387 / 387 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| Q | 7 / 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
### 1D-CNN

| Class | Support P2 / P4 | F1 P2 | F1 P4 | ΔF1 | Recall P2 | Recall P4 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|
| N | 40300 / 40319 | 0.8463 | 0.8945 | +0.0481 | 0.7626 | 0.8282 | +0.0656 |
| S | 1780 / 1781 | 0.0458 | 0.3064 | +0.2606 | 0.0539 | 0.4222 | +0.3683 |
| V | 3199 / 3199 | 0.5244 | 0.5252 | +0.0008 | 0.8274 | 0.8784 | +0.0510 |
| F | 387 / 387 | 0.0110 | 0.0037 | -0.0073 | 0.0620 | 0.0052 | -0.0568 |
| Q | 7 / 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
### Majority baseline

| Class | Support P2 / P4 | F1 P2 | F1 P4 | ΔF1 | Recall P2 | Recall P4 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|
| N | 40300 / 40319 | 0.9375 | 0.9375 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| S | 1780 / 1781 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| V | 3199 / 3199 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| F | 387 / 387 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| Q | 7 / 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
### RR-only Logistic Regression

| Class | Support P2 / P4 | F1 P2 | F1 P4 | ΔF1 | Recall P2 | Recall P4 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|
| N | 40300 / 40319 | 0.6269 | 0.3278 | -0.2991 | 0.4648 | 0.1994 | -0.2654 |
| S | 1780 / 1781 | 0.0360 | 0.0489 | +0.0129 | 0.0736 | 0.0578 | -0.0158 |
| V | 3199 / 3199 | 0.6308 | 0.4416 | -0.1892 | 0.7706 | 0.5195 | -0.2510 |
| F | 387 / 387 | 0.0000 | 0.0003 | +0.0003 | 0.0000 | 0.0052 | +0.0052 |
| Q | 7 / 7 | 0.0005 | 0.0006 | +0.0002 | 0.1429 | 0.7143 | +0.5714 |

## Rare class limitations

Phase 4 fit supports: F=31, Q=4. DS2 supports: F=387, Q=7. These small training and test supports make F/Q estimates unstable; zero scores should not be interpreted as precise estimates of population performance.

"""Independently recompute final DS2 metrics from the one-shot saved outputs.

This verifier never calls a model. It only audits predictions_ds2.npz against
final_ds2_results.json and writes a record-metric view that averages macro-F1
over classes with true support in each record.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LABELS = ["N", "S", "V", "F", "Q"]


def recompute(y: np.ndarray, pred: np.ndarray) -> dict:
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    present = [label for label in LABELS if np.any(y == label)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_all_five_classes": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_classes_with_true_support": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "classes_with_true_support": present,
        "per_class": {
            c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])}
            for c in LABELS
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    with np.load(out / "predictions_ds2.npz", allow_pickle=False) as z:
        p = {key: z[key] for key in z.files}
    y, records = p["y_test"].astype(str), p["record_test"].astype(str)
    old = p["pred_existing_rf"].astype(str)
    selected = p["pred_selected_rf"].astype(str)
    result_path = out / "final_ds2_results.json"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    global_old, global_selected = recompute(y, old), recompute(y, selected)
    for model_name, computed in (("existing_phase2_random_forest", global_old), ("selected_random_forest", global_selected)):
        reported = saved[model_name]
        assert np.isclose(computed["accuracy"], reported["accuracy"], atol=1e-12, rtol=0)
        assert np.isclose(computed["macro_f1_all_five_classes"], reported["macro_f1"], atol=1e-12, rtol=0)
        assert computed["confusion_matrix"] == reported["confusion_matrix"]
        for label in LABELS:
            for field in ("precision", "recall", "f1", "support"):
                saved_field = "f1" if field == "f1" else field
                assert np.isclose(computed["per_class"][label][field], reported["per_class"][label][saved_field], atol=1e-12, rtol=0)

    per_record = {}
    for record in sorted(np.unique(records), key=lambda value: int(value)):
        mask = records == record
        per_record[str(record)] = {"n_beats": int(mask.sum()), "existing": recompute(y[mask], old[mask]),
                                   "selected": recompute(y[mask], selected[mask])}
        saved_record = saved["record_metrics"][str(record)]
        assert per_record[str(record)]["selected"]["confusion_matrix"] == saved_record["metrics"]["confusion_matrix"]
        assert np.isclose(per_record[str(record)]["selected"]["accuracy"], saved_record["metrics"]["accuracy"], atol=1e-12, rtol=0)
    if not np.array_equal(selected, old):
        raise AssertionError("Selected RF predictions differ from existing RF; check saved comparison")

    verified = {
        "verification": "independently recomputed from predictions_ds2.npz; no model inference",
        "n_test_beats": int(len(y)),
        "existing_phase2_rf": global_old,
        "selected_rf": global_selected,
        "record_232": per_record["232"],
        "record_metrics": per_record,
    }
    (out / "final_ds2_metrics_verified.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print("Independent DS2 metrics verification passed.")
    print("macro-F1/accuracy:", global_selected["macro_f1_all_five_classes"], global_selected["accuracy"])
    print("record 232 accuracy:", per_record["232"]["selected"]["accuracy"])
    print("record 232 macro-F1 over present classes:", per_record["232"]["selected"]["macro_f1_classes_with_true_support"])


if __name__ == "__main__":
    main()

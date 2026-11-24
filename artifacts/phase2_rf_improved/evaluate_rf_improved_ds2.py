"""One-shot final DS2 evaluation of an already frozen Phase 2 RF.

Run only after train_rf_improved.py has created frozen_model_config.json.
It refuses to run twice and writes only to this separate experiment folder.
No model is trained or selected here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipeline

PHASE2 = ROOT / "artifacts" / "phase2_corrected"
OUT = Path(__file__).resolve().parent
LABELS = list(pipeline.LABELS)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])}
            for c in LABELS
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mitdb")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    result_path = out / "final_ds2_results.json"
    pred_path = out / "predictions_ds2.npz"
    if result_path.exists() or pred_path.exists():
        raise FileExistsError("One-shot DS2 outputs already exist; refusing to evaluate again")
    frozen_path = out / "frozen_model_config.json"
    checkpoint = out / "random_forest_frozen.joblib"
    if not frozen_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Freeze the DS1-selected model/config before DS2 evaluation")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("ds2_status") != "not evaluated yet; this configuration is frozen before final test evaluation":
        raise RuntimeError("Frozen config does not indicate a not-yet-evaluated DS2 run")
    if frozen["labels"] != LABELS:
        raise RuntimeError("Frozen model label order differs from pipeline AAMI labels")

    phase2_config = json.loads((PHASE2 / "experiment_config.json").read_text(encoding="utf-8"))
    test_records = phase2_config["test_records"]
    pipeline.DATA_DIR = args.data_dir.resolve()
    test_data = pipeline.build_dataset(test_records)
    y = test_data["y"].astype(str)
    records = test_data["record"].astype(str)
    beat_numbers = test_data["beat_number"].astype(int)

    # Alignment against historical frozen rows only; no tuning/selection occurs.
    with np.load(PHASE2 / "predictions.npz", allow_pickle=False) as previous:
        old_y = previous["y_test"].astype(str)
        old_records = previous["record_test"].astype(str)
        old_beats = previous["beat_number_test"].astype(int)
        old_pred = previous["pred_rf"].astype(str)
    if not (np.array_equal(y, old_y) and np.array_equal(records, old_records)
            and np.array_equal(beat_numbers, old_beats)):
        raise RuntimeError("Corrected extractor DS2 rows do not align with the frozen Phase 2 rows")

    model = joblib.load(checkpoint)
    feature_set = frozen["selected_candidate"]["feature_set"]
    if feature_set == "full_34":
        X = test_data["X_rf"]
    elif feature_set == "rr_only_4":
        X = test_data["X_rf"][:, :4]
    elif feature_set == "waveform_only_30":
        X = test_data["X_rf"][:, 4:]
    else:
        raise ValueError(f"Unknown frozen feature set: {feature_set}")

    # This is the sole new DS2 inference call. Do not rerun to change choices.
    selected_prob = model.predict_proba(X)
    selected_pred = model.classes_[selected_prob.argmax(axis=1)].astype(str)
    old_metrics = metrics(y, old_pred)
    selected_metrics = metrics(y, selected_pred)
    ds2_metrics_by_record = {}
    for record in sorted(np.unique(records), key=lambda x: int(x)):
        mask = records == record
        ds2_metrics_by_record[str(record)] = {
            "n_beats": int(mask.sum()),
            "metrics": metrics(y[mask], selected_pred[mask]),
        }
    record232 = ds2_metrics_by_record.get("232")
    if record232 is None:
        raise RuntimeError("Expected held-out record 232 is missing from the DS2 final evaluation")

    comparison = {
        "selected_minus_existing_macro_f1": selected_metrics["macro_f1"] - old_metrics["macro_f1"],
        "selected_minus_existing_accuracy": selected_metrics["accuracy"] - old_metrics["accuracy"],
        "predictions_identical": bool(np.array_equal(selected_pred, old_pred)),
        "per_class_f1_delta": {c: selected_metrics["per_class"][c]["f1"] - old_metrics["per_class"][c]["f1"] for c in LABELS},
        "per_class_recall_delta": {c: selected_metrics["per_class"][c]["recall"] - old_metrics["per_class"][c]["recall"] for c in LABELS},
    }
    results = {
        "evaluation": "single final DS2 evaluation after frozen DS1-only model/config",
        "ds2_used_for": "final evaluation and comparison only; no selection or tuning",
        "selected_candidate": frozen["selected_candidate"],
        "fit_records": frozen["fit_records"],
        "validation_records": frozen["validation_records"],
        "test_records": test_records,
        "n_test_beats": int(len(y)),
        "existing_phase2_random_forest": old_metrics,
        "selected_random_forest": selected_metrics,
        "selected_vs_existing": comparison,
        "record_232": record232,
        "record_metrics": ds2_metrics_by_record,
    }
    np.savez_compressed(
        pred_path, y_test=y, record_test=records, beat_number_test=beat_numbers,
        pred_existing_rf=old_pred, pred_selected_rf=selected_pred, proba_selected_rf=selected_prob,
    )
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("DS2 final evaluation completed once after DS1 freeze.")
    print("Existing RF:", old_metrics["macro_f1"], old_metrics["accuracy"])
    print("Selected RF:", selected_metrics["macro_f1"], selected_metrics["accuracy"])
    print("Record 232:", json.dumps(record232, indent=2))
    print("Outputs:", out)


if __name__ == "__main__":
    main()

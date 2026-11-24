"""DS1-only controlled RF experiment for the corrected Phase 2 task.

This program intentionally loads only the DS1 training/validation records.
It never reads DS2 predictions, labels, signals, or the DS2 record list.
After choosing the validation model it writes a frozen checkpoint/config;
DS2 evaluation is a separate, one-shot script.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipeline

PHASE2 = ROOT / "artifacts" / "phase2_corrected"
OUT = Path(__file__).resolve().parent
LABELS = list(pipeline.LABELS)
SEED = 42
N_ESTIMATORS = 300
EXISTING_WEIGHT = {"N": 1.0, "S": 5.0, "V": 1.0, "F": 8.0, "Q": 1.0}
EXPECTED_FIT_RECORDS = [
    "106", "108", "109", "112", "114", "115", "116", "121", "122", "124",
    "201", "203", "207", "209", "215", "220", "223", "230",
]
EXPECTED_VALIDATION_RECORDS = ["101", "118", "119", "205", "208"]

# Predeclared decision rule: select a DS1-validation candidate only if macro-F1
# rises by >=0.01, S F1 rises by >=0.02, N and V F1 each fall by no more than
# 0.02, and accuracy falls by no more than 0.02. F/Q are still reported, but
# their validation supports (383 and 4) are too small to use as stable gates.
MIN_MACRO_GAIN = 0.01
MIN_S_F1_GAIN = 0.02
MAX_N_V_F1_DROP = 0.02
MAX_ACCURACY_DROP = 0.02


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])}
            for c in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def feature_names() -> list[str]:
    pts = np.linspace(0, pipeline.WINDOW - 1, 30).round().astype(int)
    return ["rr_prev", "rr_next", "rr_prev / rr_local", "rr_next / rr_local"] + [
        f"wave[{int(point) - pipeline.PRE}]" for point in pts
    ]


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace("=", "_")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mitdb")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protected_outputs = ["frozen_model_config.json", "random_forest_frozen.joblib",
                         "ds1_results.json", "ds1_validation_predictions.npz"]
    existing_outputs = [name for name in protected_outputs if (args.out / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite completed DS1 study outputs in {args.out}: {existing_outputs}. "
            "Choose a new empty --out directory for a fresh reproduction."
        )
    pipeline.DATA_DIR = args.data_dir.resolve()

    # The approved DS1 split is frozen explicitly here. Do not read the Phase 2
    # config (which also contains DS2 metadata), nor any DS2 prediction fields.
    ds1_records = list(pipeline.DS1_RECORDS)
    data = pipeline.build_dataset(ds1_records)
    X, y, groups = data["X_rf"], data["y"], data["record"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    fit_idx, val_idx = next(split.split(X, y, groups))
    fit_records = sorted(set(groups[fit_idx].tolist()))
    validation_records = sorted(set(groups[val_idx].tolist()))
    if fit_records != EXPECTED_FIT_RECORDS or validation_records != EXPECTED_VALIDATION_RECORDS:
        raise RuntimeError("Generated DS1 fit/validation records differ from the approved Phase 2 split")
    if set(groups[fit_idx]) & set(groups[val_idx]):
        raise RuntimeError("DS1 fit/validation record overlap")
    y_fit, y_val = y[fit_idx], y[val_idx]
    X_fit, X_val = X[fit_idx], X[val_idx]
    if set(np.unique(y_fit)) != set(LABELS) or set(np.unique(y_val)) != set(LABELS):
        raise RuntimeError("The fixed DS1 split must contain all five classes in fit and validation")

    # Current model, used only as a DS1 validation benchmark and for feature
    # importance / probability diagnostics.
    current = joblib.load(PHASE2 / "random_forest.joblib")
    current_pred = current.predict(X_val).astype(str)
    current_prob = current.predict_proba(X_val)
    with np.load(PHASE2 / "predictions.npz", allow_pickle=False) as frozen:
        saved_val_y = frozen["y_validation"].astype(str)
        saved_val_records = frozen["record_validation"].astype(str)
        saved_val_pred = frozen["pred_rf_validation"].astype(str)
    if not (np.array_equal(y_val, saved_val_y) and np.array_equal(groups[val_idx], saved_val_records)
            and np.array_equal(current_pred, saved_val_pred)):
        raise RuntimeError("Current RF does not reproduce the saved Phase 2 DS1 validation predictions")
    baseline_metrics = compute_metrics(y_val, current_pred)
    baseline_params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": current.get_params()["max_depth"],
        "min_samples_leaf": current.get_params()["min_samples_leaf"],
        "class_weight": current.get_params()["class_weight"],
        "max_features": current.get_params()["max_features"],
        "random_state": SEED,
    }

    # Candidate grid is fixed before looking at any candidate outcome.
    weights = {
        "existing_custom": EXISTING_WEIGHT,
        "balanced": "balanced",
        "balanced_subsample": "balanced_subsample",
        "unweighted": None,
        "S10": {"N": 1.0, "S": 10.0, "V": 1.0, "F": 8.0, "Q": 1.0},
        "S20": {"N": 1.0, "S": 20.0, "V": 1.0, "F": 8.0, "Q": 1.0},
    }
    depths = [None, 24, 12]
    leaves = [1, 3, 5]
    candidates: list[dict] = []
    prediction_arrays: dict[str, np.ndarray] = {"y_validation": y_val, "record_validation": groups[val_idx]}

    def evaluate_candidate(name: str, feature_key: str, indices: np.ndarray, weight_name: str,
                           class_weight, depth, leaf) -> None:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=depth, min_samples_leaf=leaf,
            class_weight=class_weight, max_features="sqrt", n_jobs=-1, random_state=SEED,
        )
        model.fit(X_fit[:, indices], y_fit)
        pred = model.predict(X_val[:, indices]).astype(str)
        metrics = compute_metrics(y_val, pred)
        key = safe_name(name)
        prediction_arrays[f"pred_{key}"] = pred
        candidates.append({
            "name": name, "feature_set": feature_key,
            "params": {"n_estimators": N_ESTIMATORS, "max_depth": depth,
                       "min_samples_leaf": leaf, "class_weight": class_weight,
                       "max_features": "sqrt", "random_state": SEED},
            "metrics": metrics,
        })
        print(f"DS1 validation candidate {len(candidates)} complete: {name} macro-F1={metrics['macro_f1']:.5f} S-F1={metrics['per_class']['S']['f1']:.5f}", flush=True)
        del model

    full = np.arange(X.shape[1])
    # Compact, predeclared search: six weight alternatives at the original
    # structure; six depth/leaf alternatives with the original weights; and
    # four S-upweight/regularization interactions. This avoids an unnecessarily
    # broad repeated search against the same small validation set.
    specifications = []
    for weight_name, weight in weights.items():
        specifications.append((weight_name, weight, baseline_params["max_depth"], baseline_params["min_samples_leaf"]))
    for depth in (24, 12):
        for leaf in leaves:
            specifications.append(("existing_custom", EXISTING_WEIGHT, depth, leaf))
    for weight_name in ("S10", "S20"):
        for depth in (None, 12):
            specifications.append((weight_name, weights[weight_name], depth, 3))
    seen = set()
    for weight_name, weight, depth, leaf in specifications:
        signature = (json.dumps(weight, sort_keys=True), depth, leaf)
        if signature in seen:
            continue
        seen.add(signature)
        name = f"grid_{weight_name}_depth_{depth if depth is not None else 'None'}_leaf_{leaf}"
        evaluate_candidate(name, "full_34", full, weight_name, weight, depth, leaf)

    # Pre-specified feature-group ablations, using the existing RF settings.
    # These test feature family contribution; feature importance is descriptive
    # and correlated inputs can redistribute impurity importance.
    for feature_key, indices in (
        ("rr_only_4", np.arange(0, 4)),
        ("waveform_only_30", np.arange(4, 34)),
    ):
        name = f"feature_ablation_{feature_key}"
        evaluate_candidate(name, feature_key, indices, "existing_custom", EXISTING_WEIGHT,
                           baseline_params["max_depth"], baseline_params["min_samples_leaf"])

    baseline = baseline_metrics
    eligible = []
    for candidate in candidates:
        m = candidate["metrics"]
        qualifies = (
            m["macro_f1"] >= baseline["macro_f1"] + MIN_MACRO_GAIN
            and m["per_class"]["S"]["f1"] >= baseline["per_class"]["S"]["f1"] + MIN_S_F1_GAIN
            and m["per_class"]["N"]["f1"] >= baseline["per_class"]["N"]["f1"] - MAX_N_V_F1_DROP
            and m["per_class"]["V"]["f1"] >= baseline["per_class"]["V"]["f1"] - MAX_N_V_F1_DROP
            and m["accuracy"] >= baseline["accuracy"] - MAX_ACCURACY_DROP
        )
        candidate["passes_predeclared_selection_rule"] = bool(qualifies)
        if qualifies:
            eligible.append(candidate)
    # Select solely by DS1 validation macro-F1, then S F1, then accuracy.
    selected = max(eligible, key=lambda c: (c["metrics"]["macro_f1"], c["metrics"]["per_class"]["S"]["f1"], c["metrics"]["accuracy"], c["name"])) if eligible else None

    if selected is None:
        selected = {
            "name": "existing_phase2_random_forest_retained",
            "feature_set": "full_34",
            "params": baseline_params,
            "metrics": baseline_metrics,
            "passes_predeclared_selection_rule": False,
            "retained_existing": True,
        }
        selected_model = current
        decision = "No candidate met the predeclared improvement and non-degradation rule; retain the existing Phase 2 RF."
    else:
        feature_set = selected["feature_set"]
        selected_indices = full if feature_set == "full_34" else np.arange(0, 4) if feature_set == "rr_only_4" else np.arange(4, 34)
        selected_model = RandomForestClassifier(**selected["params"])
        selected_model.fit(X_fit[:, selected_indices], y_fit)
        decision = "Selected the highest DS1-validation macro-F1 candidate that passed the predeclared S-improvement and non-degradation rule."

    # Freeze the selected DS1 model and candidate record before any DS2 access.
    joblib.dump(selected_model, args.out / "random_forest_frozen.joblib")
    frozen_config = {
        "phase": "Phase 2 RF DS1-only improvement study",
        "task": "five-class AAMI offline/delayed classification",
        "labels": LABELS,
        "seed": SEED,
        "fit_records": fit_records,
        "validation_records": validation_records,
        "data_used_for_experiments": "DS1 fit and validation records only; no DS2 inputs, scores, or predictions read",
        "prediction_design": "offline/delayed; rr_next, future intervals in rr_local, and 110 post-peak samples",
        "window_samples": pipeline.WINDOW,
        "pre_samples": pipeline.PRE,
        "post_samples": pipeline.POST,
        "baseline_params": baseline_params,
        "baseline_validation_metrics": baseline_metrics,
        "predeclared_selection_rule": {
            "minimum_macro_f1_gain": MIN_MACRO_GAIN,
            "minimum_S_f1_gain": MIN_S_F1_GAIN,
            "maximum_N_or_V_f1_drop": MAX_N_V_F1_DROP,
            "maximum_accuracy_drop": MAX_ACCURACY_DROP,
            "ranking": "DS1 validation macro-F1, then S F1, then accuracy",
        },
        "selected_candidate": selected,
        "decision": decision,
        "ds2_status": "not evaluated yet; this configuration is frozen before final test evaluation",
        "candidate_count": len(candidates),
        "class_weights_tested": list(weights.keys()),
        "max_depth_values_tested": depths,
        "min_samples_leaf_values_tested": leaves,
        "feature_ablation_sets": ["existing full_34 checkpoint", "rr_only_4", "waveform_only_30"],
    }
    (args.out / "frozen_model_config.json").write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
    results = {
        "labels": LABELS,
        "fit_records": fit_records,
        "validation_records": validation_records,
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "decision": decision,
        "candidates": candidates,
        "ds1_s_distribution_by_record": [],
        "ds1_validation_metrics_by_record": [],
        "validation_s_probability_by_true_class": {},
        "feature_importance": [],
        "feature_group_importance": {},
    }
    for record in ds1_records:
        mask = groups == record
        counts = {c: int(np.sum(y[mask] == c)) for c in LABELS}
        results["ds1_s_distribution_by_record"].append({
            "record": record, "n_beats": int(mask.sum()), "class_counts": counts,
            "s_share": float(counts["S"] / mask.sum()),
        })
    baseline_pred = current_pred
    for record in validation_records:
        mask = groups[val_idx] == record
        results["ds1_validation_metrics_by_record"].append({
            "record": record, "n_beats": int(mask.sum()), **compute_metrics(y_val[mask], baseline_pred[mask]),
        })
    s_idx = list(current.classes_).index("S")
    for truth in LABELS:
        vals = current_prob[y_val == truth, s_idx]
        results["validation_s_probability_by_true_class"][truth] = {
            "support": int(len(vals)), "mean": float(vals.mean()),
            "quantiles_0_25_50_75_90_100": np.quantile(vals, [0, .25, .5, .75, .9, 1]).astype(float).tolist(),
        }
    names = feature_names()
    for index in np.argsort(current.feature_importances_)[::-1]:
        results["feature_importance"].append({"feature": names[int(index)], "importance": float(current.feature_importances_[index])})
    results["feature_group_importance"] = {
        "RR (4 features)": float(current.feature_importances_[:4].sum()),
        "waveform (30 features)": float(current.feature_importances_[4:].sum()),
    }
    (args.out / "ds1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "ds1_validation_predictions.npz", **prediction_arrays)

    # Flat tables for convenient audit and presentation.
    candidate_rows = []
    for cand in candidates:
        row = {"name": cand["name"], "feature_set": cand["feature_set"],
               "validation_macro_f1": cand["metrics"]["macro_f1"], "validation_accuracy": cand["metrics"]["accuracy"],
               "S_precision": cand["metrics"]["per_class"]["S"]["precision"],
               "S_recall": cand["metrics"]["per_class"]["S"]["recall"],
               "S_f1": cand["metrics"]["per_class"]["S"]["f1"],
               "N_f1": cand["metrics"]["per_class"]["N"]["f1"],
               "V_f1": cand["metrics"]["per_class"]["V"]["f1"],
               "F_f1": cand["metrics"]["per_class"]["F"]["f1"],
               "Q_f1": cand["metrics"]["per_class"]["Q"]["f1"],
               "passes_selection_rule": cand["passes_predeclared_selection_rule"], **cand["params"]}
        row["class_weight"] = json.dumps(row["class_weight"], sort_keys=True)
        candidate_rows.append(row)
    write_csv(args.out / "ds1_candidate_metrics.csv", candidate_rows)
    write_csv(args.out / "ds1_s_distribution_by_record.csv", [
        {"record": r["record"], "n_beats": r["n_beats"], **r["class_counts"], "s_share": r["s_share"]}
        for r in results["ds1_s_distribution_by_record"]
    ])
    val_rows = []
    for record in results["ds1_validation_metrics_by_record"]:
        row = {"record": record["record"], "n_beats": record["n_beats"],
               "accuracy": record["accuracy"], "macro_f1": record["macro_f1"]}
        for c in LABELS:
            row.update({f"{c}_{key}": value for key, value in record["per_class"][c].items()})
        val_rows.append(row)
    write_csv(args.out / "ds1_validation_record_metrics.csv", val_rows)
    write_csv(args.out / "current_rf_feature_importance.csv", results["feature_importance"])

    # DS1-only diagnostic figures.
    fig, ax = plt.subplots(figsize=(11, 5))
    sr = results["ds1_s_distribution_by_record"]
    ax.bar([r["record"] for r in sr], [r["class_counts"]["S"] for r in sr], color="#2878a0")
    ax.set(title="DS1 S-class count by record", xlabel="DS1 record", ylabel="Eligible S beats")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(args.out / "ds1_s_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for truth in LABELS:
        vals = current_prob[y_val == truth, s_idx]
        ax.hist(vals, bins=np.linspace(0, 1, 31), density=True, histtype="step", linewidth=1.7,
                label=f"true {truth} (n={len(vals)})")
    ax.set(title="Current RF validation probability assigned to S — DS1 only",
           xlabel="Predicted probability for S", ylabel="Density")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(args.out / "current_rf_s_probability_ds1_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    top_importance = results["feature_importance"][:15]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([item["feature"] for item in top_importance[::-1]],
            [item["importance"] for item in top_importance[::-1]], color="#2878a0")
    ax.set(title="Current Phase 2 RF feature importance — DS1 fit", xlabel="Impurity-based importance")
    ax.grid(axis="x", alpha=.25)
    fig.savefig(args.out / "current_rf_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    val_s = results["ds1_validation_metrics_by_record"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(val_s))
    supports = [r["per_class"]["S"]["support"] for r in val_s]
    true_positives = [r["per_class"]["S"]["recall"] * r["per_class"]["S"]["support"] for r in val_s]
    ax.bar(x - .18, supports, width=.36, label="True S support", color="#9ab4c9")
    ax.bar(x + .18, true_positives, width=.36, label="Correct S predictions", color="#d66b46")
    ax.set(xticks=x, xticklabels=[r["record"] for r in val_s], xlabel="DS1 validation record", ylabel="Beat count",
           title="Current RF S-class validation support and true positives")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.savefig(args.out / "current_rf_validation_s_by_record.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for candidate in candidates:
        metrics = candidate["metrics"]
        feature = candidate["feature_set"]
        ax.scatter(metrics["per_class"]["S"]["f1"], metrics["macro_f1"],
                   marker="s" if feature != "full_34" else "o", alpha=.75)
        ax.annotate(candidate["name"], (metrics["per_class"]["S"]["f1"], metrics["macro_f1"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.scatter(baseline_metrics["per_class"]["S"]["f1"], baseline_metrics["macro_f1"],
               marker="*", s=180, color="black", label="Current Phase 2 RF")
    ax.set(xlabel="DS1 validation S F1", ylabel="DS1 validation macro-F1",
           title="DS1-only candidate tradeoff (no DS2 used)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(args.out / "ds1_candidate_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("DS1 fit records:", fit_records)
    print("DS1 validation records:", validation_records)
    print("Baseline validation:", json.dumps(baseline_metrics, indent=2))
    print("Candidates evaluated:", len(candidates))
    print("Passing predeclared rule:", len(eligible))
    print("Frozen decision:", decision)
    print("Selected:", selected["name"], "macro-F1", selected["metrics"]["macro_f1"], "S F1", selected["metrics"]["per_class"]["S"]["f1"])
    print("Saved frozen DS1 model/config to:", args.out.resolve())


if __name__ == "__main__":
    main()

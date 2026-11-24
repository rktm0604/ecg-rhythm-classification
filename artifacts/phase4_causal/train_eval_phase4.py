"""Strictly causal ECG experiment. No feature uses samples/annotations after R.

Run from project root with .venv/Scripts/python.exe
artifacts/phase4_causal/train_eval_phase4.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipeline

OUT = Path(__file__).resolve().parent
SEED = 42
PRE, POST, WINDOW = 199, 1, 200  # [R-199, R+1): includes R, no later sample
LABELS = pipeline.LABELS


def causal_record_features(signal, positions, symbols):
    """Extract records causally at R peak; returns rows and exclusion counts.

    RR endpoints are consecutive valid AAMI heartbeat symbols only. rr_local
    is the mean of up to four intervals ending at/before the current R peak.
    """
    signal = np.asarray(signal, dtype=np.float32)
    positions = np.asarray(positions, dtype=int)
    symbols = np.asarray(symbols)
    if len(positions) != len(symbols):
        raise ValueError("Annotation positions and symbols must have equal length")
    hb_indices = [i for i, s in enumerate(symbols) if str(s) in pipeline.LABEL_MAP]
    hb_positions = positions[hb_indices]
    intervals = np.diff(hb_positions).astype(float)
    rows, excluded = [], {}

    def exclude(label, reason):
        key = f"{label}:{reason}"
        excluded[key] = excluded.get(key, 0) + 1

    for j, ann_i in enumerate(hb_indices):
        label = pipeline.LABEL_MAP[str(symbols[ann_i])]
        if j == 0:
            exclude(label, "missing_previous_heartbeat")
            continue
        peak = int(hb_positions[j])
        start, end = peak - PRE, peak + POST
        if start < 0 or end > len(signal):
            exclude(label, "causal_window_boundary")
            continue
        raw = signal[start:end]
        if len(raw) != WINDOW:
            exclude(label, "invalid_window_length")
            continue
        if not np.isfinite(raw).all():
            exclude(label, "non_finite_waveform")
            continue
        rr_prev = float(intervals[j - 1])
        local_intervals = intervals[max(0, j - 4):j]
        if not len(local_intervals):
            exclude(label, "missing_previous_rr_context")
            continue
        rr_local = float(np.mean(local_intervals))
        if not np.isfinite([rr_prev, rr_local]).all() or rr_prev <= 0 or rr_local <= 0:
            exclude(label, "invalid_rr_interval")
            continue
        wave = pipeline._safe_zscore(raw)
        points = np.linspace(0, WINDOW - 1, 30).round().astype(int)
        # Explicitly causal feature vector: no rr_next.
        rf_features = np.asarray([rr_prev, rr_local, rr_prev / rr_local, *wave[points]], dtype=np.float32)
        rows.append({"label": label, "wave": wave, "features": rf_features,
                     "rr_prev": rr_prev, "rr_local": rr_local,
                     "sample": peak, "annotation_index": int(ann_i)})
    return rows, excluded


def load_causal_dataset(records):
    all_rows, records_out, beat_num, by_record, excluded_by_record = [], [], [], {}, {}
    for record in records:
        record = str(record)
        if record in pipeline.PACED_RECORDS:
            continue
        signal, positions, symbols = pipeline._read_record(record)
        rows, excluded = causal_record_features(signal, positions, symbols)
        by_record[record] = len(rows)
        excluded_by_record[record] = excluded
        for row in rows:
            all_rows.append(row)
            records_out.append(record)
            beat_num.append(row["annotation_index"])
        print(f"record {record}: {len(rows):,} causal beats", flush=True)
    return {
        "X_rf": np.stack([r["features"] for r in all_rows]),
        "X_wave": np.stack([r["wave"] for r in all_rows]),
        "y": np.asarray([r["label"] for r in all_rows], dtype="U1"),
        "record": np.asarray(records_out, dtype="U3"),
        "beat_number": np.asarray(beat_num, dtype=np.int32),
        "beats_per_record": by_record,
        "excluded_by_record": excluded_by_record,
    }


def metrics(y, pred):
    return {"macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "report": classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y, pred, labels=LABELS).tolist()}


class CausalECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(128, len(LABELS)))

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(w_fit, y_fit, w_val, y_val):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_id = {v: i for i, v in enumerate(LABELS)}
    yi = np.asarray([to_id[v] for v in y_fit], dtype=np.int64)
    yvi = np.asarray([to_id[v] for v in y_val], dtype=np.int64)
    cw = compute_class_weight("balanced", classes=np.arange(5), y=yi)
    model = CausalECGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))

    def make_loader(w, y, shuffle):
        return DataLoader(TensorDataset(torch.tensor(w[:, None, :], dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=256, shuffle=shuffle)

    train_loader, val_loader = make_loader(w_fit, yi, True), make_loader(w_val, yvi, False)
    best, best_epoch, state, stale = -1.0, 0, None, 0
    for ep in range(35):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        model.eval(); pred, truth = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred.extend(model(xb.to(device)).argmax(1).cpu().numpy()); truth.extend(yb.numpy())
        score = f1_score(truth, pred, labels=list(range(5)), average="macro", zero_division=0)
        print(f"Causal CNN epoch {ep+1}/35 validation macro-F1={score:.5f}", flush=True)
        if score > best:
            best, best_epoch, stale = float(score), ep + 1, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 7: break
    model.load_state_dict(state); model.eval()

    def prob(w):
        batches = []
        with torch.no_grad():
            for s in range(0, len(w), 256):
                xb = torch.tensor(w[s:s+256, None, :], dtype=torch.float32, device=device)
                batches.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        return np.concatenate(batches)
    return model, prob(w_val), prob, best, best_epoch, device.type, cw.tolist()


def main():
    cfg2 = json.loads((ROOT / "artifacts" / "phase2_corrected" / "experiment_config.json").read_text(encoding="utf-8"))
    ds1, ds2 = pipeline.check_record_split()
    if ds1 != cfg2["train_records"] or ds2 != cfg2["test_records"]:
        raise RuntimeError("Phase 2 record lists differ; stop to preserve matched comparison")
    data = load_causal_dataset(ds1 + ds2)
    train_mask, test_mask = np.isin(data["record"], ds1), np.isin(data["record"], ds2)
    X, W, y, g = data["X_rf"][train_mask], data["X_wave"][train_mask], data["y"][train_mask], data["record"][train_mask]
    Xt, Wt, yt = data["X_rf"][test_mask], data["X_wave"][test_mask], data["y"][test_mask]
    fit_records, val_records = cfg2["fit_records"], cfg2["validation_records"]
    fi, vi = np.isin(g, fit_records), np.isin(g, val_records)
    if set(np.unique(y[fi])) != set(LABELS) or set(np.unique(y[vi])) != set(LABELS) or set(np.unique(yt)) != set(LABELS):
        raise RuntimeError("Causal split lacks one or more AAMI classes")
    if set(np.unique(g[fi])) != set(fit_records) or set(np.unique(g[vi])) != set(val_records):
        raise RuntimeError("Causal extraction omitted a configured fit/validation record")
    yfit, yval = y[fi], y[vi]

    # RF tuning identical candidate space; select on DS1 validation only.
    candidates = []
    for depth in (None, 24):
        for leaf in (1, 3):
            for cw in ("balanced", "balanced_subsample", {"N": 1., "S": 5., "V": 1., "F": 8., "Q": 1.}):
                model = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                               class_weight=cw, n_jobs=-1, random_state=SEED)
                model.fit(X[fi], yfit); pv = model.predict(X[vi])
                candidates.append((float(f1_score(yval, pv, labels=LABELS, average="macro", zero_division=0)), model, pv))
                print(f"RF candidate depth={depth}, leaf={leaf}: val macro-F1={candidates[-1][0]:.5f}", flush=True)
    best_rf_score, rf, rf_val_pred = max(candidates, key=lambda x: x[0])

    majority = DummyClassifier(strategy="most_frequent").fit(X[fi], yfit)
    rr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    rr.fit(X[fi, :3], yfit)
    cnn, cnn_val_prob, cnn_prob, cnn_val_score, cnn_epoch, device, cnn_cw = train_cnn(W[fi], yfit, W[vi], yval)
    cnn_val_pred = np.asarray(LABELS)[cnn_val_prob.argmax(1)]

    # DS2 is touched only after RF/CNN choices are frozen.
    ytest = yt
    rf_test_pred = rf.predict(Xt); rr_test_pred = rr.predict(Xt[:, :3]); maj_test_pred = majority.predict(Xt)
    rf_test_prob = rf.predict_proba(Xt); cnn_test_prob = cnn_prob(Wt); cnn_test_pred = np.asarray(LABELS)[cnn_test_prob.argmax(1)]
    val_results = {"Random Forest": metrics(yval, rf_val_pred), "1D-CNN": metrics(yval, cnn_val_pred),
                   "Majority baseline": metrics(yval, majority.predict(X[vi])), "RR-only Logistic Regression": metrics(yval, rr.predict(X[vi, :3]))}
    test_results = {"Random Forest": metrics(ytest, rf_test_pred), "1D-CNN": metrics(ytest, cnn_test_pred),
                    "Majority baseline": metrics(ytest, maj_test_pred), "RR-only Logistic Regression": metrics(ytest, rr_test_pred)}
    joblib.dump(rf, OUT / "random_forest.joblib"); joblib.dump(rr, OUT / "rr_logistic_baseline.joblib")
    torch.save({"state_dict": cnn.state_dict(), "labels": LABELS, "window_samples": 200, "causal": True, "best_epoch": cnn_epoch}, OUT / "cnn.pt")
    np.savez_compressed(OUT / "predictions.npz", y_validation=yval, record_validation=g[vi],
        pred_rf_validation=rf_val_pred, pred_cnn_validation=cnn_val_pred, y_test=ytest, record_test=data["record"][test_mask],
        beat_number_test=data["beat_number"][test_mask], pred_rf=rf_test_pred, pred_cnn=cnn_test_pred,
        pred_majority=maj_test_pred, pred_rr=rr_test_pred, proba_rf=rf_test_prob, proba_cnn=cnn_test_prob)
    config = {"seed": SEED, "labels": LABELS, "train_records": ds1, "test_records": ds2,
        "fit_records": fit_records, "validation_records": val_records,
        "split_protocol": "Exact same DS1 fit, validation, and DS2 records as Phase 2",
        "selection_data": "DS1 validation only; DS2 used once for final reporting",
        "prediction_time": "annotated R-peak; causal waveform contains samples [R-199, R] inclusive",
        "window_samples": WINDOW, "samples_before_R": PRE, "samples_after_R": 0,
        "rr_prev": "distance from previous valid mapped heartbeat to current R peak",
        "rr_local": "mean of up to four preceding valid-heartbeat intervals; no future intervals",
        "rr_next": "not computed",
        "features_rf": ["rr_prev", "rr_local", "rr_prev/rr_local", "30 evenly spaced causal waveform samples"],
        "feature_timing_doc": str((OUT / "feature_timing.md").name),
        "fit_counts": {c: int(np.sum(yfit == c)) for c in LABELS}, "validation_counts": {c: int(np.sum(yval == c)) for c in LABELS},
        "test_counts": {c: int(np.sum(ytest == c)) for c in LABELS}, "all_counts": {c: int(np.sum(data["y"] == c)) for c in LABELS},
        "beats_per_record": data["beats_per_record"], "exclusions_by_record": data["excluded_by_record"],
        "rf_selected_params": {k: rf.get_params()[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state")},
        "rf_candidate_validation_macro_f1": [x[0] for x in candidates], "cnn_best_epoch": cnn_epoch, "cnn_device": device,
        "cnn_fit_class_weights": cnn_cw, "cnn_validation_macro_f1": cnn_val_score}
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"validation": val_results, "final_test": test_results}, indent=2), encoding="utf-8")
    phase2 = json.loads((ROOT / "artifacts" / "phase2_corrected" / "results.json").read_text(encoding="utf-8"))["final_test"]
    comparison = {}
    for name in test_results:
        off, causal = phase2[name], test_results[name]
        comparison[name] = {
            "phase2_offline_macro_f1": off["macro_f1"], "phase4_causal_macro_f1": causal["macro_f1"],
            "macro_f1_difference_causal_minus_offline": causal["macro_f1"] - off["macro_f1"],
            "phase2_offline_accuracy": off["accuracy"], "phase4_causal_accuracy": causal["accuracy"],
            "accuracy_difference_causal_minus_offline": causal["accuracy"] - off["accuracy"],
            "per_class": {c: {"phase2_f1": off["report"][c]["f1-score"],
                               "phase4_f1": causal["report"][c]["f1-score"],
                               "f1_difference": causal["report"][c]["f1-score"] - off["report"][c]["f1-score"],
                               "phase2_recall": off["report"][c]["recall"],
                               "phase4_recall": causal["report"][c]["recall"],
                               "recall_difference": causal["report"][c]["recall"] - off["report"][c]["recall"],
                               "support_phase2": int(off["report"][c]["support"]),
                               "support_phase4": int(causal["report"][c]["support"])} for c in LABELS},
        }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    lines = ["# Phase 2 offline/delayed vs Phase 4 causal comparison", "",
             "These are related but different prediction tasks. Phase 2 uses a 200-sample waveform extending 110 samples after R and future RR intervals. Phase 4 predicts at R using only samples R-199 through R and past RR context. Metrics are descriptive comparisons on the same DS2 records; DS2 was not used for selection.", "",
             "| Model | P2 macro-F1 | P4 macro-F1 | Difference | P2 accuracy | P4 accuracy | Difference |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in comparison.items():
        lines.append(f"| {name} | {item['phase2_offline_macro_f1']:.4f} | {item['phase4_causal_macro_f1']:.4f} | {item['macro_f1_difference_causal_minus_offline']:+.4f} | {item['phase2_offline_accuracy']:.4f} | {item['phase4_causal_accuracy']:.4f} | {item['accuracy_difference_causal_minus_offline']:+.4f} |")
    lines.extend(["", "## Per-class differences", "", "F1 and recall differences are Phase 4 minus Phase 2. Supports are the same DS2 annotations when both experiments retain the same beats; any support difference is reported below.", ""])
    for name, item in comparison.items():
        lines.extend([f"### {name}", "", "| Class | Support P2 / P4 | F1 P2 | F1 P4 | ΔF1 | Recall P2 | Recall P4 | ΔRecall |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for c in LABELS:
            m = item["per_class"][c]
            lines.append(f"| {c} | {m['support_phase2']} / {m['support_phase4']} | {m['phase2_f1']:.4f} | {m['phase4_f1']:.4f} | {m['f1_difference']:+.4f} | {m['phase2_recall']:.4f} | {m['phase4_recall']:.4f} | {m['recall_difference']:+.4f} |")
    lines.extend(["", "## Rare class limitations", "", f"Phase 4 fit supports: F={config['fit_counts']['F']}, Q={config['fit_counts']['Q']}. DS2 supports: F={config['test_counts']['F']}, Q={config['test_counts']['Q']}. These small training and test supports make F/Q estimates unstable; zero scores should not be interpreted as precise estimates of population performance.", ""])
    (OUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Counts", config["all_counts"])
    print("RF val macro-F1", best_rf_score, "CNN val macro-F1", cnn_val_score, "epoch", cnn_epoch)
    print("DS2 final", json.dumps({k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in test_results.items()}, indent=2))
    print("Output directory", OUT.resolve())


if __name__ == "__main__":
    main()

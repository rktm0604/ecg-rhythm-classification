"""Matched RF/CNN Phase 2 experiment using corrected extraction.

Run from project root with the existing environment:
  .venv/Scripts/python.exe artifacts/phase2_corrected/train_eval_phase2.py
All generated experiment files remain in this directory.
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import LABELS, build_dataset, check_record_split

SEED = 42
OUT = Path(__file__).resolve().parent


def scores(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, len(LABELS)))

    def forward(self, x):
        return self.head(self.features(x))


def fit_cnn(wave_fit, y_fit, wave_val, y_val, epochs=35, batch_size=256):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_id = {label: i for i, label in enumerate(LABELS)}
    y_fit_id = np.asarray([to_id[x] for x in y_fit], dtype=np.int64)
    y_val_id = np.asarray([to_id[x] for x in y_val], dtype=np.int64)
    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_fit_id)
    model = ECGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    def loader(waves, labels, shuffle):
        x = torch.tensor(waves[:, None, :], dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    fit_loader = loader(wave_fit, y_fit_id, True)
    val_loader = loader(wave_val, y_val_id, False)
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        predictions, truth = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                predictions.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                truth.extend(yb.numpy())
        score = f1_score(truth, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        print(f"CNN epoch {epoch + 1}/{epochs}: DS1 validation macro-F1={score:.5f}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = float(score), epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 7:
                break
    if best_state is None:
        raise RuntimeError("CNN produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    def predict_probs(waves):
        batches = []
        with torch.no_grad():
            for start in range(0, len(waves), batch_size):
                xb = torch.tensor(waves[start:start + batch_size, None, :], dtype=torch.float32, device=device)
                batches.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    val_prob = predict_probs(wave_val)
    return model, val_prob, best_score, best_epoch, device.type, class_weights.tolist(), predict_probs


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_records, test_records = check_record_split()
    data = build_dataset(train_records + test_records)
    train_mask = np.isin(data["record"], train_records)
    test_mask = np.isin(data["record"], test_records)
    X, W, y, groups = data["X_rf"][train_mask], data["X_wave"][train_mask], data["y"][train_mask], data["record"][train_mask]
    X_test, W_test, y_test = data["X_rf"][test_mask], data["X_wave"][test_mask], data["y"][test_mask]
    test_record, test_beat = data["record"][test_mask], data["beat_number"][test_mask]
    if set(groups) & set(data["record"][test_mask]):
        raise RuntimeError("Record overlap detected")
    if set(np.unique(y)) != set(LABELS) or set(np.unique(y_test)) != set(LABELS):
        raise RuntimeError("Corrected extraction/split does not contain all five classes")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X, y, groups))
    fit_records = sorted(set(groups[fit_idx].tolist()))
    val_records = sorted(set(groups[val_idx].tolist()))
    if set(fit_records) & set(val_records):
        raise RuntimeError("DS1 fit/validation record overlap")
    y_fit, y_val = y[fit_idx], y[val_idx]
    if set(np.unique(y_fit)) != set(LABELS) or set(np.unique(y_val)) != set(LABELS):
        raise RuntimeError("DS1 fit/validation split lacks a class; stop before training")

    # DS1-only baselines and RF candidate selection.
    majority = DummyClassifier(strategy="most_frequent").fit(X[fit_idx], y_fit)
    rr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    rr.fit(X[fit_idx, :4], y_fit)
    candidates = []
    for depth in (None, 24):
        for leaf in (1, 3):
            for weight in ("balanced", "balanced_subsample", {"N": 1.0, "S": 5.0, "V": 1.0, "F": 8.0, "Q": 1.0}):
                model = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                               class_weight=weight, n_jobs=-1, random_state=SEED)
                model.fit(X[fit_idx], y_fit)
                val_pred = model.predict(X[val_idx])
                candidates.append({"model": model, "params": model.get_params(),
                                   "validation_macro_f1": float(f1_score(y_val, val_pred, labels=LABELS, average="macro", zero_division=0)),
                                   "validation_predictions": val_pred})
                print(f"RF candidate depth={depth} leaf={leaf} weight={weight}: validation macro-F1={candidates[-1]['validation_macro_f1']:.5f}", flush=True)
    best_rf = max(candidates, key=lambda item: item["validation_macro_f1"])
    rf_model = best_rf["model"]

    cnn_model, cnn_val_prob, cnn_val_score, cnn_epoch, cnn_device, cnn_weights, cnn_predict = fit_cnn(
        W[fit_idx], y_fit, W[val_idx], y_val
    )
    cnn_val_pred = np.asarray(LABELS)[cnn_val_prob.argmax(axis=1)]

    # Freeze all choices before first inference on DS2.
    rf_pred = rf_model.predict(X_test)
    rr_pred = rr.predict(X_test[:, :4])
    majority_pred = majority.predict(X_test)
    cnn_test_prob = cnn_predict(W_test)
    cnn_test_pred = np.asarray(LABELS)[cnn_test_prob.argmax(axis=1)]

    validation = {
        "Random Forest": scores(y_val, best_rf["validation_predictions"]),
        "1D-CNN": scores(y_val, cnn_val_pred),
        "Majority baseline": scores(y_val, majority.predict(X[val_idx])),
        "RR-only Logistic Regression": scores(y_val, rr.predict(X[val_idx, :4])),
    }
    final = {
        "Random Forest": scores(y_test, rf_pred),
        "1D-CNN": scores(y_test, cnn_test_pred),
        "Majority baseline": scores(y_test, majority_pred),
        "RR-only Logistic Regression": scores(y_test, rr_pred),
    }

    joblib.dump(rf_model, OUT / "random_forest.joblib")
    joblib.dump(rr, OUT / "rr_logistic_baseline.joblib")
    torch.save({"state_dict": cnn_model.state_dict(), "labels": LABELS, "window_samples": 200,
                "best_epoch": cnn_epoch}, OUT / "cnn.pt")
    np.savez_compressed(
        OUT / "predictions.npz", y_validation=y_val, record_validation=groups[val_idx],
        pred_rf_validation=best_rf["validation_predictions"], pred_cnn_validation=cnn_val_pred,
        y_test=y_test, record_test=test_record, beat_number_test=test_beat,
        pred_rf=rf_pred, pred_cnn=cnn_test_pred, pred_majority=majority_pred, pred_rr=rr_pred,
        proba_cnn=cnn_test_prob, proba_rf=rf_model.predict_proba(X_test),
    )
    config = {
        "seed": SEED, "labels": LABELS, "split_protocol": "DS1 GroupShuffleSplit 80/20 by record; same fit/validation rows for all models",
        "train_records": train_records, "validation_records": val_records, "fit_records": fit_records,
        "test_records": test_records, "excluded_same_subject_records": ["202"],
        "rr_local": "mean of up to two preceding and two following intervals between valid mapped heartbeat positions",
        "prediction_design": "offline/delayed; rr_next, future intervals in rr_local, and 110 post-peak samples",
        "window_samples": 200, "pre_samples": 90, "post_samples": 110,
        "fit_class_counts": {c: int(np.sum(y_fit == c)) for c in LABELS},
        "validation_class_counts": {c: int(np.sum(y_val == c)) for c in LABELS},
        "test_class_counts": {c: int(np.sum(y_test == c)) for c in LABELS},
        "all_corrected_counts": {c: int(np.sum(data["y"] == c)) for c in LABELS},
        "beats_per_record": data["beats_per_record"],
        "rf_selected_params": {k: v for k, v in best_rf["params"].items() if k in ("n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state")},
        "rf_candidates": [{"params": {k: v for k, v in c["params"].items() if k in ("max_depth", "min_samples_leaf", "class_weight")}, "validation_macro_f1": c["validation_macro_f1"]} for c in candidates],
        "cnn_best_epoch": cnn_epoch, "cnn_device": cnn_device, "cnn_fit_class_weights": cnn_weights,
        "selection_data": "DS1 validation only; DS2 used once for final reporting",
    }
    results = {"config": config, "validation": validation, "final_test": final,
               "rf_validation_macro_f1": best_rf["validation_macro_f1"],
               "cnn_validation_macro_f1": cnn_val_score}
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nDS1 fit records:", fit_records)
    print("DS1 validation records:", val_records)
    print("DS2 test records:", test_records)
    print("Corrected counts:", config["all_corrected_counts"])
    print("Best RF validation macro-F1:", best_rf["validation_macro_f1"])
    print("Best CNN validation macro-F1:", cnn_val_score, "epoch", cnn_epoch)
    print("DS2 final:", json.dumps({k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in final.items()}, indent=2))
    print("Outputs:", OUT.resolve())


if __name__ == "__main__":
    main()

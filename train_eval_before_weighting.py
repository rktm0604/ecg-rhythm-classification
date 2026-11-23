"""Train approved five-class RF/CNN models and evaluate once on DS2.

Run: python train_eval.py
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, make_scorer)
from sklearn.utils.class_weight import compute_class_weight
from pipeline import (build_dataset, DS1_RECORDS, DS2_RECORDS, LABELS,
                      check_record_split, EXCLUDED_FOR_SUBJECT_OVERLAP)

SEED = 42
def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))

def report_dict(y, pred):
    return classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)

def train_cnn(Xtr, ytr, groups, Xtest, epochs=35, batch_size=256):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN. Install requirements.txt.") from exc

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = LABELS
    to_id = {c:i for i,c in enumerate(classes)}
    yids = np.asarray([to_id[v] for v in ytr], dtype=np.int64)

    # Patient/record-grouped validation split within DS1 only.
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(Xtr, yids, groups))
    if len(set(yids[fit_idx])) < len(classes):
        raise RuntimeError("CNN training partition lacks one or more classes. Review DS1 split.")
    if len(set(yids[val_idx])) < len(classes):
        print("WARNING: CNN validation partition does not contain every class.")

    class_weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=yids[fit_idx])
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    class ECGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(128, len(classes)))
        def forward(self, x): return self.head(self.features(x))

    def loader(indices, shuffle):
        xx = torch.tensor(Xtr[indices, None, :], dtype=torch.float32)
        yy = torch.tensor(yids[indices], dtype=torch.long)
        return DataLoader(TensorDataset(xx, yy), batch_size=batch_size, shuffle=shuffle)
    model = ECGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state, best_val, patience = None, -1, 7
    stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader(fit_idx, True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for xb, yb in loader(val_idx, False):
                logits = model(xb.to(device))
                preds.extend(logits.argmax(1).cpu().numpy())
                truth.extend(yb.numpy())
        score = f1_score(truth, preds, labels=list(range(len(classes))), average="macro", zero_division=0)
        print(f"CNN epoch {epoch+1:02d}/{epochs} · validation macro-F1={score:.4f}")
        if score > best_val:
            best_val, stale = score, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a validation checkpoint.")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xtest[:, None, :], dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return model, probs, best_val, device.type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--skip-cnn", action="store_true", help="Run RF/baselines only; not a complete Phase 2 comparison.")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds1, ds2 = check_record_split()
    print("DS1 records:", ds1)
    print("DS2 records:", ds2)
    print("Excluded from DS2 for known same-subject overlap:", sorted(EXCLUDED_FOR_SUBJECT_OVERLAP))
    data = build_dataset(ds1 + ds2)
    train_mask = np.isin(data["record"], ds1)
    test_mask = np.isin(data["record"], ds2)
    Xtr, Xte = data["X_rf"][train_mask], data["X_rf"][test_mask]
    Wtr, Wte = data["X_wave"][train_mask], data["X_wave"][test_mask]
    ytr, yte = data["y"][train_mask], data["y"][test_mask]
    gtr, gte = data["record"][train_mask], data["record"][test_mask]
    rec_test = data["record"][test_mask]
    beat_test = data["beat_number"][test_mask]
    assert not set(gtr) & set(gte)
    assert set(np.unique(ytr)) == set(LABELS), f"DS1 missing classes: {set(LABELS)-set(ytr)}"
    assert set(np.unique(yte)) == set(LABELS), f"DS2 missing classes: {set(LABELS)-set(yte)}"

    # Baseline 1: majority class, fitted only on DS1.
    majority = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    base_majority = majority.predict(Xte)
    # Baseline 2: Logistic Regression on the four RR features only.
    rr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED))
    rr.fit(Xtr[:, :4], ytr)
    base_rr = rr.predict(Xte[:, :4])

    # RF tuning uses grouped CV within DS1 only; DS2 is not touched.
    n_splits = min(3, len(np.unique(gtr)))
    cv = GroupKFold(n_splits=n_splits)
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=SEED),
        {"max_depth": [None, 24], "min_samples_leaf": [1, 3]},
        scoring=make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0),
        cv=cv, n_jobs=-1, refit=True, verbose=1)
    rf.fit(Xtr, ytr, groups=gtr)
    rf_model = rf.best_estimator_
    rf_probs = rf_model.predict_proba(Xte)
    rf_pred = rf_model.classes_[rf_probs.argmax(axis=1)]
    joblib.dump(rf_model, out / "random_forest.joblib")
    joblib.dump(rr, out / "rr_logistic_baseline.joblib")

    results = {
        "labels": LABELS,
        "train_records": sorted(set(gtr.tolist())),
        "test_records": sorted(set(gte.tolist())),
        "excluded_same_subject_records": sorted(EXCLUDED_FOR_SUBJECT_OVERLAP),
        "window_samples": 200,
        "window_pre_samples": 90,
        "window_post_samples": 110,
        "primary_metric": "macro_f1",
        "best_rf_params": rf.best_params_,
        "rf_validation_macro_f1": float(rf.best_score_),
        "models": {},
        "class_counts_all": {c:int((data["y"] == c).sum()) for c in LABELS},
        "beats_per_record": data["beats_per_record"],
        "mean_wave": {c:data["X_wave"][data["y"] == c].mean(0).tolist() for c in LABELS},
        "std_wave": {c:data["X_wave"][data["y"] == c].std(0).tolist() for c in LABELS},
    }
    models = {
        "Majority baseline": base_majority,
        "RR-only Logistic Regression": base_rr,
        "Random Forest": rf_pred,
    }
    probas = {"Random Forest": rf_probs}
    if not args.skip_cnn:
        cnn, cnn_probs, cnn_val, device = train_cnn(Wtr, ytr, gtr, Wte, epochs=args.epochs)
        cnn_pred = np.asarray(LABELS)[cnn_probs.argmax(axis=1)]
        import torch
        torch.save({"state_dict": cnn.state_dict(), "labels": LABELS, "window_samples": 200}, out / "cnn.pt")
        models["1D-CNN"] = cnn_pred
        probas["1D-CNN"] = cnn_probs
        results["cnn_validation_macro_f1"] = float(cnn_val)
        results["cnn_device"] = device

    for name, pred in models.items():
        results["models"][name] = {
            "macro_f1": macro_f1(yte, pred),
            "accuracy": float(accuracy_score(yte, pred)),
            "report": report_dict(yte, pred),
            "confusion": confusion_matrix(yte, pred, labels=LABELS).tolist(),
        }
    results["best_baseline_macro_f1"] = max(
        results["models"]["Majority baseline"]["macro_f1"],
        results["models"]["RR-only Logistic Regression"]["macro_f1"])
    results["success_margin_required"] = 0.05
    results["best_model_macro_f1"] = max(v["macro_f1"] for k,v in results["models"].items() if k not in ("Majority baseline", "RR-only Logistic Regression"))
    results["improvement_over_best_baseline"] = results["best_model_macro_f1"] - results["best_baseline_macro_f1"]
    results["success_criterion_met"] = results["improvement_over_best_baseline"] >= 0.05

    # Store test-only arrays for transparent dashboard error analysis.
    np.savez_compressed(
        out / "test_data.npz", X_rf=Xte, X_wave=Wte, y=yte, pred_rf=rf_pred,
        record=rec_test, beat_number=beat_test,
        **{f"proba_{name.replace(' ','_').replace('-','_')}":p for name,p in probas.items()})
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nFINAL DS2 RESULTS")
    for name, item in results["models"].items():
        print(f"{name}: macro-F1={item['macro_f1']:.4f}, accuracy={item['accuracy']:.4f}")
    print(f"Improvement over best baseline: {results['improvement_over_best_baseline']:.4f}")
    print("Success criterion met:", results["success_criterion_met"])
    print("Artifacts saved to", out.resolve())

if __name__ == "__main__":
    main()

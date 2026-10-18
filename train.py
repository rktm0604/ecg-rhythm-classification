"""
ECG beat classification (Normal vs Abnormal) on a small subset of MIT-BIH.
Small-scale prototype: Random Forest, split BY RECORD (no patient leakage).
Run:  python train.py
Outputs: model.pkl, metrics.json, confusion_matrix.png, sample_beats.json
"""
import json
import joblib
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# MIT-BIH annotation symbols mapped to 2 classes
NORMAL = {"N", "L", "R", "e", "j"}
ABNORMAL = {"A", "a", "J", "S", "V", "E", "F"}

# Different patients in train and test (no overlap)
TRAIN_RECORDS = ["101", "106", "119", "201", "203", "208"]
TEST_RECORDS = ["200", "210", "221"]

HALF = 90  # samples on each side of the R-peak (360 Hz -> 0.5 s window)


def beats_from_record(rec):
    """Return (features, labels) for every usable beat in one record."""
    r = wfdb.rdrecord(rec, pn_dir="mitdb", channels=[0])
    a = wfdb.rdann(rec, "atr", pn_dir="mitdb")
    sig = r.p_signal[:, 0]
    X, y = [], []
    for i in range(2, len(a.sample) - 2):
        sym = a.symbol[i]
        if sym in NORMAL:
            label = 0
        elif sym in ABNORMAL:
            label = 1
        else:
            continue
        p = a.sample[i]
        if p < HALF or p + HALF > len(sig):
            continue
        w = sig[p - HALF : p + HALF]
        w = (w - w.mean()) / (w.std() + 1e-8)  # normalize each beat
        rr_prev = a.sample[i] - a.sample[i - 1]
        rr_next = a.sample[i + 1] - a.sample[i]
        avg_rr = np.mean(np.diff(a.sample[i - 2 : i + 3]))
        feats = [rr_prev, rr_next, rr_prev / avg_rr, rr_next / avg_rr] + list(w[::6])
        X.append(feats)
        y.append(label)
    return np.array(X), np.array(y)


def load(records):
    xs, ys = [], []
    for rec in records:
        print(f"  loading record {rec} ...")
        X, y = beats_from_record(rec)
        print(f"    beats: {len(y)}  normal: {(y == 0).sum()}  abnormal: {(y == 1).sum()}")
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def main():
    print("Loading TRAIN records")
    Xtr, ytr = load(TRAIN_RECORDS)
    print("Loading TEST records")
    Xte, yte = load(TEST_RECORDS)

    print("\nTrain class counts [normal, abnormal]:", np.bincount(ytr, minlength=2))
    print("Test  class counts [normal, abnormal]:", np.bincount(yte, minlength=2))
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        raise SystemExit("A split is missing a class. Swap some records in TRAIN/TEST and rerun.")

    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    names = ["Normal", "Abnormal"]
    print("\nAccuracy:", round(accuracy_score(yte, pred), 4))
    print(classification_report(yte, pred, target_names=names, digits=3))
    cm = confusion_matrix(yte, pred)
    print("Confusion matrix (rows = true, cols = predicted):\n", cm)

    # Save artifacts
    joblib.dump(clf, "model.pkl")
    report = classification_report(yte, pred, target_names=names, output_dict=True)
    report["train_records"] = TRAIN_RECORDS
    report["test_records"] = TEST_RECORDS
    with open("metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test records)")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    # One real normal and one real abnormal test beat, for the API demo
    samples = {}
    for label, name in [(0, "normal_example"), (1, "abnormal_example")]:
        idx = int(np.where(yte == label)[0][0])
        samples[name] = {"features": Xte[idx].tolist()}
    with open("sample_beats.json", "w") as f:
        json.dump(samples, f)

    print("\nSaved: model.pkl, metrics.json, confusion_matrix.png, sample_beats.json")


if __name__ == "__main__":
    main()

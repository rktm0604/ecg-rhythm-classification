"""Reproduce Phase 2 DS2 error analysis from frozen predictions.

This script does not load model checkpoints or fit/tune anything. Optional raw
MIT-BIH data is used only to retrieve annotated waveform windows for selected
saved misclassifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "phase2_corrected"
OUT_DEFAULT = ART / "error_analysis"
MODELS = {"Random Forest": "pred_rf", "1D-CNN": "pred_cnn"}


def one_vs_rest_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    result = {}
    total = int(matrix.sum())
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fn = int(matrix[i, :].sum() - tp)
        fp = int(matrix[:, i].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": int(matrix[i, :].sum()),
            "precision": precision, "recall": recall, "f1": f1,
        }
    return result


def record_metrics(y: np.ndarray, pred: np.ndarray, records: np.ndarray, labels: list[str]) -> list[dict]:
    output = []
    for record in sorted(np.unique(records), key=lambda x: int(x)):
        mask = records == record
        yt, yp = y[mask], pred[mask]
        class_metrics = one_vs_rest_metrics(yt, yp, labels)
        present = [label for label in labels if class_metrics[label]["support"] > 0]
        output.append({
            "record": str(record),
            "n_beats": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1_present_classes": float(np.mean([class_metrics[c]["f1"] for c in present])),
            "classes_present": present,
            "class_metrics": class_metrics,
            "errors": int(np.count_nonzero(yt != yp)),
            "error_rate": float(np.mean(yt != yp)),
            "error_share": float(np.count_nonzero(yt != yp) / np.count_nonzero(y != pred)),
        })
    return output


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_examples(y: np.ndarray, pred: np.ndarray, records: np.ndarray,
                    beat_numbers: np.ndarray, model: str) -> list[dict]:
    # Spread examples across clinically important error patterns; no example
    # is fabricated and every row is an existing saved DS2 prediction.
    preferences = {
        "Random Forest": [("S", "N"), ("S", "V"), ("F", "N"), ("Q", "N")],
        "1D-CNN": [("N", "V"), ("N", "F"), ("S", "N"), ("Q", "V")],
    }[model]
    examples = []
    used_pairs = set()
    for true_label, predicted_label in preferences:
        found = np.flatnonzero((y == true_label) & (pred == predicted_label))
        if len(found) == 0:
            continue
        # Choose the middle observation to avoid presenting a hand-picked
        # extreme; source row/index is retained for exact reproducibility.
        idx = int(found[len(found) // 2])
        examples.append({
            "model": model,
            "row_index": idx,
            "record": str(records[idx]),
            "beat_number": int(beat_numbers[idx]),
            "true": true_label,
            "predicted": predicted_label,
        })
        used_pairs.add((true_label, predicted_label))
    if len(examples) < 3:
        # Fill any missing slots with the largest remaining confusion pairs.
        counts = Counter((str(a), str(b)) for a, b in zip(y, pred) if a != b)
        for pair, _ in counts.most_common():
            if pair in used_pairs:
                continue
            found = np.flatnonzero((y == pair[0]) & (pred == pair[1]))
            idx = int(found[len(found) // 2])
            examples.append({
                "model": model, "row_index": idx, "record": str(records[idx]),
                "beat_number": int(beat_numbers[idx]), "true": pair[0], "predicted": pair[1],
            })
            used_pairs.add(pair)
            if len(examples) >= 3:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mitdb",
                        help="Local MIT-BIH files for waveform examples; predictions analysis needs no dataset")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads((ART / "experiment_config.json").read_text(encoding="utf-8"))
    stored_results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    with np.load(ART / "predictions.npz", allow_pickle=False) as saved:
        pred_data = {key: saved[key] for key in saved.files}
    labels = list(config["labels"])
    y = pred_data["y_test"].astype(str)
    records = pred_data["record_test"].astype(str)
    beat_numbers = pred_data["beat_number_test"].astype(int)
    expected_records = set(map(str, config["test_records"]))
    if set(np.unique(records)) != expected_records:
        raise ValueError("Saved test prediction records do not match experiment_config.json")
    if not (len(y) == len(records) == len(beat_numbers)):
        raise ValueError("Saved labels, records, and beat indices have different lengths")
    if not set(np.unique(y)).issubset(labels):
        raise ValueError("Saved labels include a class outside the configured AAMI classes")

    summary = {
        "analysis": "Phase 2 DS2 error analysis from saved predictions; no model inference/training",
        "labels_order": labels,
        "test_records": sorted(expected_records, key=int),
        "n_test_beats": int(len(y)),
        "models": {},
        "representative_examples": {},
        "waveform_examples_status": "not generated",
    }
    all_example_records: list[dict] = []
    for model, key in MODELS.items():
        yp = pred_data[key].astype(str)
        if len(yp) != len(y):
            raise ValueError(f"{key} length does not match saved DS2 labels")
        cm = confusion_matrix(y, yp, labels=labels)
        classwise = one_vs_rest_metrics(y, yp, labels)
        confusions = [
            {"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
            for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]
        ]
        confusions.sort(key=lambda row: (-row["count"], row["true"], row["predicted"]))
        by_record = record_metrics(y, yp, records, labels)
        model_summary = {
            "accuracy": float(accuracy_score(y, yp)),
            "macro_f1": float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "class_metrics_one_vs_rest": classwise,
            "most_common_confusions": confusions,
            "record_metrics": by_record,
            "record_worst_macro_f1": sorted(by_record, key=lambda row: (row["macro_f1_present_classes"], -row["n_beats"]))[:5],
            "record_worst_accuracy": sorted(by_record, key=lambda row: (row["accuracy"], -row["n_beats"]))[:5],
            "record_largest_error_contribution": sorted(by_record, key=lambda row: (-row["error_share"], -row["errors"]))[:5],
        }
        summary["models"][model] = model_summary
        stored = stored_results["final_test"][model]
        if not np.isclose(model_summary["accuracy"], stored["accuracy"], atol=1e-12, rtol=0):
            raise AssertionError(f"Recomputed {model} accuracy disagrees with results.json")
        if not np.isclose(model_summary["macro_f1"], stored["macro_f1"], atol=1e-12, rtol=0):
            raise AssertionError(f"Recomputed {model} macro-F1 disagrees with results.json")
        if model_summary["confusion_matrix"] != stored["confusion"]:
            raise AssertionError(f"Recomputed {model} confusion matrix disagrees with results.json")
        for label in labels:
            computed = classwise[label]
            reported = stored["report"][label]
            for ours, theirs in ((computed["precision"], reported["precision"]),
                                 (computed["recall"], reported["recall"]),
                                 (computed["f1"], reported["f1-score"]),
                                 (computed["support"], reported["support"])):
                if not np.isclose(ours, theirs, atol=1e-12, rtol=0):
                    raise AssertionError(f"Recomputed {model} {label} metrics disagree with results.json")
        np.savetxt(args.out / f"confusion_matrix_{'rf' if model == 'Random Forest' else 'cnn'}.csv",
                   cm, delimiter=",", fmt="%d", header=",".join(labels), comments="")
        save_csv(args.out / f"class_metrics_{'rf' if model == 'Random Forest' else 'cnn'}.csv", [
            {"class": label, **values} for label, values in classwise.items()
        ], ["class", "tp", "fp", "fn", "tn", "support", "precision", "recall", "f1"])
        flat_rows = []
        for row in by_record:
            base = {k: row[k] for k in ("record", "n_beats", "accuracy", "macro_f1_present_classes", "classes_present", "errors", "error_rate", "error_share")}
            for label, values in row["class_metrics"].items():
                for metric, value in values.items():
                    base[f"{label}_{metric}"] = value
            flat_rows.append(base)
        save_csv(args.out / f"record_metrics_{'rf' if model == 'Random Forest' else 'cnn'}.csv", flat_rows, list(flat_rows[0]))
        examples = choose_examples(y, yp, records, beat_numbers, model)
        summary["representative_examples"][model] = examples
        all_example_records.extend(examples)

        fig, ax = plt.subplots(figsize=(6.4, 5.3), constrained_layout=True)
        im = ax.imshow(cm, cmap="Blues")
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels, xlabel="Predicted class", ylabel="True class",
               title=f"Phase 2 DS2 confusion matrix — {model}")
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i,j] > cm.max() * 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="Beat count")
        fig.savefig(args.out / f"confusion_matrix_{'rf' if model == 'Random Forest' else 'cnn'}.png", dpi=170)
        plt.close(fig)

    summary["macro_f1_delta_rf_minus_cnn"] = summary["models"]["Random Forest"]["macro_f1"] - summary["models"]["1D-CNN"]["macro_f1"]
    summary["accuracy_delta_rf_minus_cnn"] = summary["models"]["Random Forest"]["accuracy"] - summary["models"]["1D-CNN"]["accuracy"]
    summary["reproduced_saved_results_json"] = True

    # Record-level overview. Macro-F1 is averaged over classes present in the
    # true labels for that record; per-class metrics still preserve all five
    # configured classes and report support explicitly.
    rf_records = summary["models"]["Random Forest"]["record_metrics"]
    cnn_records = summary["models"]["1D-CNN"]["record_metrics"]
    x = np.arange(len(rf_records))
    tick_labels = [row["record"] for row in rf_records]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, [r["macro_f1_present_classes"] for r in rf_records], marker="o", label="Random Forest")
    axes[0].plot(x, [r["macro_f1_present_classes"] for r in cnn_records], marker="o", label="1D-CNN")
    axes[0].set_ylabel("Macro-F1\n(classes present)")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, [r["accuracy"] for r in rf_records], marker="o", label="Random Forest")
    axes[1].plot(x, [r["accuracy"] for r in cnn_records], marker="o", label="1D-CNN")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xticks(x, tick_labels)
    axes[1].set_xlabel("DS2 record")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Phase 2 saved-prediction performance by DS2 record")
    fig.savefig(args.out / "recordwise_performance.png", dpi=170)
    plt.close(fig)

    if args.data_dir.exists():
        sys.path.insert(0, str(ROOT))
        import pipeline
        pipeline.DATA_DIR = args.data_dir
        cache = {}
        plotted = []
        for example in all_example_records:
            record = example["record"]
            if record not in cache:
                signal, positions, symbols = pipeline._read_record(record)
                rows, _, _, _ = pipeline.extract_record_beats(signal, positions, symbols)
                cache[record] = rows
            rows = cache[record]
            beat_no = example["beat_number"]
            if beat_no < 0 or beat_no >= len(rows):
                raise IndexError(f"Saved beat_number {beat_no} cannot map to record {record}")
            row = rows[beat_no]
            if row["label"] != example["true"]:
                raise ValueError(f"Saved true label does not match extracted record {record} beat {beat_no}")
            example["sample"] = int(row["sample"])
            example["waveform_source"] = "pipeline.extract_record_beats normalized 200-sample window"
            example["waveform"] = row["wave"].astype(float).round(7).tolist()
            plotted.append((example, row["wave"]))
        if plotted:
            cols = 3
            rows_n = int(np.ceil(len(plotted) / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(15, rows_n * 3.15), squeeze=False, constrained_layout=True)
            for ax, (example, wave) in zip(axes.flat, plotted):
                x = (np.arange(len(wave)) - pipeline.PRE) / pipeline.FS
                ax.plot(x, wave, color="#156f82", linewidth=1.15)
                ax.axvline(0, color="#c43d3d", linestyle="--", linewidth=1, label="Annotated R")
                ax.set_title(f"{example['model']} · record {example['record']} · beat {example['beat_number']}\nTrue {example['true']} → predicted {example['predicted']}", fontsize=9)
                ax.set_xlabel("Seconds from annotated R")
                ax.set_ylabel("Per-window z-score")
                ax.grid(alpha=0.2)
            for ax in axes.flat[len(plotted):]:
                ax.remove()
            fig.savefig(args.out / "representative_misclassified_beats.png", dpi=170)
            plt.close(fig)
            summary["waveform_examples_status"] = "generated and matched to saved DS2 row using record and beat_number"
            summary["waveform_window"] = "200 samples: 90 before and 110 from R; per-window z-score; R marked at 0 seconds"
        else:
            summary["waveform_examples_status"] = "no saved misclassifications available to plot"
    else:
        summary["waveform_examples_status"] = f"not generated; local dataset not found at {args.data_dir}"

    (args.out / "error_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved reproducible DS2 error analysis to {args.out}")
    for model, result in summary["models"].items():
        print(f"{model}: accuracy={result['accuracy']:.6f}, macro_f1={result['macro_f1']:.6f}")
        print("  common confusions:", result["most_common_confusions"][:5])
        print("  weakest record macro-F1:", [(r['record'], round(r['macro_f1_present_classes'], 4), r['n_beats']) for r in result['record_worst_macro_f1']])
        print("  largest error contributions:", [(r['record'], r['errors'], round(r['error_share'], 3)) for r in result['record_largest_error_contribution']])


if __name__ == "__main__":
    main()

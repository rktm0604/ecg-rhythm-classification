"""Generate descriptive Phase 2 EDA from DS1 only.

This script does not fit models, inspect DS2, or change saved experiment outputs.
Pass --data-dir when the local MIT-BIH files are outside the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


FEATURE_NAMES = [
    "rr_prev",
    "rr_next",
    "rr_prev / rr_local",
    "rr_next / rr_local",
]


def _rf_feature_names():
    wave_points = np.linspace(0, pipeline.WINDOW - 1, 30).round().astype(int)
    return FEATURE_NAMES + [f"wave[{int(point) - pipeline.PRE}]" for point in wave_points]


def generate_eda(data_dir: Path, output_dir: Path, model_path: Path) -> dict:
    pipeline.DATA_DIR = Path(data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pipeline.DS1_RECORDS
    label_counts: Counter[str] = Counter()
    rows_by_class: dict[str, list[dict]] = {label: [] for label in pipeline.LABELS}
    example_by_class: dict[str, tuple[str, dict]] = {}

    for record in records:
        signal, positions, symbols = pipeline._read_record(record)
        rows, _, _, _ = pipeline.extract_record_beats(signal, positions, symbols)
        for row in rows:
            label = row["label"]
            label_counts[label] += 1
            rows_by_class[label].append(row)
            example_by_class.setdefault(label, (record, row))

    missing_classes = [label for label in pipeline.LABELS if not rows_by_class[label]]
    if missing_classes:
        raise RuntimeError(f"DS1 extraction has no eligible examples for: {missing_classes}")

    # 1. DS1 class distribution; logarithmic scale keeps rare classes visible.
    counts = [label_counts[label] for label in pipeline.LABELS]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(pipeline.LABELS, counts, color=["#277da1", "#43aa8b", "#f9c74f", "#f9844a", "#9b5de5"])
    ax.set_yscale("log")
    ax.set_ylabel("Eligible DS1 beats (log scale)")
    ax.set_xlabel("AAMI class")
    ax.set_title("Corrected Phase 2 class distribution — DS1 only")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution_ds1.png", dpi=180)
    plt.close(fig)

    # 2. First eligible beat per class in deterministic DS1 record order.
    fig, axes = plt.subplots(len(pipeline.LABELS), 1, figsize=(9, 11), sharex=True)
    offsets_seconds = (np.arange(pipeline.WINDOW) - pipeline.PRE) / pipeline.FS
    for ax, label in zip(axes, pipeline.LABELS):
        record, row = example_by_class[label]
        ax.plot(offsets_seconds, row["wave"], lw=0.8, color="#176b87")
        ax.axvline(0, color="#e76f51", lw=1, ls="--")
        ax.set_ylabel(label)
        ax.set_title(f"Class {label} · record {record} · sample {row['sample']}", loc="left", fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Seconds relative to annotated R-peak (R = 0)")
    fig.suptitle("Representative normalized Phase 2 windows — DS1", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "representative_waveforms_ds1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. RR distributions per class. Values are converted from samples to seconds.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    rr_columns = [("rr_prev", "Previous RR (s)"), ("rr_next", "Following RR (s)"), ("rr_local", "Local RR mean (s)")]
    for ax, (field, title) in zip(axes, rr_columns):
        values = [np.asarray([row[field] for row in rows_by_class[label]], dtype=float) / pipeline.FS for label in pipeline.LABELS]
        ax.boxplot(values, tick_labels=pipeline.LABELS, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("RR feature distributions by class — DS1 only (outliers hidden)")
    fig.tight_layout()
    fig.savefig(output_dir / "rr_distributions_ds1.png", dpi=180)
    plt.close(fig)

    # 4. Previous versus following RR relationship, capped deterministically per class.
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"N": "#277da1", "S": "#43aa8b", "V": "#f9844a", "F": "#9b5de5", "Q": "#d00000"}
    for label in pipeline.LABELS:
        rows = rows_by_class[label]
        indices = np.arange(len(rows))
        if len(indices) > 2500:
            indices = np.sort(rng.choice(indices, size=2500, replace=False))
        x = np.asarray([rows[i]["rr_prev"] for i in indices]) / pipeline.FS
        y = np.asarray([rows[i]["rr_next"] for i in indices]) / pipeline.FS
        ax.scatter(x, y, s=7, alpha=0.22, label=f"{label} (n={len(rows):,})", color=colors[label], rasterized=True)
    sampled_x = np.concatenate([
        np.asarray([row["rr_prev"] for row in rows_by_class[label]], dtype=float) / pipeline.FS
        for label in pipeline.LABELS
    ])
    sampled_y = np.concatenate([
        np.asarray([row["rr_next"] for row in rows_by_class[label]], dtype=float) / pipeline.FS
        for label in pipeline.LABELS
    ])
    # Extreme pause intervals remain in the analysis but are outside the
    # display range so the dense central relationship remains legible.
    ax.set_xlim(0, float(np.quantile(sampled_x, 0.995)))
    ax.set_ylim(0, float(np.quantile(sampled_y, 0.995)))
    ax.text(0.02, 0.98, "Axes end at 99.5th percentile; longer pauses are clipped", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("Previous RR (s)")
    ax.set_ylabel("Following RR (s)")
    ax.set_title("Previous and following RR intervals — DS1 only")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "rr_relationship_ds1.png", dpi=180)
    plt.close(fig)

    # 5. Frozen Phase 2 RF feature importance. This is descriptive, not tuning.
    rf = joblib.load(model_path)
    names = _rf_feature_names()
    if len(names) != len(rf.feature_importances_):
        raise RuntimeError("Feature name count does not match the saved Phase 2 RF checkpoint")
    ranking = np.argsort(rf.feature_importances_)[::-1]
    top = ranking[:12]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh([names[i] for i in top[::-1]], rf.feature_importances_[top[::-1]], color="#277da1")
    ax.set_xlabel("Saved RF impurity-based feature importance")
    ax.set_title("Top Phase 2 Random Forest features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "rf_feature_importance.png", dpi=180)
    plt.close(fig)

    rr_summary = {}
    for label in pipeline.LABELS:
        rr_summary[label] = {}
        for field, _ in rr_columns:
            values = np.asarray([row[field] for row in rows_by_class[label]], dtype=float) / pipeline.FS
            rr_summary[label][field] = {
                "median_seconds": float(np.median(values)),
                "q1_seconds": float(np.quantile(values, 0.25)),
                "q3_seconds": float(np.quantile(values, 0.75)),
            }

    top_features = [
        {"feature": names[int(i)], "importance": float(rf.feature_importances_[i])}
        for i in top
    ]

    summary = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "source_data_dir": str(Path(data_dir).resolve()),
        "records_used": records,
        "split_used_for_eda": "DS1 only; no DS2 data loaded",
        "eligible_counts_by_class": {label: int(label_counts[label]) for label in pipeline.LABELS},
        "eligible_proportions_by_class": {
            label: float(label_counts[label] / sum(label_counts.values())) for label in pipeline.LABELS
        },
        "total_eligible_beats": int(sum(label_counts.values())),
        "representative_waveform_records": {label: example_by_class[label][0] for label in pipeline.LABELS},
        "rr_summary_seconds": rr_summary,
        "top_saved_rf_features": top_features,
        "rr_units_in_plots": "seconds (sample intervals divided by 360 Hz)",
        "rf_importance_source": str(Path(model_path).resolve()),
        "figures": [
            "class_distribution_ds1.png",
            "representative_waveforms_ds1.png",
            "rr_distributions_ds1.png",
            "rr_relationship_ds1.png",
            "rf_feature_importance.png",
        ],
        "interpretation_note": (
            "These figures describe DS1 and the already selected saved RF. They were not used to select or tune models."
        ),
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "mitdb")
    parser.add_argument("--out", type=Path, default=root / "artifacts" / "phase2_corrected" / "eda")
    parser.add_argument("--model", type=Path, default=root / "artifacts" / "phase2_corrected" / "random_forest.joblib")
    args = parser.parse_args()
    summary = generate_eda(args.data_dir, args.out, args.model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

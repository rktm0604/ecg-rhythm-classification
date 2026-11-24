"""Audit extraction eligibility with the exact pipeline rules.

Corrected reports default to artifacts/phase1_corrected so earlier reports
and model artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import wfdb

from pipeline import (
    DATA_DIR,
    DS1_RECORDS,
    DS2_RECORDS,
    LABEL_MAP,
    LABELS,
    POST,
    PRE,
    WINDOW,
    _read_record,
    extract_record_beats,
)

DS1 = DS1_RECORDS
DS2 = DS2_RECORDS
OUTPUT_DIR = Path("artifacts") / "phase1_corrected"


def audit_record(record, split):
    record_path = DATA_DIR / str(record)
    header = wfdb.rdheader(str(record_path))
    signal, positions, symbols = _read_record(record)
    rows, excluded, mapped, unmapped = extract_record_beats(signal, positions, symbols)

    annotated = Counter(
        LABEL_MAP[str(symbol)] for symbol in symbols if str(symbol) in LABEL_MAP
    )
    eligible = Counter(row["label"] for row in rows)
    reason_counts = Counter()
    for (label, reason), count in excluded.items():
        reason_counts[reason] += count

    result = {
        "split": split,
        "record": str(record),
        "sampling_frequency": header.fs,
        "signal_length": header.sig_len,
        "annotation_beats": mapped,
        "unmapped_annotations": unmapped,
        "eligible_beats": len(rows),
        "exclusion_reasons": dict(sorted(reason_counts.items())),
    }
    for cls in LABELS:
        result[f"{cls}_annotated"] = int(annotated[cls])
        result[f"{cls}_eligible"] = int(eligible[cls])
        for reason in sorted(reason_counts):
            result[f"{cls}_excluded_{reason}"] = int(excluded[(cls, reason)])
    return result


def audit_dataset(output_dir=OUTPUT_DIR):
    all_rows = [audit_record(r, "DS1") for r in DS1]
    all_rows.extend(audit_record(r, "DS2") for r in DS2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    csv_path = output_dir / "dataset_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "window_before": PRE,
        "window_after": POST,
        "window_size": WINDOW,
        "rr_definition": (
            "Consecutive mapped heartbeat positions only; rr_local is the mean "
            "of up to two preceding and two following heartbeat intervals."
        ),
        "classes": LABELS,
        "ds1": {"records": [r for r in all_rows if r["split"] == "DS1"]},
        "ds2": {"records": [r for r in all_rows if r["split"] == "DS2"]},
    }
    for split in ("ds1", "ds2"):
        rows = report[split]["records"]
        report[split]["eligible_totals"] = {
            c: sum(r[f"{c}_eligible"] for r in rows) for c in LABELS
        }
        report[split]["excluded_totals_by_reason"] = dict(
            sum((Counter(r["exclusion_reasons"]) for r in rows), Counter())
        )
    json_path = output_dir / "dataset_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CSV saved: {csv_path.resolve()}")
    print(f"JSON saved: {json_path.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    audit_dataset(args.out)


if __name__ == "__main__":
    main()

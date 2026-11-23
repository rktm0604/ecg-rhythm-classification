"""Approved five-class MIT-BIH data pipeline.

Labels follow the Phase 1 mapping: N, S, V, F, Q.
Windows contain 90 samples before and 110 after each annotated R peak.
Recordings and annotations are read from the local data/mitdb directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb


FS = 360
PRE = 90
POST = 110
WINDOW = PRE + POST

# Resolve the data folder relative to this file.
DATA_DIR = Path(__file__).resolve().parent / "data" / "mitdb"

LABEL_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

LABELS = ["N", "S", "V", "F", "Q"]

# Phase 1-approved recording groups.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "121", "122", "124", "201", "203", "205",
    "207", "208", "209", "215", "220", "223", "230",
]

DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "123", "200",
    "210", "212", "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
]

PACED_RECORDS = {"102", "104", "107", "217"}

# Records 201 and 202 belong to the same subject.
KNOWN_SAME_SUBJECT = {frozenset(("201", "202"))}
EXCLUDED_FOR_SUBJECT_OVERLAP = {"202"}


def check_record_split(ds1=None, ds2=None):
    """Check recording-level and known subject-level separation."""
    ds1 = set(DS1_RECORDS if ds1 is None else ds1)
    ds2 = set(DS2_RECORDS if ds2 is None else ds2)
    ds2 -= EXCLUDED_FOR_SUBJECT_OVERLAP

    if ds1 & ds2:
        raise ValueError(
            f"Recording overlap between DS1 and DS2: {sorted(ds1 & ds2)}"
        )

    if (ds1 | ds2) & PACED_RECORDS:
        raise ValueError("Paced recordings must be excluded.")

    for pair in KNOWN_SAME_SUBJECT:
        if pair & ds1 and pair & ds2:
            raise ValueError(
                f"Subject leakage: records {sorted(pair)} cross DS1/DS2."
            )

    return sorted(ds1), sorted(ds2)


def _read_record(record):
    """Read one ECG recording and its annotations from local disk."""
    record_path = DATA_DIR / str(record)

    header_path = record_path.with_suffix(".hea")
    signal_path = record_path.with_suffix(".dat")
    annotation_path = record_path.with_suffix(".atr")

    missing = [
        path.name
        for path in (header_path, signal_path, annotation_path)
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Local MIT-BIH files missing for record {record}: "
            f"{', '.join(missing)}. Expected files in: {DATA_DIR}"
        )

    # No pn_dir argument: read directly from local files.
    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    # Select MLII by channel label where possible.
    names = [
        str(name).upper().replace(" ", "")
        for name in getattr(rec, "sig_name", [])
    ]

    mlii = next(
        (
            index
            for index, name in enumerate(names)
            if "MLII" in name or "MODIFIEDLIMBII" in name
        ),
        None,
    )

    channel = mlii if mlii is not None else 0

    if mlii is None:
        channel_name = names[0] if names else "unknown"
        print(
            f"WARNING: {record}: MLII channel label not found; "
            f"using channel 0 ({channel_name})."
        )

    signal = np.asarray(rec.p_signal[:, channel], dtype=np.float32)

    return (
        signal,
        np.asarray(ann.sample, dtype=int),
        np.asarray(ann.symbol),
    )


def _safe_zscore(window):
    """Normalize one heartbeat window."""
    window = np.asarray(window, dtype=np.float32)
    return (window - window.mean()) / (window.std() + 1e-8)


def _engineered_features(window, rr_prev, rr_next, rr_local):
    """Return four RR features and 30 evenly spaced waveform values."""
    points = np.linspace(0, len(window) - 1, 30).round().astype(int)

    return np.asarray(
        [
            rr_prev,
            rr_next,
            rr_prev / max(rr_local, 1e-8),
            rr_next / max(rr_local, 1e-8),
            *window[points].tolist(),
        ],
        dtype=np.float32,
    )


def build_dataset(records):
    """Read local MIT-BIH records and return beat windows and labels.

    Reference annotations provide beat locations and labels. Annotation
    symbols are not included as model input features.
    """
    X_rf = []
    X_wave = []
    y = []
    rec_ids = []
    beat_nums = []
    beats_per_record = {}

    for record in records:
        record = str(record)

        if record in PACED_RECORDS:
            print(f"record {record}: skipped (paced recording)")
            continue

        signal, positions, symbols = _read_record(record)
        kept = 0

        for index, symbol in enumerate(symbols):
            label = LABEL_MAP.get(str(symbol))

            if label is None or index == 0 or index == len(positions) - 1:
                continue

            peak = int(positions[index])
            start = peak - PRE
            end = peak + POST

            if start < 0 or end > len(signal):
                continue

            raw = signal[start:end]

            if len(raw) != WINDOW or not np.isfinite(raw).all():
                continue

            wave = _safe_zscore(raw)

            rr_prev = float(positions[index] - positions[index - 1])
            rr_next = float(positions[index + 1] - positions[index])

            nearby_positions = positions[
                max(0, index - 2):min(len(positions), index + 3)
            ]

            local = float(np.mean(np.diff(nearby_positions)))

            if (
                not np.isfinite([rr_prev, rr_next, local]).all()
                or local <= 0
            ):
                continue

            X_rf.append(
                _engineered_features(wave, rr_prev, rr_next, local)
            )
            X_wave.append(wave)
            y.append(label)
            rec_ids.append(record)
            beat_nums.append(kept)
            kept += 1

        beats_per_record[record] = kept
        print(f"record {record}: {kept:,} retained labelled beats")

    if not y:
        raise RuntimeError(
            "No labelled beats were extracted. Check local data files "
            "and the selected record list."
        )

    return {
        "X_rf": np.stack(X_rf),
        "X_wave": np.stack(X_wave),
        "y": np.asarray(y, dtype="U1"),
        "record": np.asarray(rec_ids, dtype="U3"),
        "beat_number": np.asarray(beat_nums, dtype=np.int32),
        "beats_per_record": beats_per_record,
    }

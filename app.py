"""Five-class dashboard for the completed corrected Phase 2 experiment.

Run from the project root: .venv/Scripts/streamlit.exe run app.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parent / "artifacts" / "phase2_corrected"
LABEL_NAMES = {"N": "Normal", "S": "Supraventricular ectopic", "V": "Ventricular ectopic", "F": "Fusion", "Q": "Unknown / paced / unclassifiable"}
COLORS = {"N": "#43D9C5", "S": "#8AB4F8", "V": "#F0A36B", "F": "#B69CFA", "Q": "#E879A5"}
MODEL_PREDICTION_KEYS = {
    "Random Forest": "pred_rf", "1D-CNN": "pred_cnn",
    "Majority baseline": "pred_majority", "RR-only Logistic Regression": "pred_rr",
}

st.set_page_config(page_title="PULSE | Corrected ECG Experiment", page_icon="🫀", layout="wide")


@st.cache_data
def load_phase2():
    results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    config = json.loads((ART / "experiment_config.json").read_text(encoding="utf-8"))
    with np.load(ART / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    return results, config, predictions


try:
    R, C, P = load_phase2()
except Exception as exc:
    st.title("Corrected Phase 2 dashboard")
    st.error(f"Could not load artifacts from {ART}: {exc}")
    st.stop()

labels = C["labels"]
y = P["y_test"].astype(str)
records = P["record_test"].astype(str)
models = R["final_test"]

st.title("ECG Rhythm Classification")
st.caption("Corrected Phase 2 · MIT-BIH · Five-class AAMI · Offline/delayed evaluation")
st.info("Research prototype only. Reference annotations locate beats. This offline/delayed design uses future RR information and 110 post-peak waveform samples; it is not causal at the R peak.")

with st.sidebar:
    st.header("Phase 2 corrected")
    page = st.radio("View", ["Overview", "Model comparison", "Test predictions", "Method & limitations"])
    st.caption(f"DS1 fit: {len(C['fit_records'])} records")
    st.caption(f"DS1 validation: {len(C['validation_records'])} records")
    st.caption(f"DS2 test: {len(C['test_records'])} records · {len(y):,} beats")
    st.caption("Research prototype · Not for clinical decisions")

if page == "Overview":
    st.subheader("Final DS2 results")
    rf, cnn = models["Random Forest"], models["1D-CNN"]
    a, b, c, d = st.columns(4)
    a.metric("RF macro-F1", f"{rf['macro_f1']:.3f}")
    b.metric("RF accuracy", f"{rf['accuracy']:.3f}")
    c.metric("CNN macro-F1", f"{cnn['macro_f1']:.3f}")
    d.metric("Test beats", f"{len(y):,}")
    st.warning("Minority-class performance is limited. In DS2, F support is 387 and Q support is 7; RF recall is zero for both, and CNN recall is zero for Q. Accuracy is dominated by class N.")
    counts = C["all_corrected_counts"]
    st.subheader("Corrected extracted class counts")
    st.bar_chart(pd.Series(counts, name="beats"))
    st.caption("Counts include DS1 and DS2; only DS1 was used for model selection.")

elif page == "Model comparison":
    st.subheader("DS1 validation and DS2 final test")
    rows = []
    for split, split_models in (("DS1 validation", R["validation"]), ("DS2 final test", models)):
        for name, metrics in split_models.items():
            rows.append({"Split": split, "Model": name, "Macro-F1": metrics["macro_f1"], "Accuracy": metrics["accuracy"]})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
    selected = st.selectbox("Model details", list(models))
    report = models[selected]["report"]
    table = [{"Class": f"{c} — {LABEL_NAMES[c]}", "Precision": report[c]["precision"], "Recall": report[c]["recall"], "F1": report[c]["f1-score"], "Support": int(report[c]["support"])} for c in labels]
    st.dataframe(pd.DataFrame(table).round(4), hide_index=True, use_container_width=True)
    cm = np.asarray(models[selected]["confusion"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="PuBuGn")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{selected} · DS2 confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > cm.max() * .5 else "#142238")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif page == "Test predictions":
    st.subheader("Saved DS2 predictions")
    model_name = st.selectbox("Model", list(MODEL_PREDICTION_KEYS))
    prediction = P[MODEL_PREDICTION_KEYS[model_name]].astype(str)
    true_filter = st.selectbox("True class", ["All"] + labels)
    record_filter = st.selectbox("Record", ["All"] + sorted(set(records.tolist())))
    mask = np.ones(len(y), dtype=bool)
    if true_filter != "All": mask &= y == true_filter
    if record_filter != "All": mask &= records == record_filter
    rows = pd.DataFrame({"Record": records[mask], "Beat index": P["beat_number_test"][mask], "True class": y[mask], "Predicted class": prediction[mask], "Correct": y[mask] == prediction[mask]})
    st.caption(f"Showing {len(rows):,} beats · predictions are loaded from predictions.npz")
    st.dataframe(rows.head(1000), hide_index=True, use_container_width=True)

else:
    st.subheader("Method, split, and limitations")
    st.markdown("""
**Mapping:** N = N/L/R/e/j; S = A/a/J/S; V = V/E; F = F; Q = //f/Q.

**Features:** Corrected Phase 1 extraction uses consecutive mapped heartbeat positions only. `rr_prev` and `rr_next` are adjacent valid-heartbeat intervals. `rr_local` averages up to two preceding and two following intervals. Windows contain 90 pre-peak and 110 post-peak samples.

**Selection:** RF candidates and CNN early stopping use the same record-grouped DS1 fit/validation split. Baselines use that fit split too. DS2 is reserved for final reporting and was not used for selection.

**Limits:** F and Q are very rare, especially Q. DS2 has 387 F and 7 Q beats; zero recall for some classes means the five-class performance is not uniform. This offline/delayed experiment uses future beat timing and post-peak signal. It does not test causal or real-time classification. Reference annotations provide beat positions, and MIT-BIH class proportions are not population prevalence.
""")
    st.json({"fit_records": C["fit_records"], "validation_records": C["validation_records"], "test_records": C["test_records"], "rf_selected_params": C["rf_selected_params"], "cnn_best_epoch": C["cnn_best_epoch"], "selection_data": C["selection_data"]})

st.divider()
st.caption("PULSE · Corrected Phase 2 results · Academic research only · Not a medical device")

"""
[L2] Simple Streamlit front end for the existing FastAPI backend.
Backend endpoints used (change the 3 small functions below if yours differ):
  GET  /sample   -> {"normal_example": {"features": [...]}, "abnormal_example": {...}}
  POST /predict  -> {"features": [...]}  returns {"class": "...", "confidence": 0.xx}
  GET  /metrics  -> classification report dict (precision/recall/f1 per class, accuracy)
Run:  streamlit run ui.py
"""
import json
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="ECG Rhythm Classifier", layout="wide")
API = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000").rstrip("/")
st.sidebar.caption("Research prototype, not a medical device.")

# ---- the only three places that talk to the backend -------------------
def get_sample():
    return requests.get(f"{API}/sample", timeout=10).json()

def predict(features):
    r = requests.post(f"{API}/predict", json={"features": features}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_metrics():
    return requests.get(f"{API}/metrics", timeout=10).json()
# -----------------------------------------------------------------------

st.title("ECG Rhythm Classifier")
try:
    requests.get(API, timeout=3)
except Exception:
    st.error(f"Cannot reach the backend at {API}. Start it first (uvicorn api:app).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Classify a beat", "Model performance", "About"])

# ------------------------------ Tab 1 ----------------------------------
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("1. Choose a beat")
        samples = get_sample()
        choice = st.radio("Example beat from the held-out test records",
                          list(samples.keys()), horizontal=True)
        text = st.text_area("Features (JSON list). Edit or paste your own.",
                            json.dumps(samples[choice]["features"]), height=140)
        run = st.button("Classify", type="primary")
    with right:
        st.subheader("2. Result")
        try:
            feats = json.loads(text)
            rr, wave = feats[:4], feats[4:]
            fig, ax = plt.subplots(figsize=(5, 2.2))
            ax.plot(wave, color="#0F766E", lw=1.6)
            ax.set_title("Beat shape (normalized waveform features)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig)
            st.caption(f"RR features: previous {rr[0]:.0f}, next {rr[1]:.0f} samples")
            if run:
                out = predict(feats)
                c1, c2 = st.columns(2)
                c1.metric("Predicted class", out["class"])
                c2.metric("Confidence", f'{out["confidence"]:.0%}')
                st.progress(float(out["confidence"]))
        except Exception as e:
            st.warning(f"Could not classify: {e}")

# ------------------------------ Tab 2 ----------------------------------
with tab2:
    m = get_metrics()
    skip = {"accuracy", "macro avg", "weighted avg", "train_records", "test_records", "confusion"}
    classes = [k for k in m if k not in skip and isinstance(m[k], dict)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f'{m["accuracy"]:.3f}')
    if "macro avg" in m:
        c2.metric("Macro F1", f'{m["macro avg"]["f1-score"]:.3f}')
    c3.metric("Test beats", int(m["macro avg"]["support"]) if "macro avg" in m else "-")
    st.subheader("Per-class results")
    st.dataframe(pd.DataFrame({c: m[c] for c in classes}).T
                 [["precision", "recall", "f1-score", "support"]].round(3))
    st.bar_chart(pd.DataFrame({c: {k: m[c][k] for k in ("precision", "recall", "f1-score")}
                               for c in classes}).T)
    if "confusion" in m:                                     # optional, if your backend adds it
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(m["confusion"], index=classes, columns=classes))
    if "train_records" in m:
        st.caption(f'Train records: {", ".join(map(str, m["train_records"]))}  |  '
                   f'Test records: {", ".join(map(str, m["test_records"]))}')

# ------------------------------ Tab 3 ----------------------------------
with tab3:
    st.markdown("""
**What it does:** classifies a heartbeat from ECG features using a Random Forest trained on
MIT-BIH Arrhythmia Database records, evaluated on records the model never saw (split by patient).

**Features (34):** 4 rhythm features (RR intervals) + 30 waveform samples of the normalized beat.

**Limits:** small subset of records, one model, annotated R-peaks used. Not a medical device.
""")

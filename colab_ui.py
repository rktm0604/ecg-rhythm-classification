# ECG Rhythm Classifier: UI that runs inside Google Colab (no backend server needed).
# Cell 1:  !pip install -q gradio
# Cell 2:  from google.colab import files; files.upload()   # choose model.pkl, metrics.json, sample_beats.json
# Cell 3:  paste everything below and run it. Open the "public URL" it prints.
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gradio as gr

model = joblib.load("model.pkl")
metrics = json.load(open("metrics.json"))
samples = json.load(open("sample_beats.json"))
N_FEATURES = model.n_features_in_
NAMES = {0: "Normal", 1: "Abnormal"}          # same labels as api.py


def load_example(name):
    return json.dumps(samples[name]["features"])


def classify(text):
    try:
        feats = json.loads(text)
        if len(feats) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(feats)}")
        proba = model.predict_proba([feats])[0]
    except Exception as e:
        return None, f"**Could not classify:** {e}", None
    probs = {NAMES.get(c, str(c)): float(p) for c, p in zip(model.classes_, proba)}
    label = max(probs, key=probs.get)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.plot(feats[4:], color="#0F766E", lw=1.8)
    ax.set_title("Beat shape (normalized waveform features)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.close(fig)
    info = (f"### Predicted class: {label}\n**Confidence:** {probs[label]:.0%}\n\n"
            f"RR features: previous {feats[0]:.0f}, next {feats[1]:.0f} samples")
    return fig, info, probs


skip = {"accuracy", "macro avg", "weighted avg", "train_records", "test_records", "confusion"}
classes = [k for k in metrics if k not in skip and isinstance(metrics[k], dict)]
table = (pd.DataFrame({c: metrics[c] for c in classes}).T
         [["precision", "recall", "f1-score", "support"]].round(3).reset_index()
         .rename(columns={"index": "class"}))
summary = f"**Accuracy:** {metrics['accuracy']:.3f}"
if "macro avg" in metrics:
    summary += f"  |  **Macro F1:** {metrics['macro avg']['f1-score']:.3f}  |  **Test beats:** {int(metrics['macro avg']['support'])}"
if "train_records" in metrics:
    summary += (f"\n\nTrain records: {', '.join(map(str, metrics['train_records']))}  \n"
                f"Test records: {', '.join(map(str, metrics['test_records']))}")

first = list(samples)[0]
with gr.Blocks(title="ECG Rhythm Classifier") as demo:
    gr.Markdown("# ECG Rhythm Classifier\nResearch prototype, not a medical device.")
    with gr.Tab("Classify a beat"):
        ex = gr.Radio(list(samples), value=first, label="Example beat from held-out test records")
        txt = gr.Textbox(value=load_example(first), lines=5, label="Features (JSON list, edit or paste your own)")
        btn = gr.Button("Classify", variant="primary")
        with gr.Row():
            plot = gr.Plot(label="Beat shape")
            out = gr.Markdown()
            lab = gr.Label(label="Class probabilities")
        ex.change(load_example, ex, txt)
        btn.click(classify, txt, [plot, out, lab])
    with gr.Tab("Model performance"):
        gr.Markdown(summary)
        gr.Dataframe(table, label="Per-class results (precision, recall, F1)")
    with gr.Tab("About"):
        gr.Markdown("Random Forest on 34 features (4 RR-interval features + 30 waveform samples), "
                    "trained on MIT-BIH records and evaluated on records the model never saw.")

demo.launch(share=True)

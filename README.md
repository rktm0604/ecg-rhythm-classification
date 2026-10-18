# ECG Beat Classification Prototype (Normal vs Abnormal)

Small-scale prototype: MIT-BIH subset, Random Forest, split by record.

## Run (in order)
1. Open a terminal in this folder.
2. `pip install -r requirements.txt`   (needs internet; wfdb downloads data from PhysioNet)
3. `python train.py`
   - Check "class counts": both train and test need Normal AND Abnormal beats.
   - Read accuracy, per-class precision/recall/F1, and the confusion matrix.
   - Creates: model.pkl, metrics.json, confusion_matrix.png, sample_beats.json
4. `uvicorn api:app`
5. Open http://127.0.0.1:8000/docs
   - GET /sample  -> copy one example ("features": [...] list)
   - POST /predict -> Try it out -> paste {"features": [...]} -> Execute
   - GET /metrics -> saved results
6. Screenshot: terminal output, confusion_matrix.png, /predict response, /metrics.
7. `git init`, `git add .`, `git commit -m "working prototype"`, `git push`

## If something fails
- A test record has no abnormal beats: edit TRAIN_RECORDS / TEST_RECORDS in train.py
  (try 207, 214 or 228 for test) and rerun.
- Slow or failed download: check internet; rerun (wfdb retries from PhysioNet).
- Do NOT use paced records (102, 104, 107, 217).

## Scope (state this honestly)
Subset of 9 records, 2 classes, 1 model. Planned next: all usable records, 5 AAMI classes,
SVM, 1D-CNN, class-imbalance handling, tests, simple UI.

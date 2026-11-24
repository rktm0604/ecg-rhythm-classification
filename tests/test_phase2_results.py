import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "phase2_corrected"


def recompute(y, pred, labels):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


class Phase2ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ART / "experiment_config.json").read_text(encoding="utf-8"))
        cls.results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
        with np.load(ART / "predictions.npz", allow_pickle=False) as z:
            cls.pred = {key: z[key] for key in z.files}
        cls.labels = cls.config["labels"]

    def assert_metrics_match(self, stored, computed):
        self.assertAlmostEqual(stored["accuracy"], computed["accuracy"], places=12)
        self.assertAlmostEqual(stored["macro_f1"], computed["macro_f1"], places=12)
        self.assertEqual(stored["confusion"], computed["confusion"])
        for label in self.labels:
            for metric in ("precision", "recall", "f1-score", "support"):
                self.assertAlmostEqual(stored["report"][label][metric], computed["report"][label][metric], places=12)

    def test_recompute_all_final_test_metrics(self):
        y = self.pred["y_test"].astype(str)
        keys = {"Random Forest": "pred_rf", "1D-CNN": "pred_cnn", "Majority baseline": "pred_majority", "RR-only Logistic Regression": "pred_rr"}
        for name, key in keys.items():
            with self.subTest(model=name):
                self.assert_metrics_match(self.results["final_test"][name], recompute(y, self.pred[key].astype(str), self.labels))

    def test_recompute_validation_metrics(self):
        y = self.pred["y_validation"].astype(str)
        keys = {"Random Forest": "pred_rf_validation", "1D-CNN": "pred_cnn_validation"}
        for name, key in keys.items():
            with self.subTest(model=name):
                self.assert_metrics_match(self.results["validation"][name], recompute(y, self.pred[key].astype(str), self.labels))

    def test_cnn_checkpoint_matches_config(self):
        checkpoint = torch.load(ART / "cnn.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["labels"], self.labels)
        self.assertEqual(checkpoint["window_samples"], self.config["window_samples"])
        self.assertEqual(checkpoint["best_epoch"], self.config["cnn_best_epoch"])
        np.testing.assert_array_equal(np.asarray(self.labels)[self.pred["proba_cnn"].argmax(axis=1)], self.pred["pred_cnn"])

    def test_rf_checkpoint_matches_config_and_predictions(self):
        model = joblib.load(ART / "random_forest.joblib")
        params = self.config["rf_selected_params"]
        self.assertEqual(model.n_features_in_, 34)
        self.assertEqual(set(model.classes_.astype(str)), set(self.labels))
        self.assertEqual(model.get_params()["n_estimators"], params["n_estimators"])
        self.assertEqual(model.get_params()["max_depth"], params["max_depth"])
        self.assertEqual(model.get_params()["min_samples_leaf"], params["min_samples_leaf"])
        self.assertEqual(model.get_params()["class_weight"], params["class_weight"])

    def test_ds2_is_separate_from_selection_records(self):
        fit = set(self.config["fit_records"])
        validation = set(self.config["validation_records"])
        test = set(self.config["test_records"])
        self.assertFalse(fit & validation)
        self.assertFalse(fit & test)
        self.assertFalse(validation & test)
        self.assertEqual(set(np.unique(self.pred["record_test"].astype(str))), test)
        self.assertEqual(self.config["selection_data"], "DS1 validation only; DS2 used once for final reporting")


if __name__ == "__main__":
    unittest.main()

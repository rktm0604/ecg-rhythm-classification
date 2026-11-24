import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "phase4_causal"
PHASE2 = ROOT / "artifacts" / "phase2_corrected"
MODEL_KEYS = {
    "Random Forest": ("pred_rf_validation", "pred_rf"),
    "1D-CNN": ("pred_cnn_validation", "pred_cnn"),
    "Majority baseline": (None, "pred_majority"),
    "RR-only Logistic Regression": (None, "pred_rr"),
}


def calculate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


class Phase4ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ART / "experiment_config.json").read_text(encoding="utf-8"))
        cls.results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
        cls.comparison = json.loads((ART / "comparison.json").read_text(encoding="utf-8"))
        cls.phase2_config = json.loads(
            (PHASE2 / "experiment_config.json").read_text(encoding="utf-8")
        )
        cls.phase2_results = json.loads((PHASE2 / "results.json").read_text(encoding="utf-8"))
        with np.load(ART / "predictions.npz", allow_pickle=False) as archive:
            cls.pred = {key: archive[key] for key in archive.files}
        cls.labels = cls.config["labels"]

    def assert_metrics_match(self, stored, computed):
        self.assertAlmostEqual(stored["accuracy"], computed["accuracy"], places=12)
        self.assertAlmostEqual(stored["macro_f1"], computed["macro_f1"], places=12)
        self.assertEqual(stored["confusion"], computed["confusion"])
        for label in self.labels:
            for metric in ("precision", "recall", "f1-score", "support"):
                self.assertAlmostEqual(
                    stored["report"][label][metric], computed["report"][label][metric], places=12
                )

    def test_recompute_validation_and_final_metrics(self):
        for name, (validation_key, test_key) in MODEL_KEYS.items():
            with self.subTest(model=name, split="validation"):
                if validation_key:
                    expected = calculate(
                        self.pred["y_validation"].astype(str),
                        self.pred[validation_key].astype(str),
                        self.labels,
                    )
                    self.assert_metrics_match(self.results["validation"][name], expected)
            with self.subTest(model=name, split="DS2"):
                expected = calculate(
                    self.pred["y_test"].astype(str), self.pred[test_key].astype(str), self.labels
                )
                self.assert_metrics_match(self.results["final_test"][name], expected)

    def test_split_is_matched_to_phase2_and_ds2_not_selected(self):
        for key in ("fit_records", "validation_records", "test_records", "labels"):
            self.assertEqual(self.config[key], self.phase2_config[key])
        self.assertEqual(
            self.config["selection_data"], "DS1 validation only; DS2 used once for final reporting"
        )
        self.assertIn(self.config["rr_next"], (None, "not computed"))
        self.assertFalse(set(self.config["fit_records"]) & set(self.config["test_records"]))
        self.assertFalse(set(self.config["validation_records"]) & set(self.config["test_records"]))
        self.assertEqual(set(np.unique(self.pred["record_test"].astype(str))), set(self.config["test_records"]))

    def test_checkpoints_and_predictions_match_configuration(self):
        checkpoint = torch.load(ART / "cnn.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["labels"], self.labels)
        self.assertTrue(checkpoint["causal"])
        self.assertEqual(checkpoint["window_samples"], self.config["window_samples"])
        self.assertEqual(checkpoint["best_epoch"], self.config["cnn_best_epoch"])
        np.testing.assert_array_equal(
            np.asarray(self.labels)[self.pred["proba_cnn"].argmax(axis=1)], self.pred["pred_cnn"]
        )
        rf = joblib.load(ART / "random_forest.joblib")
        self.assertEqual(rf.n_features_in_, 33)
        self.assertEqual(rf.get_params()["n_estimators"], self.config["rf_selected_params"]["n_estimators"])
        np.testing.assert_array_equal(
            rf.classes_.astype(str)[self.pred["proba_rf"].argmax(axis=1)], self.pred["pred_rf"]
        )

    def test_comparison_matches_phase2_and_phase4_results(self):
        for model, block in self.comparison.items():
            p2 = self.phase2_results["final_test"][model]
            p4 = self.results["final_test"][model]
            self.assertAlmostEqual(block["phase2_offline_macro_f1"], p2["macro_f1"], places=12)
            self.assertAlmostEqual(block["phase4_causal_macro_f1"], p4["macro_f1"], places=12)
            self.assertAlmostEqual(
                block["macro_f1_difference_causal_minus_offline"],
                p4["macro_f1"] - p2["macro_f1"], places=12,
            )
            self.assertAlmostEqual(block["phase2_offline_accuracy"], p2["accuracy"], places=12)
            self.assertAlmostEqual(block["phase4_causal_accuracy"], p4["accuracy"], places=12)
            for label in self.labels:
                per_class = block["per_class"][label]
                self.assertAlmostEqual(per_class["phase2_f1"], p2["report"][label]["f1-score"], places=12)
                self.assertAlmostEqual(per_class["phase4_f1"], p4["report"][label]["f1-score"], places=12)


if __name__ == "__main__":
    unittest.main()

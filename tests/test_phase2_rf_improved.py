import json
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
OLD_ART = ROOT / "artifacts" / "phase2_corrected"
NEW_ART = ROOT / "artifacts" / "phase2_rf_improved"


class ImprovedRFDS1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((NEW_ART / "frozen_model_config.json").read_text(encoding="utf-8"))
        cls.results = json.loads((NEW_ART / "ds1_results.json").read_text(encoding="utf-8"))
        with np.load(NEW_ART / "ds1_validation_predictions.npz", allow_pickle=False) as z:
            cls.pred = {k: z[k] for k in z.files}

    def test_experiment_uses_same_ds1_group_split_only(self):
        old_config = json.loads((OLD_ART / "experiment_config.json").read_text(encoding="utf-8"))
        self.assertEqual(self.config["fit_records"], old_config["fit_records"])
        self.assertEqual(self.config["validation_records"], old_config["validation_records"])
        fit, val = set(self.config["fit_records"]), set(self.config["validation_records"])
        self.assertFalse(fit & val)
        all_ds1 = set(old_config["train_records"])
        self.assertTrue(fit | val <= all_ds1)
        self.assertEqual(self.config["ds2_status"], "not evaluated yet; this configuration is frozen before final test evaluation")
        self.assertEqual(self.results["fit_records"], self.config["fit_records"])
        self.assertEqual(self.results["validation_records"], self.config["validation_records"])

    def test_every_candidate_metric_recomputes_from_saved_ds1_validation_predictions(self):
        y = self.pred["y_validation"].astype(str)
        labels = self.config["labels"]
        for candidate in self.results["candidates"]:
            with self.subTest(candidate=candidate["name"]):
                # Match the script's stable key transform.
                key = candidate["name"].replace(" ", "_").replace("/", "-").replace("=", "_")
                pred = self.pred[f"pred_{key}"].astype(str)
                metrics = candidate["metrics"]
                self.assertAlmostEqual(metrics["accuracy"], accuracy_score(y, pred), places=12)
                self.assertAlmostEqual(metrics["macro_f1"], f1_score(y, pred, labels=labels, average="macro", zero_division=0), places=12)
                self.assertEqual(metrics["confusion_matrix"], confusion_matrix(y, pred, labels=labels).tolist())

    def test_frozen_decision_retains_existing_if_no_candidate_passes_rule(self):
        self.assertEqual(len(self.results["candidates"]), 18)
        self.assertEqual(sum(c["passes_predeclared_selection_rule"] for c in self.results["candidates"]), 0)
        self.assertTrue(self.config["selected_candidate"]["retained_existing"])
        model = joblib.load(NEW_ART / "random_forest_frozen.joblib")
        original = joblib.load(OLD_ART / "random_forest.joblib")
        self.assertEqual(model.get_params(), original.get_params())
        self.assertEqual(model.n_features_in_, 34)
        self.assertEqual(len(model.estimators_), len(original.estimators_))
        self.assertEqual(len(model.estimators_), 300)
        for copied_tree, original_tree in zip(model.estimators_, original.estimators_):
            for attribute in ("children_left", "children_right", "feature", "threshold", "impurity",
                              "n_node_samples", "weighted_n_node_samples", "value"):
                np.testing.assert_array_equal(
                    getattr(copied_tree.tree_, attribute), getattr(original_tree.tree_, attribute)
                )

    def test_final_ds2_metrics_recompute_and_match_frozen_predictions(self):
        result = json.loads((NEW_ART / "final_ds2_results.json").read_text(encoding="utf-8"))
        verified = json.loads((NEW_ART / "final_ds2_metrics_verified.json").read_text(encoding="utf-8"))
        with np.load(NEW_ART / "predictions_ds2.npz", allow_pickle=False) as z:
            y = z["y_test"].astype(str)
            records = z["record_test"].astype(str)
            old = z["pred_existing_rf"].astype(str)
            selected = z["pred_selected_rf"].astype(str)
        self.assertTrue(np.array_equal(old, selected))
        self.assertEqual(len(y), 45673)
        for pred_key, model_key, verified_key in (
            (old, "existing_phase2_random_forest", "existing_phase2_rf"),
            (selected, "selected_random_forest", "selected_rf"),
        ):
            cm = confusion_matrix(y, pred_key, labels=self.config["labels"]).tolist()
            self.assertEqual(cm, result[model_key]["confusion_matrix"])
            self.assertEqual(cm, verified[verified_key]["confusion_matrix"])
            self.assertAlmostEqual(accuracy_score(y, pred_key), result[model_key]["accuracy"], places=12)
            self.assertAlmostEqual(f1_score(y, pred_key, labels=self.config["labels"], average="macro", zero_division=0), result[model_key]["macro_f1"], places=12)
        mask = records == "232"
        self.assertEqual(int(mask.sum()), 1778)
        rec = verified["record_232"]["selected"]
        self.assertAlmostEqual(accuracy_score(y[mask], selected[mask]), rec["accuracy"], places=12)
        self.assertAlmostEqual(rec["macro_f1_classes_with_true_support"], 0.2516087516087516, places=12)
        self.assertEqual(rec["per_class"]["S"]["support"], 1381)
        self.assertEqual(rec["per_class"]["S"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()

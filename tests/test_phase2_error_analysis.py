import json
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "phase2_corrected"
ERRORS = ART / "error_analysis"


class Phase2ErrorAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ART / "experiment_config.json").read_text(encoding="utf-8"))
        cls.results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
        cls.analysis = json.loads((ERRORS / "error_analysis.json").read_text(encoding="utf-8"))
        with np.load(ART / "predictions.npz", allow_pickle=False) as archive:
            cls.predictions = {key: archive[key] for key in archive.files}
        cls.labels = cls.config["labels"]

    def test_saved_prediction_analysis_reproduces_results(self):
        y = self.predictions["y_test"].astype(str)
        self.assertTrue(self.analysis["reproduced_saved_results_json"])
        for model, key in (("Random Forest", "pred_rf"), ("1D-CNN", "pred_cnn")):
            pred = self.predictions[key].astype(str)
            stored = self.analysis["models"][model]
            np.testing.assert_array_equal(
                np.asarray(stored["confusion_matrix"]), confusion_matrix(y, pred, labels=self.labels)
            )
            self.assertAlmostEqual(stored["accuracy"], accuracy_score(y, pred), places=12)
            self.assertAlmostEqual(
                stored["macro_f1"],
                f1_score(y, pred, labels=self.labels, average="macro", zero_division=0),
                places=12,
            )
            self.assertEqual(stored["confusion_matrix"], self.results["final_test"][model]["confusion"])

    def test_classwise_one_vs_rest_counts_recompute(self):
        y = self.predictions["y_test"].astype(str)
        for model, key in (("Random Forest", "pred_rf"), ("1D-CNN", "pred_cnn")):
            pred = self.predictions[key].astype(str)
            class_metrics = self.analysis["models"][model]["class_metrics_one_vs_rest"]
            cm = confusion_matrix(y, pred, labels=self.labels)
            for index, label in enumerate(self.labels):
                tp = int(cm[index, index])
                fp = int(cm[:, index].sum() - tp)
                fn = int(cm[index, :].sum() - tp)
                self.assertEqual(class_metrics[label]["tp"], tp)
                self.assertEqual(class_metrics[label]["fp"], fp)
                self.assertEqual(class_metrics[label]["fn"], fn)
                self.assertEqual(class_metrics[label]["support"], int(cm[index, :].sum()))

    def test_record_metrics_and_examples_are_traceable(self):
        records = self.predictions["record_test"].astype(str)
        y = self.predictions["y_test"].astype(str)
        self.assertEqual(set(records), set(self.config["test_records"]))
        for model, key in (("Random Forest", "pred_rf"), ("1D-CNN", "pred_cnn")):
            pred = self.predictions[key].astype(str)
            per_record = self.analysis["models"][model]["record_metrics"]
            self.assertEqual(sum(row["n_beats"] for row in per_record), len(y))
            self.assertEqual({row["record"] for row in per_record}, set(records))
            for row in per_record:
                mask = records == row["record"]
                self.assertEqual(row["n_beats"], int(mask.sum()))
                self.assertAlmostEqual(row["accuracy"], accuracy_score(y[mask], pred[mask]), places=12)
            examples = self.analysis["representative_examples"][model]
            self.assertGreaterEqual(len(examples), 3)
            for example in examples:
                idx = example["row_index"]
                self.assertEqual(example["record"], records[idx])
                self.assertEqual(example["true"], y[idx])
                self.assertEqual(example["predicted"], pred[idx])
                self.assertEqual(len(example["waveform"]), 200)
                self.assertEqual(example["waveform_source"], "pipeline.extract_record_beats normalized 200-sample window")


if __name__ == "__main__":
    unittest.main()

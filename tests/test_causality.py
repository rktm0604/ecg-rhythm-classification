import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import LABEL_MAP
from artifacts.phase4_causal.train_eval_phase4 import causal_record_features


class CausalFeatureTests(unittest.TestCase):
    def test_rr_uses_only_valid_heartbeat_positions(self):
        signal = np.arange(1200, dtype=np.float32)
        positions = [100, 200, 250, 300, 400, 500, 600, 700]
        symbols = ["N", "N", "+", "V", "N", "N", "N", "N"]
        rows, _ = causal_record_features(signal, positions, symbols)
        row = next(x for x in rows if x["sample"] == 300)
        self.assertEqual(row["rr_prev"], 100)
        self.assertEqual(row["rr_local"], 100)
        self.assertEqual(row["features"].shape, (33,))
        self.assertNotIn("rr_next", row)

    def test_local_context_uses_up_to_four_preceding_intervals_only(self):
        signal = np.zeros(1500, dtype=np.float32)
        positions = [100, 200, 300, 500, 600, 800, 900]
        rows, _ = causal_record_features(signal, positions, ["N"] * len(positions))
        row = next(x for x in rows if x["sample"] == 800)
        # Four intervals ending at 800: 200, 100, 200, 100.
        self.assertEqual(row["rr_prev"], 200)
        self.assertEqual(row["rr_local"], 150)

    def test_future_annotation_and_waveform_perturbations_do_not_change_features(self):
        signal = np.arange(1600, dtype=np.float32)
        positions = np.array([100, 200, 300, 500, 600, 800, 1000, 1200])
        symbols = np.array(["N"] * len(positions))
        baseline, _ = causal_record_features(signal, positions, symbols)
        target = next(x for x in baseline if x["sample"] == 800)

        altered_positions = positions.copy()
        altered_positions[6:] += 50
        altered_symbols = symbols.copy()
        altered_symbols[6:] = ["+", "~"]
        altered_signal = signal.copy()
        altered_signal[801:] = -99999
        changed, _ = causal_record_features(altered_signal, altered_positions, altered_symbols)
        changed_target = next(x for x in changed if x["sample"] == 800)
        np.testing.assert_array_equal(target["features"], changed_target["features"])
        np.testing.assert_array_equal(target["wave"], changed_target["wave"])

    def test_missing_previous_heartbeat_and_record_start_are_explicit(self):
        signal = np.zeros(700, dtype=np.float32)
        rows, excluded = causal_record_features(signal, [100, 250, 400, 600], ["N"] * 4)
        self.assertEqual(excluded.get("N:missing_previous_heartbeat"), 1)
        # R=250 has a complete past window; the record-start beat has no predecessor.
        self.assertIn(250, [x["sample"] for x in rows])
        rows2, excluded2 = causal_record_features(signal, [40, 150, 400, 600], ["N"] * 4)
        self.assertEqual(excluded2.get("N:causal_window_boundary"), 1)

    def test_existing_mapping_is_used(self):
        self.assertEqual(LABEL_MAP["V"], "V")
        self.assertEqual(LABEL_MAP["/"], "Q")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

import dataset_audit
import pipeline


class PipelineRRTests(unittest.TestCase):
    def test_non_beat_annotations_do_not_become_rr_endpoints(self):
        signal = np.arange(1200, dtype=np.float32)
        positions = [100, 200, 250, 300, 400, 500, 600, 700]
        symbols = ["N", "N", "+", "V", "N", "N", "N", "N"]
        rows, _, _, _ = pipeline.extract_record_beats(signal, positions, symbols)
        row = next(item for item in rows if item["sample"] == 300)
        self.assertEqual(row["rr_prev"], 100)
        self.assertEqual(row["rr_next"], 100)

    def test_rr_local_mean_two_prior_two_future_intervals(self):
        signal = np.arange(1300, dtype=np.float32)
        positions = [100, 200, 300, 500, 600, 800, 900]
        rows, _, _, _ = pipeline.extract_record_beats(signal, positions, ["N"] * 7)
        row = next(item for item in rows if item["sample"] == 500)
        self.assertEqual(row["rr_prev"], 200)
        self.assertEqual(row["rr_next"], 100)
        self.assertAlmostEqual(row["rr_local"], 150.0)

    def test_missing_neighbor_beats_are_excluded_with_reasons(self):
        signal = np.zeros(1000, dtype=np.float32)
        rows, excluded, _, _ = pipeline.extract_record_beats(
            signal, [100, 300, 500], ["N", "N", "V"]
        )
        self.assertEqual([row["sample"] for row in rows], [300])
        self.assertEqual(excluded[("N", "missing_previous_heartbeat")], 1)
        self.assertEqual(excluded[("V", "missing_next_heartbeat")], 1)

    def test_waveform_boundary_beats_are_excluded(self):
        signal = np.zeros(1000, dtype=np.float32)
        for peak in (89, 910):
            with self.subTest(peak=peak):
                positions = [max(0, peak - 200), peak, min(999, peak + 200)]
                rows, excluded, _, _ = pipeline.extract_record_beats(
                    signal, positions, ["N", "V", "N"]
                )
                self.assertFalse(any(row["sample"] == peak for row in rows))
                self.assertEqual(excluded[("V", "waveform_boundary")], 1)

    def test_nonpositive_rr_intervals_are_excluded(self):
        signal = np.zeros(1200, dtype=np.float32)
        rows, excluded, _, _ = pipeline.extract_record_beats(
            signal, [100, 300, 250, 500], ["N", "V", "S", "N"]
        )
        self.assertFalse(any(row["sample"] in (300, 250) for row in rows))
        self.assertEqual(excluded[("V", "invalid_rr_interval")], 1)
        self.assertEqual(excluded[("S", "invalid_rr_interval")], 1)

    def test_valid_window_is_exactly_200_samples(self):
        signal = np.arange(1000, dtype=np.float32)
        rows, _, _, _ = pipeline.extract_record_beats(
            signal, [100, 300, 500], ["N", "V", "N"]
        )
        self.assertEqual(rows[0]["wave"].shape, (200,))

    def test_audit_record_uses_extraction_eligibility(self):
        signal = np.zeros(1000, dtype=np.float32)
        positions = np.array([100, 300, 350, 500, 700])
        symbols = np.array(["N", "N", "+", "V", "N"])

        class Header:
            fs = 360
            sig_len = 1000

        with patch.object(dataset_audit, "_read_record", return_value=(signal, positions, symbols)), \
             patch.object(dataset_audit.wfdb, "rdheader", return_value=Header()):
            row = dataset_audit.audit_record("mock", "DS1")

        extracted, _, _, _ = pipeline.extract_record_beats(signal, positions, symbols)
        self.assertEqual(row["eligible_beats"], len(extracted))
        self.assertEqual(row["N_eligible"] + row["V_eligible"], len(extracted))
        self.assertEqual(row["unmapped_annotations"], 1)
        self.assertEqual(row["exclusion_reasons"], {
            "missing_previous_heartbeat": 1,
            "missing_next_heartbeat": 1,
        })


if __name__ == "__main__":
    unittest.main()

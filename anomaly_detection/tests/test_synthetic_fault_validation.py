"""Tests for the labeled synthetic-fault regression benchmark."""

from __future__ import annotations

import unittest

import pandas as pd

from energy_prediction_romania.run_synthetic_fault_validation import inject_faults


class SyntheticFaultInjectionTests(unittest.TestCase):
    def test_injects_labeled_non_overlapping_events(self) -> None:
        timestamp = pd.date_range("2026-06-01 08:00", periods=90, freq="h", tz="UTC")
        frame = pd.DataFrame({
            "Timestamp": timestamp,
            "GHI": 500.0,
            "AC_Output": 40.0,
            "Predicted_AC_Output": 42.0,
        })
        injected, manifest = inject_faults(frame, events_per_type=2, event_rows=2)
        self.assertEqual(len(manifest), 6)
        self.assertEqual(injected["Synthetic_Fault_Label"].ne("NONE").sum(), 12)
        self.assertEqual(injected["Synthetic_Fault_Event_ID"].dropna().nunique(), 6)
        self.assertTrue((
            injected.loc[
                injected["Synthetic_Fault_Label"].eq("SIMULATED_DAYLIGHT_OUTAGE"), "AC_Output"
            ] == 0.0
        ).all())


if __name__ == "__main__":
    unittest.main()

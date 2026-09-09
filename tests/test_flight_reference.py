import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from firetrack.flight_reference import (
    FlightTrack, attach_reference, calibration_clock_delta, comparison_metrics, enu_frame,
    load_reference, read_flight_track, sample_track, to_calibration_frame,
)
from firetrack.results import load_trajectory, trajectory_file


FRAME = {"axes": "ENU", "origin_lat_deg": 40.0, "origin_lon_deg": -86.0, "altitude_reference": "AMSL"}
CALIBRATION = {"world_frame": FRAME, "cameras": [{"video": "cam1"}, {"video": "cam2"}]}


class FlightReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "tracking_uploads/run1"
        self.output.mkdir(parents=True)
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text(json.dumps(CALIBRATION))
        self.track = FlightTrack(np.array([100.0, 100.5, 101.0]), np.array([[40, -86, 150], [40, -86, 151], [40, -86, 152]]), 0.01)
        self.epochs = np.array([100.0, 100.25, 100.5, 100.75, 101.0])
        ref = sample_track(self.track, self.epochs, FRAME)
        self.reconstruction = ref + [1, 0, 0]
        self.write_trajectory()

    def write_trajectory(self):
        np.savez(self.output / "trajectory.npz", epoch_times_s=self.epochs,
                 trajectory_raw=self.reconstruction, trajectory_smooth=self.reconstruction,
                 gt_drone=np.full_like(self.reconstruction, np.nan),
                 n_views=np.full(len(self.epochs), 3), reproj_errors_px=np.ones(len(self.epochs)))

    def attach(self, paths=None):
        with patch("firetrack.flight_reference.read_flight_track", return_value=self.track):
            return attach_reference(self.output, paths or [Path("flight.BIN")], self.calibration)

    def test_coordinate_order_origin_and_absolute_altitude(self):
        points = to_calibration_frame(np.array([[40, -86, 150], [40.00001, -86, 151], [40, -85.99999, 152]]), FRAME)
        np.testing.assert_allclose(points[0], [0, 0, 150], atol=1e-8)
        self.assertGreater(points[1, 1], 1)
        self.assertGreater(points[2, 0], 0.8)
        np.testing.assert_allclose(points[:, 2], [150, 151, 152])

    def test_frame_validation_rejects_unknown_and_inconsistent_origins(self):
        self.assertEqual(enu_frame(CALIBRATION), FRAME)
        legacy = {"frame": "ENU about lat 40.0 lon -86.0, U = AMSL; x_cam = R (X_world - p)", "cameras": [{"video": "cam1"}]}
        self.assertEqual(enu_frame(legacy), FRAME)
        for bad in ({"cameras": [{"source": {"world_frame": "XKF1 local NEU"}}]}, {"cameras": []}):
            with self.assertRaises(ValueError):
                enu_frame(bad)
        with self.assertRaises(ValueError):
            enu_frame({"cameras": [{"source": {"world_frame": FRAME}}, {"source": {"world_frame": {**FRAME, "origin_lat_deg": 41}}}]})

    def test_no_extrapolation_or_interpolation_across_large_gaps(self):
        track = FlightTrack(np.array([100.0, 100.5, 105.0]), self.track.lat_lon_alt, 0)
        result = sample_track(track, np.array([99, 100, 100.25, 102, 105, 106]), FRAME)
        self.assertTrue(np.isnan(result[[0, 3, 5]]).all())
        np.testing.assert_allclose(result[[1, 2, 4], 2], [150, 150.5, 152])

    def test_reference_does_not_modify_or_fit_the_reconstruction(self):
        path = self.output / "trajectory.npz"
        original = path.read_bytes()
        metadata = self.attach()
        self.assertEqual(path.read_bytes(), original)
        self.assertAlmostEqual(metadata["metrics"]["rmse_m"], 1.0)
        self.assertEqual(metadata["trajectory_sha256"], hashlib.sha256(original).hexdigest())
        loaded = load_trajectory(self.root, "tracking_uploads/run1")
        self.assertIsNone(loaded["gt"])
        self.assertFalse(loaded["metrics"]["has_gt"])
        self.assertEqual(len(loaded["reference"]["points"]), len(self.epochs))
        self.assertTrue(trajectory_file(self.root, "tracking_uploads/run1", "comparison").is_file())

    def test_calibration_lag_shifts_only_the_log_query(self):
        lag = 0.2
        track = FlightTrack(self.epochs-lag,
                            np.column_stack([np.full(len(self.epochs), 40.),
                                             np.full(len(self.epochs), -86.),
                                             np.arange(len(self.epochs))+150.]), 0.01)
        calibration = {**CALIBRATION, "calibration_clock": {
            "camera_minus_log_s": lag, "applies_to_tracking": False,
        }}
        self.calibration.write_text(json.dumps(calibration))
        self.reconstruction = sample_track(track, self.epochs-lag, FRAME)
        self.write_trajectory()
        with patch("firetrack.flight_reference.read_flight_track", return_value=track):
            metadata = attach_reference(self.output, [Path("flight.BIN")], self.calibration)
        self.assertEqual(calibration_clock_delta(calibration), lag)
        self.assertAlmostEqual(metadata["calibration_clock_delta_s"], lag)
        self.assertAlmostEqual(metadata["metrics"]["rmse_m"], 0.0)
        with np.load(self.output/"flight_reference.npz") as data:
            np.testing.assert_array_equal(data["epoch_times_s"], self.epochs)

    def test_changed_reconstruction_invalidates_reference_and_export(self):
        self.attach()
        self.reconstruction += 1
        self.write_trajectory()
        self.assertIn("error", load_reference(self.output / "trajectory.npz", self.epochs))
        with self.assertRaises(FileNotFoundError):
            trajectory_file(self.root, "tracking_uploads/run1", "comparison")

    def test_ambiguous_or_nonoverlapping_logs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multiple flight logs"):
            self.attach([Path("one.BIN"), Path("two.BIN")])
        shifted = FlightTrack(self.track.epochs + 100, self.track.lat_lon_alt, 0)
        with patch("firetrack.flight_reference.read_flight_track", return_value=shifted), self.assertRaisesRegex(ValueError, "No flight log overlaps"):
            attach_reference(self.output, [Path("other.BIN")], self.calibration)
        self.assertFalse((self.output / "flight_reference.npz").exists())

    def test_snapshot_wins_over_current_calibration_and_marks_calibration_overlap(self):
        calibration = {"world_frame": FRAME, "cameras": [{"source": {"window_unix": [100, 102]}}]}
        (self.output / "summary.json").write_text(json.dumps({"calibration_snapshot": calibration}))
        self.calibration.write_text("{}")
        metadata = self.attach()
        self.assertEqual(metadata["world_frame"], FRAME)
        self.assertTrue(any("not independent" in w for w in metadata["warnings"]))
        self.assertFalse(any("current calibration" in w for w in metadata["warnings"]))

    def test_metrics_exclude_missing_reconstruction(self):
        reference = np.array([[0., 0, 0], [0, 0, 0], [np.nan, np.nan, np.nan]])
        reconstruction = np.array([[3., 0, 4], [np.nan, np.nan, np.nan], [0, 0, 0]])
        metrics = comparison_metrics(reconstruction, reference)
        self.assertEqual(metrics["n_compared"], 1)
        self.assertEqual(metrics["n_reference"], 2)
        self.assertEqual(metrics["rmse_m"], 5)
        self.assertEqual(metrics["horizontal_rmse_m"], 3)
        self.assertEqual(metrics["vertical_rmse_m"], 4)

    def test_gps_time_is_converted_to_utc(self):
        def message(kind, **fields):
            return SimpleNamespace(get_type=lambda: kind, **fields)
        messages = iter([
            message("GPS", TimeUS=1000000, GWk=2433, GMS=507000000, Status=3, I=0),
            message("POS", TimeUS=1000000, Lat=40, Lng=-86, Alt=150),
            message("GPS", TimeUS=2000000, GWk=2433, GMS=507001000, Status=3, I=0),
            message("POS", TimeUS=2000000, Lat=40, Lng=-86, Alt=151),
        ])
        reader = SimpleNamespace(recv_match=lambda **kwargs: next(messages, None), close=lambda: None)
        with patch("pymavlink.DFReader.DFReader_binary", return_value=reader):
            track = read_flight_track(Path("flight.BIN"))
        expected = 315964800 + 2433 * 604800 + 507000 - 18
        np.testing.assert_array_equal(track.epochs, [expected, expected + 1])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firetrack.flight_reference import load_reference
from firetrack.webui import WebConfig, _tracking_triangulate_fn


class AutomaticFlightReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = WebConfig(self.root)
        self.make_file("tracking_uploads/run1/cam1/video.mp4")
        self.output = self.root / "triangulation/tracking_uploads/run1"
        self.output.mkdir(parents=True)
        self.trajectory = self.make_file("triangulation/tracking_uploads/run1/trajectory.npz")

    def make_file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged")
        return path

    def test_reuses_all_calibration_logs_after_triangulation(self):
        logs = [self.make_file("calibration_logs/one.BIN"), self.make_file("calibration_logs/two.bin")]
        self.make_file("calibration_logs/ignored.txt")
        self.make_file("tracking_logs/run1/old.BIN")
        calls = []
        with patch("firetrack.webui.triangulate_uploads", side_effect=lambda **kw: calls.append("triangulate")), \
                patch("firetrack.webui.attach_reference", side_effect=lambda *args: calls.append("reference")) as attach:
            _tracking_triangulate_fn(self.cfg, "run1")()
        self.assertEqual(calls, ["triangulate", "reference"])
        attach.assert_called_once_with(self.output, logs, self.cfg.uploads_calibration_json)

    def test_older_per_run_logs_remain_usable(self):
        log = self.make_file("tracking_logs/run1/old.BIN")
        self.make_file("tracking_logs/run2/unrelated.BIN")
        with patch("firetrack.webui.triangulate_uploads"), patch("firetrack.webui.attach_reference") as attach:
            _tracking_triangulate_fn(self.cfg, "run1")()
        self.assertEqual(attach.call_args.args[1], [log])

    def test_missing_logs_does_not_fail_reconstruction(self):
        with patch("firetrack.webui.triangulate_uploads"), patch("firetrack.webui.attach_reference") as attach:
            _tracking_triangulate_fn(self.cfg, "run1")()
        attach.assert_not_called()
        self.assertIn("No stored flight logs", load_reference(self.trajectory, None)["error"])
        self.assertEqual(self.trajectory.read_bytes(), b"unchanged")

    def test_reference_failure_removes_old_outputs_but_keeps_reconstruction(self):
        self.make_file("calibration_logs/one.BIN")
        self.make_file("triangulation/tracking_uploads/run2/flight_reference.json")
        for message in ("No flight log overlaps", "Multiple flight logs overlap", "Invalid coordinate frame"):
            for name in ("flight_reference.json", "flight_reference.npz", "flight_comparison.csv"):
                self.make_file("triangulation/tracking_uploads/run1/" + name)
            with patch("firetrack.webui.triangulate_uploads"), \
                    patch("firetrack.webui.attach_reference", side_effect=ValueError(message)):
                _tracking_triangulate_fn(self.cfg, "run1")()
            self.assertIn(message, json.loads((self.output / "flight_reference.json").read_text())["error"])
            self.assertFalse((self.output / "flight_reference.npz").exists())
            self.assertFalse((self.output / "flight_comparison.csv").exists())
            self.assertEqual(self.trajectory.read_bytes(), b"unchanged")
        self.assertEqual((self.output.parent / "run2/flight_reference.json").read_bytes(), b"unchanged")

    def test_failed_triangulation_does_not_start_comparison(self):
        with patch("firetrack.webui.triangulate_uploads", side_effect=RuntimeError("failed")), \
                patch("firetrack.webui.attach_reference") as attach:
            with self.assertRaises(RuntimeError):
                _tracking_triangulate_fn(self.cfg, "run1")()
        attach.assert_not_called()


if __name__ == "__main__":
    unittest.main()

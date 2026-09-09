import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firetrack.webui import WebConfig, clear_tracking_outputs, serve_webui


class ClearTrackingOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = WebConfig(self.root)

    def make_file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve me")
        return path

    def test_only_selected_run_outputs_are_removed(self):
        preserved = [self.make_file(name) for name in (
            "tracking_uploads/run1/cam1/video.mp4",
            "tracking_uploads/run1/cam1/metadata.json",
            "tracking_uploads_run1_clicks.json",
            "uploads_calibration.json",
            "detections/tracking_uploads/run2/cam1/centroids.npz",
            "triangulation/tracking_uploads/run2/trajectory.npz",
            "detections/uploads/cam1/centroids.npz",
            "detections/mocap_raw/run1/cam1/centroids.npz",
            "triangulation/mocap_raw/run1/trajectory.npz",
            "detections/mocap_run/cam1/centroids.npz",
            "triangulation/mocap_run/trajectory.npz",
        )]
        removed = [self.make_file(name) for name in (
            "detections/tracking_uploads/run1/cam1/centroids.npz",
            "detections/tracking_uploads/run1/cam2/masks.npz",
            "triangulation/tracking_uploads/run1/trajectory.npz",
        )]
        clear_tracking_outputs(self.cfg, "run1")
        clear_tracking_outputs(self.cfg, "run1")
        self.assertTrue(all(not p.exists() for p in removed))
        self.assertTrue(all(p.read_bytes() == b"preserve me" for p in preserved))

    def test_invalid_runs_and_symlinks_do_not_delete_outputs(self):
        self.make_file("tracking_uploads/run1/cam1/video.mp4")
        kept = self.make_file("detections/tracking_uploads/run1/cam1/centroids.npz")
        for run in (None, "", "../run1", "run1.zip", "missing", "mocap_raw"):
            with self.subTest(run=run), self.assertRaises(ValueError):
                clear_tracking_outputs(self.cfg, run)
        target = self.root / "triangulation/tracking_uploads/run1"
        target.parent.mkdir(parents=True)
        target.symlink_to(kept.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            clear_tracking_outputs(self.cfg, "run1")
        self.assertTrue(kept.exists())

    def test_flat_run_does_not_delete_nested_run_outputs(self):
        self.make_file("tracking_uploads/cam1/video.mp4")
        removed = self.make_file("detections/tracking_uploads/cam1/centroids.npz")
        kept = self.make_file("detections/tracking_uploads/other/cam1/centroids.npz")
        trajectory = self.make_file("triangulation/tracking_uploads/trajectory.npz")
        other = self.make_file("triangulation/tracking_uploads/other/trajectory.npz")
        clear_tracking_outputs(self.cfg, "default")
        self.assertFalse(removed.exists())
        self.assertFalse(trajectory.exists())
        self.assertTrue(kept.exists())
        self.assertTrue(other.exists())

    def test_clear_endpoint_rejects_busy_runner(self):
        self.make_file("tracking_uploads/run1/cam1/video.mp4")
        output = self.make_file("detections/tracking_uploads/run1/cam1/centroids.npz")
        with patch("firetrack.webui.ThreadingHTTPServer") as server, patch("firetrack.webui.JobRunner") as runner:
            runner.return_value.start.return_value = False
            serve_webui(work_root=self.root, host="127.0.0.1", port=0)
            handler_type = server.call_args.args[1]
            handler = object.__new__(handler_type)
            with patch.object(handler, "_json") as response:
                handler._handle_tracking_outputs_clear({"run": "run1"})
                self.assertEqual(response.call_args.args[0], 409)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

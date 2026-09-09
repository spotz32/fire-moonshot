import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firetrack.clicks import ClicksService
from firetrack.webui import WebConfig, _ClicksRegistry, clear_calibration, serve_webui


class ClearCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = WebConfig(self.root / "work")

    def make_file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve me")
        return path

    def test_clears_all_calibration_data_only_and_can_repeat(self):
        removed = [self.make_file("work/" + name) for name in (
            "uploads/cam1/video.mp4", "uploads/cam1/metadata.json",
            "uploads/cam1/camera.json", "uploads/cam1/gps.csv", "uploads/cam1/imu.csv",
            "uploads/manifest.json", "calibration_logs/one.BIN",
            "calibration_logs/two.bin", "uploads_clicks.json", "uploads_calibration.json",
            "detections/uploads/cam1/centroids.npz", "triangulation/uploads/trajectory.npz",
        )]
        preserved = [self.make_file(name) for name in (
            "original/calibration.zip", "work/webui_state.json",
            "work/tracking_uploads/run1/cam1/video.mp4",
            "work/actual_uploads/run1/cam1/video.mp4", "work/tracking_uploads_run1_clicks.json",
            "work/tracking_logs/run1/log.BIN", "work/detections/tracking_uploads/run1/centroids.npz",
            "work/triangulation/tracking_uploads/run1/summary.json",
            "work/triangulation/actual_uploads/run1/flight_reference.json",
            "work/dataset_uploads/run1/video.mp4", "work/formatted/run1/video.mp4",
            "work/clicks.json", "work/mocap_raw_clicks.json",
            "work/detections/mocap_raw/run1/centroids.npz", "work/triangulation/run1/trajectory.npz",
        )]
        clear_calibration(self.cfg)
        clear_calibration(self.cfg)
        self.assertTrue(all(not p.exists() for p in removed))
        self.assertTrue(all(p.read_bytes() == b"preserve me" for p in preserved))

    def test_rejects_symlinks_before_deleting_anything(self):
        kept = self.make_file("work/uploads/cam1/video.mp4")
        outside = self.make_file("other/uploads/centroids.npz")
        (self.cfg.work_root / "detections").symlink_to(outside.parent.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            clear_calibration(self.cfg)
        self.assertTrue(kept.exists())
        self.assertTrue(outside.exists())

    def test_nested_symlink_does_not_remove_external_files(self):
        outside = self.make_file("original/video.mp4")
        self.cfg.uploads_root.mkdir(parents=True)
        (self.cfg.uploads_root / "linked").symlink_to(outside.parent, target_is_directory=True)
        clear_calibration(self.cfg)
        self.assertEqual(outside.read_bytes(), b"preserve me")

    def test_annotation_cache_is_invalidated_without_affecting_other_sources(self):
        registry = _ClicksRegistry(self.cfg)
        service = ClicksService([], self.cfg.uploads_clicks_json)
        service.rows.append({"click_x": 1})
        other = object()
        registry._services = {"upload": service, "tracking": other}
        registry.active = service
        registry.discard_calibration()
        clear_calibration(self.cfg)
        self.assertIsNone(registry.active)
        self.assertEqual(registry._services, {"tracking": other})
        with self.assertRaises(IndexError):
            service.apply_click(0, 1, 2, 0)
        self.assertFalse(self.cfg.uploads_clicks_json.exists())
        registry.active = other
        registry.discard_calibration()
        self.assertIs(registry.active, other)

    def test_endpoint_uses_job_lock_and_does_not_delete_when_busy(self):
        kept = self.make_file("work/uploads/cam1/video.mp4")
        with patch("firetrack.webui.ThreadingHTTPServer") as server, patch("firetrack.webui.JobRunner") as runner:
            runner.return_value.start.return_value = False
            serve_webui(work_root=self.cfg.work_root, host="127.0.0.1", port=0)
            handler = object.__new__(server.call_args.args[1])
            with patch.object(handler, "_json") as response:
                handler._handle_calibration_clear()
                self.assertEqual(response.call_args.args[0], 409)
            self.assertTrue(kept.exists())
            runner.return_value.start.return_value = True
            with patch.object(handler, "_json") as response:
                handler._handle_calibration_clear()
                self.assertEqual(response.call_args.args[0], 202)
            label, job = runner.return_value.start.call_args.args
            self.assertEqual(label, "upload:clear-calibration")
            job()
            self.assertFalse(kept.exists())


if __name__ == "__main__":
    unittest.main()

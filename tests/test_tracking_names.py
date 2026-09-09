import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firetrack import results
from firetrack.sources import LEGACY_TRACKING_UPLOAD_SUBDIR, tracking_source
from firetrack.webui import (
    WebConfig,
    _set_selected_tracking_run,
    _tracking_prefix,
    _tracking_root,
    _tracking_runs,
    clear_tracking_outputs,
    serve_webui,
)


class TrackingNamesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = WebConfig(self.root)
        self.legacy = LEGACY_TRACKING_UPLOAD_SUBDIR

    def make_file(self, name, data=b"unchanged"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_old_and_new_runs_are_discovered_without_moving_data(self):
        old_video = self.make_file(f"{self.legacy}/old_run/cam1/video.mp4")
        self.make_file("tracking_uploads/new_run/cam1/video.mp4")
        old_clicks = self.make_file(f"{self.legacy}_old_run_clicks.json")
        self.assertEqual(_tracking_runs(self.cfg), ["new_run", "old_run"])
        _set_selected_tracking_run(self.cfg, "old_run")
        self.assertEqual(_tracking_root(self.cfg, "old_run"), old_video.parent.parent)
        self.assertEqual(_tracking_prefix(self.cfg, "old_run"), Path(self.legacy) / "old_run")
        self.assertEqual(self.cfg.tracking_run_clicks_json("old_run"), old_clicks)
        self.assertEqual(_tracking_prefix(self.cfg, "new_run"), Path("tracking_uploads/new_run"))
        self.assertEqual(_tracking_root(self.cfg, "next_run"), self.root / "tracking_uploads/next_run")
        self.assertTrue(old_video.exists())

    def test_results_advertise_new_names_and_resolve_old_files(self):
        rel = f"{self.legacy}/run1/cam1"
        self.make_file(f"detections/{rel}/centroids.npz")
        self.make_file(f"detections/{rel}/summary.json", json.dumps({"relative_dir": rel, "label": "cam1"}).encode())
        traj = self.make_file(f"triangulation/{self.legacy}/run1/trajectory.npz")
        rows = results.list_detections(self.cfg.detections_root)
        self.assertEqual(rows[0]["dir"], "tracking_uploads/run1/cam1")
        self.assertEqual(rows[0]["run"], "run1")
        rows = results.list_trajectories(self.cfg.triangulation_root)
        self.assertEqual(rows[0]["dir"], "tracking_uploads/run1")
        self.assertEqual(results.trajectory_file(self.cfg.triangulation_root, rows[0]["dir"], "npz"), traj)
        with self.assertRaises(ValueError):
            results.trajectory_file(self.cfg.triangulation_root, "../../outside", "npz")

    def test_download_name_changes_but_payload_does_not(self):
        original = b"NPZ payload is preserved byte for byte"
        self.make_file(f"triangulation/{self.legacy}/run1/trajectory.npz", original)
        with patch("firetrack.webui.ThreadingHTTPServer") as server:
            serve_webui(work_root=self.root, host="127.0.0.1", port=0)
            handler = object.__new__(server.call_args.args[1])
            handler.wfile = io.BytesIO()
            with patch.object(handler, "send_response"), patch.object(handler, "end_headers"), patch.object(handler, "send_header") as headers:
                handler._serve_trajectory_download({"dir": ["tracking_uploads/run1"], "fmt": ["npz"]})
                headers.assert_any_call("Content-Disposition", 'attachment; filename="tracking_run1_trajectory.npz"')
            self.assertEqual(handler.wfile.getvalue(), original)
        for fmt in ("npz", "csv"):
            self.assertEqual(results.trajectory_download_name(f"{self.legacy}/run1", f"trajectory.{fmt}"), f"tracking_run1_trajectory.{fmt}")
        self.assertEqual(results.trajectory_download_name("mocap_raw/run1", "trajectory.npz"), "mocap_raw_run1_trajectory.npz")

    def test_clear_legacy_run_keeps_new_run_and_inputs(self):
        video = self.make_file(f"{self.legacy}/old_run/cam1/video.mp4")
        old = self.make_file(f"detections/{self.legacy}/old_run/cam1/centroids.npz")
        new = self.make_file("detections/tracking_uploads/new_run/cam1/centroids.npz")
        clear_tracking_outputs(self.cfg, "old_run")
        self.assertFalse(old.exists())
        self.assertTrue(video.exists())
        self.assertTrue(new.exists())

    def test_request_compatibility_does_not_change_mocap_sources(self):
        self.assertEqual(tracking_source("actual"), "tracking")
        for source in ("tracking", "upload", "mocap_raw", "dataset"):
            self.assertEqual(tracking_source(source), source)


if __name__ == "__main__":
    unittest.main()

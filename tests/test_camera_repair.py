import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firetrack.webui import (
    WebConfig, clear_tracking_outputs, clear_calibration_camera_outputs, serve_webui,
)


class CameraRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = WebConfig(self.root)
        for camera in ('cam1', 'cam2', 'cam3'):
            for name in (f'tracking_uploads/run1/{camera}/video.mp4', f'uploads/{camera}/video.mp4',
                         f'uploads/{camera}/camera.json', f'uploads/{camera}/metadata.json',
                         f'detections/tracking_uploads/run1/{camera}/centroids.npz',
                         f'detections/uploads/{camera}/centroids.npz'):
                self.make_file(name)
        for name in ('triangulation/tracking_uploads/run1/trajectory.npz',
                     'triangulation/tracking_uploads/run1/flight_reference.json',
                     'triangulation/uploads/trajectory.npz', 'uploads_calibration.json',
                     'calibration_logs/flight.BIN', 'webui_state.json',
                     'detections/mocap_raw/run1/cam3/centroids.npz', 'original_calibration.json'):
            self.make_file(name)
        rows = [{'label': c, 'click_x': 10, 'click_y': 20, 'click_frame': 0, 'approved': True}
                for c in ('cam1', 'cam2', 'cam3')]
        for path in (self.cfg.uploads_clicks_json, self.cfg.tracking_run_clicks_json('run1')):
            path.write_text(json.dumps(rows))

    def make_file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve me')
        return path

    def snapshot(self):
        return {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def assert_only_removed(self, before, removed):
        for path, data in before.items():
            if str(path.relative_to(self.root)) in removed:
                self.assertFalse(path.exists(), str(path))
            else:
                self.assertEqual(path.read_bytes(), data, str(path))

    def test_tracking_clear_only_selected_camera_and_reconstruction(self):
        before = self.snapshot()
        clear_tracking_outputs(self.cfg, 'run1', 'cam3')
        clear_tracking_outputs(self.cfg, 'run1', 'cam3')
        self.assert_only_removed(before, {
            'detections/tracking_uploads/run1/cam3/centroids.npz',
            'triangulation/tracking_uploads/run1/trajectory.npz',
            'triangulation/tracking_uploads/run1/flight_reference.json',
        })

    def test_calibration_clear_preserves_other_cameras_and_all_inputs(self):
        before = self.snapshot()
        clear_calibration_camera_outputs(self.cfg, 'cam3')
        clear_calibration_camera_outputs(self.cfg, 'cam3')
        self.assert_only_removed(before, {
            'detections/uploads/cam3/centroids.npz',
            'triangulation/uploads/trajectory.npz', 'uploads_calibration.json',
        })

    def test_invalid_camera_or_symlink_does_not_delete_anything(self):
        before = self.snapshot()
        for camera in ('', '../cam3', 'cam4', ['cam3']):
            with self.assertRaises(ValueError):
                clear_tracking_outputs(self.cfg, 'run1', camera)
            with self.assertRaises(ValueError):
                clear_calibration_camera_outputs(self.cfg, camera)
        self.assert_only_removed(before, set())
        for root in ('detections/tracking_uploads/run1', 'detections/uploads'):
            target = self.root/root/'cam3/centroids.npz'
            target.unlink()
            target.parent.rmdir()
            target.parent.symlink_to(self.root/root/'cam2', target_is_directory=True)
        with self.assertRaises(ValueError):
            clear_tracking_outputs(self.cfg, 'run1', 'cam3')
        with self.assertRaises(ValueError):
            clear_calibration_camera_outputs(self.cfg, 'cam3')
        self.assertTrue(self.cfg.uploads_calibration_json.exists())
        self.assertEqual((self.root/'detections/uploads/cam2/centroids.npz').read_bytes(), b'preserve me')

    def test_legacy_run_and_flat_default_scope(self):
        (self.root/'tracking_uploads').rename(self.root/'actual_uploads')
        (self.root/'detections/tracking_uploads').rename(self.root/'detections/actual_uploads')
        (self.root/'triangulation/tracking_uploads').rename(self.root/'triangulation/actual_uploads')
        clear_tracking_outputs(self.cfg, 'run1', 'cam3')
        self.assertTrue((self.root/'detections/actual_uploads/run1/cam1/centroids.npz').exists())
        self.assertFalse((self.root/'detections/actual_uploads/run1/cam3').exists())
        self.make_file('actual_uploads/cam3/video.mp4')
        self.make_file('detections/actual_uploads/cam3/centroids.npz')
        self.make_file('triangulation/actual_uploads/trajectory.npz')
        kept = self.make_file('triangulation/actual_uploads/run1/trajectory.npz')
        clear_tracking_outputs(self.cfg, 'default', 'cam3')
        self.assertTrue(kept.exists())
        self.assertFalse((self.root/'triangulation/actual_uploads/trajectory.npz').exists())

    def test_endpoints_busy_clear_without_sam_and_detect_only_selected_camera(self):
        for source in ('upload', 'tracking'):
            with self.subTest(source=source), patch('firetrack.webui.ThreadingHTTPServer') as server, \
                    patch('firetrack.webui.JobRunner') as runner, patch('firetrack.webui.run_detection_on_specs') as detect:
                serve_webui(work_root=self.root, host='127.0.0.1', port=0)
                handler = object.__new__(server.call_args.args[1])
                body = {'run': 'run1', 'camera': 'cam3'}
                runner.return_value.start.return_value = False
                before = self.snapshot()
                with patch.object(handler, '_json') as response:
                    handler._handle_camera_action(body, 'clear-outputs', source)
                    self.assertEqual(response.call_args.args[0], 409)
                self.assert_only_removed(before, set())
                runner.return_value.start.return_value = True
                with patch.object(handler, '_json'):
                    handler._handle_camera_action(body, 'clear-outputs', source)
                    runner.return_value.start.call_args.args[1]()
                detect.assert_not_called()
                with patch.object(handler, '_json') as response:
                    handler._handle_camera_action(body, 'detect', source)
                    self.assertEqual(response.call_args.args[0], 202)
                    runner.return_value.start.call_args.args[1]()
                specs = detect.call_args.args[0]
                self.assertEqual([s.label for s in specs], ['cam3'])
                self.assertEqual(specs[0].relative_dir, Path('uploads/cam3' if source == 'upload' else 'tracking_uploads/run1/cam3'))
                self.assertTrue((self.root/'detections/uploads/cam1/centroids.npz').exists())
                self.assertTrue((self.root/'detections/tracking_uploads/run1/cam1/centroids.npz').exists())

    def test_missing_annotation_rejects_detect_without_deleting_outputs(self):
        self.cfg.uploads_clicks_json.write_text('[]')
        before = self.snapshot()
        with patch('firetrack.webui.ThreadingHTTPServer') as server, patch('firetrack.webui.JobRunner') as runner:
            serve_webui(work_root=self.root, host='127.0.0.1', port=0)
            handler = object.__new__(server.call_args.args[1])
            with patch.object(handler, '_json'):
                handler._handle_camera_action({'camera': 'cam3'}, 'detect', 'upload')
            with self.assertRaisesRegex(ValueError, 'Annotate cam3'):
                runner.return_value.start.call_args.args[1]()
        self.assert_only_removed(before, set())


if __name__ == '__main__':
    unittest.main()

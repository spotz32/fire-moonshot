import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from firetrack.calibrate_logs import (
    DroneTrack, calibrate_from_log, load_camera_clips, _image_space_K, _find_centroids, load_drone_track,
)
from firetrack.calibration_priors import load_priors
from firetrack.calibration_solver import SplinePath, project
from firetrack.flight_reference import FlightTrack, enu_frame
from firetrack.triangulate_uploads import _with_upload_metadata, validate_calibration, rotation_matrix
from firetrack.webui import WebConfig, _best_calibration_log_path


FRAME = {'axes': 'ENU', 'origin_lat_deg': 40., 'origin_lon_deg': -86., 'altitude_reference': 'AMSL'}


class CalibrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'uploads'
        self.detections = self.root / 'detections'
        self.output = self.root / 'calibration.json'
        self.K = np.array([[905., 0, 540], [0, 905, 960], [0, 0, 1]])
        self.epoch = 1787950000.
        times = np.linspace(0, 20, 401)
        points = np.column_stack([4*np.sin(times*.5), 3*np.cos(times*.7), 150+2*np.sin(times*.9)])
        self.track = DroneTrack(self.epoch+times, points, FRAME, self.epoch+times)
        self.path = SplinePath(self.track.epoch_s, points, smoothing_m=0)
        self.R = Rotation.from_rotvec([.08, -.14, .04]).as_matrix()
        self.p = np.array([1., -2., 138.])
        self.delta = .237
        self.poses = {
            'cam1': (self.R, self.p),
            'cam2': (Rotation.from_rotvec([-.06, .11, -.03]).as_matrix(), np.array([4., 2., 139.])),
        }
        for name, (rotation, position) in self.poses.items():
            folder = self.run / name
            folder.mkdir(parents=True)
            (folder / 'video.mp4').touch()
            (folder / 'metadata.json').write_text(json.dumps({'startTime': (self.epoch+3)*1e6}))
            (folder / 'camera.json').write_text(json.dumps({'K': self.K.tolist(), 'resolution': [1080, 1920], 'dist': [0]*5}))
            target = self.detections / 'uploads' / name
            target.mkdir(parents=True)
            times = 3+np.arange(240)/20
            pixels = project((self.path.evaluate(times-self.delta)-position) @ rotation.T, self.K, np.zeros(5))[0]
            np.savez(target/'centroids.npz', centroids=pixels, fps=20., width=1080, height=1920, frame_indices=np.arange(240))

    def calibrate(self):
        with patch('firetrack.calibrate_logs.load_drone_track', return_value=self.track):
            return calibrate_from_log(log_path=Path('test.BIN'), run_root=self.run,
                                      detections_root=self.detections, out_json=self.output,
                                      offset_search_radius_s=.7, smoothing_m=0)

    def test_end_to_end_export_preserves_common_focal_and_corrects_tracking_times(self):
        self.calibrate()
        document = json.loads(self.output.read_text())
        self.assertEqual(enu_frame(document), FRAME)
        cameras = validate_calibration(document)
        for cam in cameras:
            np.testing.assert_allclose(cam['K'], self.K, atol=2e-5)
            expected_R, expected_p = self.poses[cam['video']]
            np.testing.assert_allclose(rotation_matrix(cam), expected_R, atol=1e-5)
            np.testing.assert_allclose(cam['t'], -expected_R @ expected_p, atol=1e-3)
            self.assertEqual(cam['source']['time_offset_s'], 0.0)
            self.assertAlmostEqual(cam['source']['calibration_clock_delta_s'], self.delta, places=5)
            self.assertFalse(cam['source']['calibration_lag_applies_to_tracking'])
            tracking = self.root / 'later' / cam['video']
            tracking.mkdir(parents=True)
            (tracking/'metadata.json').write_text(json.dumps({'startTime': (self.epoch+200)*1e6}))
            mapped = _with_upload_metadata(cam, tracking.parent)
            self.assertAlmostEqual(mapped['start_epoch_s'], self.epoch+200, places=5)
        self.assertAlmostEqual(document['calibration_clock']['camera_minus_log_s'], self.delta, places=5)

    def test_one_camera_failure_does_not_replace_existing_calibration(self):
        self.output.write_text('previous calibration')
        (self.detections/'uploads/cam2/centroids.npz').unlink()
        with self.assertRaisesRegex(RuntimeError, 'all cameras'):
            self.calibrate()
        self.assertEqual(self.output.read_text(), 'previous calibration')

    def test_explicit_intrinsics_are_not_silently_scaled_or_rotated(self):
        clips = load_camera_clips(self.run, self.detections)
        np.testing.assert_array_equal(_image_space_K(clips[0], 1080, 1920), self.K)
        with self.assertRaisesRegex(ValueError, 'no automatic scaling'):
            _image_space_K(clips[0], 1920, 1080)

    def test_missing_inputs_reports_each_camera_and_exact_file(self):
        (self.run/'cam1/camera.json').unlink()
        (self.run/'cam2/camera.json').unlink()
        with self.assertRaises(ValueError) as error:
            load_camera_clips(self.run, self.detections)
        self.assertIn('cam1/camera.json', str(error.exception))
        self.assertIn('cam2/camera.json', str(error.exception))
        self.assertNotIn('cam1/metadata.json', str(error.exception))

    def test_detections_never_fall_back_to_another_run(self):
        self.assertIsNone(_find_centroids(self.detections, 'other_run', 'cam1'))
        (self.detections/'cam1').mkdir()
        (self.detections/'cam1/centroids.npz').touch()
        self.assertIsNone(_find_centroids(self.detections, 'other_run', 'cam1'))
        self.assertIsNotNone(_find_centroids(self.detections/'uploads', 'uploads', 'cam1'))

    def test_log_loader_uses_shared_utc_frame(self):
        track = FlightTrack(np.array([100., 101., 102., 103.]), np.array([[40, -86, 150]]*4), .01)
        with patch('firetrack.calibrate_logs.read_flight_track', return_value=track):
            converted = load_drone_track(Path('test.BIN'))
        np.testing.assert_array_equal(converted.epoch_s, track.epochs)
        np.testing.assert_allclose(converted.points_world, [[0, 0, 150]]*4)
        self.assertEqual(converted.frame_name, FRAME)

    def test_log_selection_rejects_ambiguity_and_ignores_nonoverlap(self):
        cfg = WebConfig(self.root)
        cfg.calibration_logs_root.mkdir()
        for name in ('one.BIN', 'two.BIN'):
            (cfg.calibration_logs_root/name).touch()
        with patch('firetrack.webui.summarize_log_and_run', return_value={'cameras': [{'overlap_s': 10}]}):
            with self.assertRaisesRegex(ValueError, 'Multiple uploaded'):
                _best_calibration_log_path(cfg)
        with patch('firetrack.webui.summarize_log_and_run', side_effect=[{'cameras': [{'overlap_s': 0}]}, {'cameras': [{'overlap_s': 10}]}]):
            self.assertEqual(_best_calibration_log_path(cfg).name, 'two.BIN')

    def test_sensor_priors_require_explicit_conventions(self):
        folder = self.run/'cam1'
        self.assertFalse(load_priors(folder, FRAME, 100, 101).names)
        camera = json.loads((folder/'camera.json').read_text())
        camera['calibration_priors'] = {'use_gravity': True}
        (folder/'camera.json').write_text(json.dumps(camera))
        with self.assertRaisesRegex(ValueError, 'conventions_verified'):
            load_priors(folder, FRAME, 100, 101)
        camera['calibration_priors'] = {
            'conventions_verified': True, 'use_phone_gps': True, 'gps_altitude_reference': 'AMSL',
            'use_gravity': True, 'device_to_camera': np.eye(3).tolist(), 'accelerometer_convention': 'specific_force',
            'use_magnetometer': True, 'magnetic_field_world': [0, 1, 0],
        }
        (folder/'camera.json').write_text(json.dumps(camera))
        (folder/'gps.csv').write_text('timestamp_us,latitude,longitude,altitude,accuracy_m\n100000000,40,-86,150,1\n')
        (folder/'imu.csv').write_text('timestamp_us,accel_x,accel_y,accel_z,mag_x,mag_y,mag_z\n100000000,0,0,9.81,0,25,0\n')
        priors = load_priors(folder, FRAME, 100, 101)
        self.assertEqual(len(priors.names), 3)
        np.testing.assert_array_equal(priors.position, [0, 0, 150])
        np.testing.assert_array_equal(priors.directions[0][1], [0, 0, 1])
        camera['calibration_priors']['gps_altitude_reference'] = 'ellipsoid'
        (folder/'camera.json').write_text(json.dumps(camera))
        with self.assertRaisesRegex(ValueError, 'AMSL'):
            load_priors(folder, FRAME, 100, 101)


if __name__ == '__main__':
    unittest.main()

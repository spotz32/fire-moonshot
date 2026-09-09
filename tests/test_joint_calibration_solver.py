import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from firetrack.calibration_solver import Priors, SplinePath
from firetrack.joint_calibration_solver import (
    JointCamera,
    fit_joint_calibration,
    prepare_joint_observations,
)


class JointCalibrationSolverTests(unittest.TestCase):
    def setUp(self):
        log_times = np.linspace(0, 24, 721)
        points = np.column_stack([
            4*np.sin(.43*log_times) + .04*log_times**2,
            3*np.cos(.61*log_times) + .15*log_times,
            14 + 2*np.sin(.79*log_times) + .03*log_times,
        ])
        self.path = SplinePath(log_times, points, smoothing_m=0)
        self.delta = .237
        self.epochs = np.linspace(3, 21, 360)
        self.true = [
            (Rotation.from_rotvec([.05, -.12, .03]).as_matrix(), np.array([-2., -1., .5]), 920.),
            (Rotation.from_rotvec([-.04, .10, -.02]).as_matrix(), np.array([3., -1.5, .8]), 980.),
            (Rotation.from_rotvec([.02, .04, .08]).as_matrix(), np.array([.5, 3., .2]), 890.),
        ]
        self.cameras = []
        for index, (R, position, focal) in enumerate(self.true):
            K_true = np.array([[focal, 0., 540.], [0., focal, 960.], [0., 0., 1.]])
            K_start = K_true.copy()
            K_start[0, 0] *= 1.06
            K_start[1, 1] *= .95
            dist = np.array([.015, -.002, .0005, -.0007, .0001])
            world = self.path.evaluate(self.epochs-self.delta)
            camera_points = (world-position) @ R.T
            pixels = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K_true, dist)[0].reshape(-1, 2)
            pixels[(3+7*index)::29] = np.nan
            self.cameras.append(JointCamera(
                f'cam{index+1}', self.epochs, pixels, K_start, dist, Priors(),
            ))

    def test_recovers_shared_lag_poses_and_common_focal_lengths(self):
        result = fit_joint_calibration(
            self.path, self.cameras, delta_center=.22, search_radius=.4,
            search_step=.2, sample_count=300, focal_sigma=np.inf,
        )
        self.assertAlmostEqual(result['delta_s'], self.delta, places=3)
        self.assertEqual(result['diagnostics']['intrinsics_mode'], 'free_common_focal')
        for estimated, (R, position, focal) in zip(result['cameras'], self.true):
            np.testing.assert_allclose(estimated['R'], R, atol=2e-3)
            np.testing.assert_allclose(estimated['position'], position, atol=3e-2)
            self.assertEqual(estimated['fx'], estimated['fy'])
            self.assertAlmostEqual(estimated['fx'], focal, delta=2.)
            self.assertEqual(estimated['sd_fx_px'], estimated['sd_fy_px'])

    def test_fixed_mode_uses_mean_common_focal(self):
        result = fit_joint_calibration(
            self.path, self.cameras, delta_center=self.delta, search_radius=0,
            search_step=.2, sample_count=240, focal_sigma=0,
        )
        for estimated, supplied in zip(result['cameras'], self.cameras):
            expected = 0.5 * (supplied.K[0, 0] + supplied.K[1, 1])
            self.assertEqual(estimated['fx'], expected)
            self.assertEqual(estimated['fy'], expected)
            self.assertIsNone(estimated['sd_focal_px'])

    def test_camera_identity_and_distortion_validation(self):
        duplicate = [self.cameras[0], JointCamera(
            self.cameras[0].name, self.cameras[1].epochs, self.cameras[1].pixels,
            self.cameras[1].K, self.cameras[1].dist, Priors(),
        )]
        with self.assertRaisesRegex(ValueError, 'uniquely named'):
            prepare_joint_observations(self.path, duplicate, (-.1, .5), 200)
        tilted = self.cameras[0].dist.tolist()+[0.]*7+[.01, 0.]
        bad = JointCamera('tilted', self.epochs, self.cameras[0].pixels,
                          self.cameras[0].K, np.array(tilted), Priors())
        with self.assertRaisesRegex(ValueError, 'tilted-sensor'):
            prepare_joint_observations(self.path, [bad, self.cameras[1]], (-.1, .5), 200)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from firetrack.calibration_solver import (
    SplinePath, Priors, fit_pose_time, observation_residual, chart_residual, project,
)


class CalibrationSolverTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 20, 401)
        self.points = np.column_stack((4*np.sin(self.t*.5), 3*np.cos(self.t*.7), 12+2*np.sin(self.t*.9)))
        self.path = SplinePath(self.t, self.points, smoothing_m=0)
        self.K = np.array([[900., 0, 540], [0, 910, 960], [0, 0, 1]])
        self.dist = np.array([.02, -.003, .001, -.002, .0001])
        self.R = Rotation.from_rotvec([.08, -.14, .04]).as_matrix()
        self.p = np.array([1., -2., .3])
        self.delta = .237
        self.times = np.linspace(3, 17, 240)
        q = (self.path.evaluate(self.times-self.delta)-self.p) @ self.R.T
        self.pixels = project(q, self.K, self.dist)[0]

    def fit(self, pixels=None, **kwargs):
        return fit_pose_time(self.path, self.times, self.pixels if pixels is None else pixels,
                             self.K, self.dist, search_radius=.7, search_step=.2, **kwargs)

    def test_recovers_pose_and_fractional_frame_clock_offset(self):
        fit = self.fit()
        self.assertAlmostEqual(fit['delta_s'], self.delta, places=5)
        np.testing.assert_allclose(fit['position'], self.p, atol=1e-5)
        np.testing.assert_allclose(fit['R'], self.R, atol=1e-6)
        self.assertLess(fit['diagnostics']['validation_median_px'], 1e-4)

    def test_missing_and_outlier_detections(self):
        rng = np.random.default_rng(20)
        pixels = self.pixels + rng.normal(0, .25, self.pixels.shape)
        pixels[::11] = np.nan
        pixels[5::13] += [100, -80]
        fit = self.fit(pixels)
        self.assertLess(abs(fit['delta_s']-self.delta), .005)
        self.assertLess(np.linalg.norm(fit['position']-self.p), .03)
        self.assertLess(fit['diagnostics']['validation_median_px'], 1)
        self.assertGreater(fit['diagnostics']['validation_rmse_all_px'], 10)

    def test_analytic_observation_and_chart_jacobians(self):
        priors = Priors(position=self.p+.2, directions=[(np.array([0., 0., 1.]), self.R[:, 2], .03)])
        args = (self.R, self.p, self.delta, self.path, self.times[:8], self.pixels[:8], self.K, self.dist, priors, 3., 1.)
        x = np.array([.02, -.01, .03, .1, -.2, .05, .2])
        _, analytic = chart_residual(x, *args)
        numeric = np.empty_like(analytic)
        for j in range(7):
            step = np.eye(7)[j] * 1e-6
            numeric[:, j] = (chart_residual(x+step, *args)[0]-chart_residual(x-step, *args)[0]) / 2e-6
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)
        residual, jac, _ = observation_residual(self.path, self.times[:8], self.pixels[:8], self.K, self.dist, self.R, self.p, self.delta)
        numeric_delta = (observation_residual(self.path, self.times[:8], self.pixels[:8], self.K, self.dist, self.R, self.p, self.delta+1e-6)[0]
                         - observation_residual(self.path, self.times[:8], self.pixels[:8], self.K, self.dist, self.R, self.p, self.delta-1e-6)[0]) / 2e-6
        np.testing.assert_allclose(jac[:, :, 6], numeric_delta, rtol=1e-5, atol=1e-5)

    def test_straight_line_is_rejected(self):
        line = SplinePath(self.t, np.column_stack([self.t, self.t*0, self.t*0+20]), smoothing_m=0)
        with self.assertRaisesRegex(ValueError, 'straight line'):
            fit_pose_time(line, self.times, self.pixels, self.K, self.dist)

    def test_constant_speed_circle_is_rejected(self):
        points = np.column_stack([3*np.sin(self.t*.5), 3*np.cos(self.t*.5), self.t*0+12])
        circle = SplinePath(self.t, points, smoothing_m=0)
        pixels = project((circle.evaluate(self.times-self.delta)-self.p) @ self.R.T, self.K, self.dist)[0]
        with self.assertRaisesRegex(ValueError, 'ambiguous|rank deficient|converge'):
            fit_pose_time(circle, self.times, pixels, self.K, self.dist, search_radius=.7)

    def test_fixed_clock_reduces_to_pose_only(self):
        fit = fit_pose_time(self.path, self.times, self.pixels, self.K, self.dist,
                            delta_center=self.delta, search_radius=0)
        np.testing.assert_allclose(fit['position'], self.p, atol=1e-5)
        self.assertIsNone(fit['diagnostics']['clock_offset_std_s_local'])

    def test_verified_priors_can_constrain_circle_ambiguity(self):
        points = np.column_stack([3*np.sin(self.t*.5), 3*np.cos(self.t*.5), self.t*0+12])
        circle = SplinePath(self.t, points, smoothing_m=0)
        pixels = project((circle.evaluate(self.times-self.delta)-self.p) @ self.R.T, self.K, self.dist)[0]
        priors = Priors(position=self.p, position_sigma=np.full(3, .1),
                        directions=[(np.array([0., 0., 1.]), self.R[:, 2], .03),
                                    (np.array([0., 1., 0.]), self.R[:, 1], .1)],
                        names=['phone_GPS', 'gravity', 'magnetometer_direction'])
        fit = fit_pose_time(circle, self.times, pixels, self.K, self.dist,
                            search_radius=.7, priors=priors)
        self.assertAlmostEqual(fit['delta_s'], self.delta, places=4)
        self.assertTrue(any('depends on sensor priors' in w for w in fit['diagnostics']['warnings']))

    def test_insufficient_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_pose_time(self.path, self.times+100, self.pixels, self.K, self.dist)

    def test_bad_intrinsics_and_solver_settings_are_rejected(self):
        K = self.K.copy()
        K[0, 1] = 1.
        with self.assertRaisesRegex(ValueError, 'zero skew'):
            fit_pose_time(self.path, self.times, self.pixels, K, self.dist)
        for values in ({'search_radius': -1}, {'search_step': 0}, {'huber_px': 0}, {'sample_count': 5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_pose_time(self.path, self.times, self.pixels, self.K, self.dist, **values)

    def test_spline_gaps_and_extrapolation_are_rejected(self):
        indices = np.r_[0:100, 150:401]
        path = SplinePath(self.t[indices], self.points[indices])
        for query in ([-1], [6], [21]):
            with self.assertRaises(ValueError):
                path.evaluate(query)
        self.assertFalse(path.supports([6], (-.1, .1))[0])

    def test_smoothing_reduces_log_noise(self):
        noisy = self.points + np.random.default_rng(1).normal(0, .08, self.points.shape)
        smooth = SplinePath(self.t, noisy, smoothing_m=.08)
        self.assertLess(np.mean((smooth.evaluate(self.t)-self.points)**2), np.mean((noisy-self.points)**2))


if __name__ == '__main__':
    unittest.main()

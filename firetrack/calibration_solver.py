"""Fixed-intrinsic, continuous-time pose calibration for a moving target.

R maps world to camera, p is camera position, and log_time = camera_time - delta.
SciPy solves bounded local SO(3) charts with analytic Jacobians; accepted charts
are retracted onto the rotation group. Priors remain quadratic, not robustified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(w):
    angle = np.linalg.norm(w)
    W = skew(w)
    if angle < 1e-5:
        return np.eye(3) + (0.5 - angle**2 / 24) * W + (1/6 - angle**2 / 120) * (W @ W)
    return np.eye(3) + (1 - np.cos(angle)) / angle**2 * W + (angle - np.sin(angle)) / angle**3 * (W @ W)


class SplinePath:
    def __init__(self, times, points, smoothing_m=0.1, max_gap_s=1.0):
        self.times = np.asarray(times, dtype=float)
        points = np.asarray(points, dtype=float)
        if (len(self.times) < 4 or points.shape != (len(self.times), 3)
                or not np.isfinite(self.times).all() or not np.isfinite(points).all()
                or np.any(np.diff(self.times) <= 0) or not np.isfinite([smoothing_m, max_gap_s]).all()
                or smoothing_m < 0 or max_gap_s <= 0):
            raise ValueError("Spline needs finite positions, increasing times, and valid smoothing/gap settings.")
        self.origin = float(self.times[0])
        self.segments = []
        breaks = np.flatnonzero(np.diff(self.times) > max_gap_s) + 1
        for indices in np.split(np.arange(len(self.times)), breaks):
            if len(indices) < 4:
                continue
            t = self.times[indices] - self.origin
            splines = [UnivariateSpline(t, points[indices, k], s=len(t)*smoothing_m**2, ext=2) for k in range(3)]
            self.segments.append((t[0], t[-1], splines))
        if not self.segments:
            raise ValueError("No continuous flight-log segment has four position samples.")

    def evaluate(self, relative_times, derivative=0):
        t = np.asarray(relative_times, dtype=float)
        out = np.full((len(t), 3), np.nan)
        for lo, hi, splines in self.segments:
            mask = (t >= lo) & (t <= hi)
            if mask.any():
                out[mask] = np.column_stack([s(t[mask], nu=derivative) for s in splines])
        if not np.isfinite(out).all():
            raise ValueError("Time query crosses a flight-log gap or needs extrapolation.")
        return out

    def supports(self, relative_times, delta_bounds):
        t = np.asarray(relative_times)
        valid = np.zeros(len(t), dtype=bool)
        for lo, hi, _ in self.segments:
            valid |= (t - delta_bounds[1] >= lo) & (t - delta_bounds[0] <= hi)
        return valid


@dataclass
class Priors:
    position: np.ndarray | None = None
    position_sigma: np.ndarray = field(default_factory=lambda: np.full(3, 2.0))
    # (world unit direction, observed camera unit direction, sigma radians)
    directions: list = field(default_factory=list)
    names: list = field(default_factory=list)


def project(points_camera, K, dist):
    points_camera = np.asarray(points_camera, dtype=float)
    pixels, jac = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3), K, dist)
    return pixels.reshape(-1, 2), jac[:, 3:6].reshape(-1, 2, 3)


def observation_residual(path, times, pixels, K, dist, R, p, delta):
    world = path.evaluate(times - delta)
    velocity = path.evaluate(times - delta, 1)
    q = (world - p) @ R.T
    predicted, Jp = project(q, K, dist)
    J = np.empty((len(times), 2, 7))
    J[:, :, :3] = Jp @ np.array([skew(v) for v in q])
    J[:, :, 3:6] = Jp @ R
    J[:, :, 6] = np.einsum('nij,nj->ni', Jp, velocity @ R.T)
    return pixels - predicted, J, q


def huber_blocks(residual, jacobian, threshold):
    """Exact vector-Huber objective expressed as least-squares residual blocks."""
    norm = np.linalg.norm(residual, axis=1)
    high = norm > threshold
    scale = np.ones(len(norm))
    deriv = np.zeros(len(norm))
    n = norm[high]
    scale[high] = np.sqrt(2*threshold/n - (threshold/n)**2)
    deriv[high] = (-threshold/n**2 + threshold**2/n**3) / scale[high]
    transform = scale[:, None, None] * np.eye(2)
    transform += (deriv / np.maximum(norm, 1e-15))[:, None, None] * residual[:, :, None] * residual[:, None, :]
    return residual * scale[:, None], transform @ jacobian


def prior_residual(priors, R, p):
    residuals, jacobians = [], []
    if priors.position is not None:
        residuals.append((priors.position - p) / priors.position_sigma)
        J = np.zeros((3, 7))
        J[:, 3:6] = -np.diag(1 / priors.position_sigma)
        jacobians.append(J)
    for world, observed, sigma in priors.directions:
        predicted = R @ world
        residuals.append((observed - predicted) / sigma)
        J = np.zeros((3, 7))
        J[:, :3] = skew(predicted) / sigma
        jacobians.append(J)
    return (np.concatenate(residuals), np.vstack(jacobians)) if residuals else (np.empty(0), np.empty((0, 7)))


def chart_residual(x, R0, p0, fixed_delta, path, times, pixels, K, dist, priors, huber_px, pixel_sigma):
    R = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
    p = p0 + x[3:6]
    delta = x[6] if len(x) == 7 else fixed_delta
    residual, J, q = observation_residual(path, times, pixels, K, dist, R, p, delta)
    residual, J = huber_blocks(residual / pixel_sigma, J / pixel_sigma, huber_px / pixel_sigma)
    pr, pj = prior_residual(priors, R, p)
    # Reject behind-camera solutions while keeping the optimization differentiable
    # away from the depth hinge. PnP seeds are already checked for positive depth.
    depth = np.maximum(0, 0.1 - q[:, 2]) * 100
    dj = np.zeros((len(times), 7))
    dq = np.concatenate((-np.array([skew(v) for v in q]), np.broadcast_to(-R, (len(q), 3, 3)),
                         -(path.evaluate(times-delta, 1) @ R.T)[:, :, None]), axis=2)
    dj[depth > 0] = -100 * dq[depth > 0, 2, :]
    jac = np.vstack((J.reshape(-1, 7), pj, dj))
    jac[:, :3] = jac[:, :3] @ left_jacobian(x[:3])
    return np.concatenate((residual.ravel(), pr, depth)), jac[:, :len(x)]


def _refine(seed, path, times, pixels, K, dist, priors, bounds, huber_px, pixel_sigma, gate_px):
    R, p, delta = seed
    active = np.ones(len(times), dtype=bool)
    variable_time = bounds[1] > bounds[0]
    converged = False
    for iteration in range(20):
        if active.sum() < 12:
            raise ValueError("Too few consistent training detections.")
        x0 = np.r_[np.zeros(6), delta] if variable_time else np.zeros(6)
        lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
        lo[:3], hi[:3] = -0.35, 0.35
        if variable_time:
            lo[6], hi[6] = bounds
        args = (R, p, delta, path, times[active], pixels[active], K, dist, priors, huber_px, pixel_sigma)
        solved = least_squares(lambda x: chart_residual(x, *args)[0], x0,
                               jac=lambda x: chart_residual(x, *args)[1], bounds=(lo, hi),
                               x_scale='jac', max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-8)
        R = Rotation.from_rotvec(solved.x[:3]).as_matrix() @ R
        p = p + solved.x[3:6]
        previous_delta = delta
        delta = float(solved.x[6]) if variable_time else delta
        residual, _, q = observation_residual(path, times, pixels, K, dist, R, p, delta)
        selected = (np.linalg.norm(residual, axis=1) <= gate_px) & (q[:, 2] > 0.1)
        stable = np.array_equal(selected, active)
        active = selected
        if stable and solved.success and np.linalg.norm(solved.x[:6]) < 1e-5 and abs(delta-previous_delta) < 1e-6:
            converged = True
            break
    if not converged:
        raise ValueError("Pose/time optimization or detection inlier selection did not converge.")
    residual, J, _ = observation_residual(path, times, pixels, K, dist, R, p, delta)
    robust, _ = huber_blocks(residual / pixel_sigma, J / pixel_sigma, huber_px / pixel_sigma)
    pr, _ = prior_residual(priors, R, p)
    score = float(np.sum(robust**2) + np.sum(pr**2))
    return score, R, p, delta, active, iteration + 1


def fit_pose_time(path, camera_epochs, pixels, K, dist, *, delta_center=0., search_radius=2.,
                  search_step=0.25, sample_count=600, huber_px=3., gate_px=8., pixel_sigma=1., priors=None):
    """Fit on 80% of temporal blocks; report remaining blocks without refitting."""
    priors = priors or Priors()
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9
            or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()):
        raise ValueError("Expected fixed OpenCV intrinsics with zero skew and valid distortion coefficients.")
    settings = [delta_center, search_radius, search_step, huber_px, gate_px, pixel_sigma]
    if not np.isfinite(settings).all() or search_radius < 0 or min(search_step, huber_px, gate_px, pixel_sigma) <= 0 or sample_count < 30:
        raise ValueError("Invalid calibration solver settings (sample_count must be >=30).")
    bounds = (delta_center-search_radius, delta_center+search_radius)
    times = np.asarray(camera_epochs, float) - path.origin
    pixels = np.asarray(pixels, float)
    if pixels.shape != (len(times), 2) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Expected ordered frame timestamps and one 2D centroid per frame.")
    eligible = np.isfinite(pixels).all(axis=1) & path.supports(times, bounds)
    indices = np.flatnonzero(eligible)
    if len(indices) > sample_count:
        indices = indices[np.linspace(0, len(indices)-1, sample_count).round().astype(int)]
    if len(indices) < 30:
        raise ValueError("Fewer than 30 detections overlap continuous log data over the full offset search range.")
    times, pixels = times[indices], pixels[indices]
    # Hold out two contiguous time blocks, not only adjacent alternating frames.
    validation = (np.minimum(9, (10*(times-times[0])/(times[-1]-times[0])).astype(int)) % 5) == 4
    train = ~validation
    if validation.sum() < 6 or train.sum() < 24:
        raise ValueError("Insufficient temporal coverage for held-out validation.")
    world = path.evaluate(times[train]-delta_center)
    spread = np.linalg.svd(world-world.mean(axis=0), compute_uv=False)
    if spread[0] < 0.1 or spread[1] / spread[0] < 1e-3:
        raise ValueError("Calibration flight is stationary or effectively a straight line; pose/timing is poorly constrained.")
    count = int(np.ceil(2*search_radius/search_step)) + 1
    if count > 401:
        raise ValueError("Offset initialization would exceed 401 candidates; increase the search step.")
    seeds = []
    for delta in np.linspace(*bounds, count):
        world = path.evaluate(times[train]-delta)
        # Center world points before OpenCV initialization for better conditioning.
        center = world.mean(axis=0)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(world-center), np.ascontiguousarray(pixels[train]), K, dist,
            iterationsCount=300, reprojectionError=gate_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 12:
            continue
        R = Rotation.from_rotvec(rvec.ravel()).as_matrix()
        p = center - R.T @ tvec.ravel()
        residual, J, q = observation_residual(path, times[train], pixels[train], K, dist, R, p, delta)
        if (q[:, 2] > 0).mean() < 0.9:
            continue
        robust, _ = huber_blocks(residual/pixel_sigma, J/pixel_sigma, huber_px/pixel_sigma)
        pr, _ = prior_residual(priors, R, p)
        seeds.append((float(np.sum(robust**2)+np.sum(pr**2)), R, p, delta))
    if not seeds:
        raise ValueError("No positive-depth PnP initialization fits the detections at the searched offsets.")
    fits = []
    for _, R, p, delta in sorted(seeds, key=lambda s: s[0])[:4]:
        try:
            fits.append(_refine((R, p, delta), path, times[train], pixels[train], K, dist,
                                priors, bounds, huber_px, pixel_sigma, gate_px))
        except ValueError:
            continue
    if not fits:
        raise ValueError("Joint pose/time refinement did not converge to a supported fit.")
    fits.sort(key=lambda f: f[0])
    score, R, p, delta, active, iterations = fits[0]
    for other in fits[1:]:
        if other[0] <= score*1.01+1e-6 and (abs(other[3]-delta) > 0.05 or np.linalg.norm(other[2]-p) > 0.5):
            raise ValueError("Multiple similarly good pose/time solutions; calibration is ambiguous.")
    residual, J, q = observation_residual(path, times, pixels, K, dist, R, p, delta)
    errors = np.linalg.norm(residual, axis=1)
    inlier = (errors <= gate_px) & (q[:, 2] > 0.1)
    if inlier[train].mean() < 0.6 or inlier[validation].mean() < 0.6:
        raise ValueError(f"Fewer than 60% of training or held-out detections agree within {gate_px:g}px. "
                         f"Training: {inlier[train].sum()}/{train.sum()}, median {np.median(errors[train]):.2f}px; "
                         f"held-out: {inlier[validation].sum()}/{validation.sum()}, median {np.median(errors[validation]):.2f}px. "
                         "Check fixed intrinsics, target identity, log accuracy, and timestamps.")
    nparam = 7 if search_radius > 0 else 6
    data_J = J[train & inlier].reshape(-1, 7)[:, :nparam] / pixel_sigma
    _, prior_J = prior_residual(priors, R, p)
    full_J = np.vstack((data_J, prior_J[:, :nparam]))

    def spectrum(A):
        scales = np.linalg.norm(A, axis=0)
        return np.linalg.svd(A / np.maximum(scales, 1e-15), compute_uv=False)

    data_s, full_s = spectrum(data_J), spectrum(full_J)
    if full_s[-1] / full_s[0] < 1e-5:
        raise ValueError("Pose/time Jacobian is rank deficient or ill-conditioned; use a more varied calibration flight.")
    if search_radius > 0 and min(delta-bounds[0], bounds[1]-delta) < 1e-4:
        raise ValueError("Clock offset reached the search boundary; enlarge the offset search range.")
    variance = max(1., float(np.sum((residual[train & inlier]/pixel_sigma)**2) / max(1, len(data_J)-nparam)))
    _, singular_values, Vt = np.linalg.svd(full_J, full_matrices=False)
    covariance = (Vt.T / singular_values**2) @ Vt * variance
    warnings = ["Intrinsics were held fixed; autofocus mismatch and flight-log bias are not included in uncertainty.",
                "Clock-offset stability across recordings is assumed, not measured."]
    if not priors.names:
        warnings.append("No verified phone sensor priors were enabled.")
    if data_s[-1] / data_s[0] < 1e-4:
        warnings.append("Image observations weakly constrain pose/time; this fit depends on sensor priors.")
    offset_std = float(np.sqrt(max(0, covariance[-1, -1]))) if nparam == 7 else None
    if offset_std is not None and offset_std > 0.05:
        warnings.append("Estimated clock offset has more than 50 ms local uncertainty.")
    diagnostics = {
        "method": "spline_SO3_pose_clock_Huber", "association": "iterative single-target inlier gating (not multi-candidate association)",
        "sampled_frames": len(times), "eligible_frames": int(eligible.sum()), "training_frames": int(train.sum()),
        "validation_frames": int(validation.sum()), "training_inliers": int(inlier[train].sum()),
        "validation_inliers": int(inlier[validation].sum()), "iterations": iterations,
        "median_reproj_px": float(np.median(errors)),
        "training_median_px": float(np.median(errors[train])),
        "validation_median_px": float(np.median(errors[validation])),
        "validation_rmse_all_px": float(np.sqrt(np.mean(errors[validation]**2))),
        "validation_rmse_inliers_px": float(np.sqrt(np.mean(errors[validation & inlier]**2))),
        "data_jacobian_singular_values": data_s.tolist(), "map_jacobian_singular_values": full_s.tolist(),
        "clock_offset_std_s_local": offset_std, "priors": priors.names, "warnings": warnings,
    }
    return {"R": R, "position": p, "delta_s": delta, "diagnostics": diagnostics}

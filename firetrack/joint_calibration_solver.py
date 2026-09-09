"""Joint moving-target camera calibration with a shared camera/log clock lag.

At synchronized camera times, detections are triangulated from every available
camera. Camera poses and focal lengths are adjusted so those points agree both
with the ArduPilot position spline and with the source pixels. CasADi supplies
the geometry Jacobian; SciPy supplies bounded robust nonlinear least squares.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration_solver import Priors, SplinePath, left_jacobian, skew


TRACK_SIGMA_M = 0.20
PIXEL_SIGMA_PX = 3.0
PARAMS_PER_CAMERA = 7  # local rotation, position delta, log(f/f0), with fx = fy = f


@dataclass(frozen=True)
class JointCamera:
    name: str
    epochs: np.ndarray
    pixels: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    priors: Priors


@dataclass(frozen=True)
class JointObservations:
    epochs: np.ndarray
    pixels: np.ndarray  # samples x cameras x 2
    visible: np.ndarray  # samples x cameras
    training: np.ndarray
    validation: np.ndarray


def _validate_camera(camera: JointCamera) -> JointCamera:
    epochs = np.asarray(camera.epochs, dtype=float)
    pixels = np.asarray(camera.pixels, dtype=float)
    K = np.asarray(camera.K, dtype=float)
    dist = np.asarray(camera.dist, dtype=float).reshape(-1)
    if (not camera.name or pixels.shape != (len(epochs), 2) or len(epochs) < 2
            or not np.isfinite(epochs).all() or np.any(np.diff(epochs) <= 0)):
        raise ValueError(f"{camera.name or 'camera'}: expected ordered timestamps and one centroid per frame.")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0]) > 1e-9):
        raise ValueError(f"{camera.name}: invalid OpenCV camera matrix.")
    if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError(f"{camera.name}: expected 4, 5, 8, 12, or 14 OpenCV distortion coefficients.")
    if dist.size == 14 and np.any(np.abs(dist[12:14]) > 1e-12):
        raise ValueError(f"{camera.name}: tilted-sensor distortion is not supported by joint calibration.")
    return JointCamera(camera.name, epochs, pixels, K, dist, camera.priors)


def _interpolate_track(query: np.ndarray, epochs: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((len(query), 2), np.nan)
    valid = np.zeros(len(query), dtype=bool)
    right = np.searchsorted(epochs, query, side="left")
    frame_step = float(np.median(np.diff(epochs)))
    for row, r in enumerate(right):
        if r < len(epochs) and abs(epochs[r] - query[row]) <= 1e-6:
            if np.isfinite(pixels[r]).all():
                out[row], valid[row] = pixels[r], True
            continue
        if r == 0 or r >= len(epochs):
            continue
        left = r - 1
        gap = epochs[r] - epochs[left]
        if gap <= 0 or gap > 1.5 * frame_step or not np.isfinite(pixels[[left, r]]).all():
            continue
        weight = (query[row] - epochs[left]) / gap
        out[row] = (1 - weight) * pixels[left] + weight * pixels[r]
        valid[row] = True
    return out, valid


def prepare_joint_observations(
    path: SplinePath,
    cameras: list[JointCamera],
    delta_bounds: tuple[float, float],
    sample_count: int,
) -> tuple[list[JointCamera], JointObservations]:
    cameras = [_validate_camera(camera) for camera in cameras]
    names = [camera.name for camera in cameras]
    if len(cameras) < 2 or len(set(names)) != len(names):
        raise ValueError("Joint calibration needs at least two uniquely named cameras.")
    if sample_count < 30 or not np.isfinite(delta_bounds).all() or delta_bounds[0] > delta_bounds[1]:
        raise ValueError("Invalid joint calibration sample count or lag bounds.")

    # A real camera's frame grid is preferable to a synthetic uniform grid. Use
    # the camera with the most finite detections so the grid is well supported.
    reference = max(cameras, key=lambda camera: int(np.isfinite(camera.pixels).all(axis=1).sum()))
    epochs = reference.epochs.copy()
    relative = epochs - path.origin
    supported = np.zeros(len(epochs), dtype=bool)
    for lo, hi, _ in path.segments:
        supported |= ((relative - delta_bounds[1]) >= lo) & ((relative - delta_bounds[0]) <= hi)

    observations = np.zeros((len(epochs), len(cameras), 2), dtype=float)
    visible = np.zeros((len(epochs), len(cameras)), dtype=bool)
    for index, camera in enumerate(cameras):
        observations[:, index], visible[:, index] = _interpolate_track(epochs, camera.epochs, camera.pixels)
    keep = supported & (visible.sum(axis=1) >= 2)
    selected = np.flatnonzero(keep)
    if len(selected) > sample_count:
        selected = selected[np.linspace(0, len(selected) - 1, sample_count).round().astype(int)]
    if len(selected) < 30:
        raise ValueError("Fewer than 30 synchronized frames have at least two detections and log coverage.")
    epochs, observations, visible = epochs[selected], observations[selected], visible[selected]

    blocks = np.minimum(9, (10 * (epochs - epochs[0]) / max(epochs[-1] - epochs[0], 1e-9)).astype(int))
    validation = (blocks % 5) == 4
    training = ~validation
    if validation.sum() < 6 or training.sum() < 24:
        raise ValueError("Insufficient temporal coverage for held-out joint calibration validation.")
    for index, camera in enumerate(cameras):
        if visible[training, index].sum() < 12 or visible[validation, index].sum() < 3:
            raise ValueError(f"{camera.name}: insufficient training or held-out detections.")
    return cameras, JointObservations(epochs, observations, visible, training, validation)


def _initial_poses(
    path: SplinePath,
    observations: JointObservations,
    cameras: list[JointCamera],
    delta_bounds: tuple[float, float],
    search_step: float,
    ransac_px: float,
) -> list[tuple[float, list[dict]]]:
    width = delta_bounds[1] - delta_bounds[0]
    count = max(1, int(np.ceil(width / search_step)) + 1)
    if count > 401:
        raise ValueError("Shared-lag initialization exceeds 401 candidates; increase the search step.")
    relative = observations.epochs - path.origin
    candidates = []
    for delta in np.linspace(delta_bounds[0], delta_bounds[1], count):
        poses, score = [], 0.0
        for index, camera in enumerate(cameras):
            use = observations.training & observations.visible[:, index]
            world = path.evaluate(relative[use] - delta)
            center = world.mean(axis=0)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.ascontiguousarray(world - center),
                np.ascontiguousarray(observations.pixels[use, index]),
                camera.K,
                camera.dist,
                iterationsCount=300,
                reprojectionError=ransac_px,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 12:
                poses = []
                break
            R = Rotation.from_rotvec(rvec.ravel()).as_matrix()
            position = center - R.T @ tvec.ravel()
            q = (world - position) @ R.T
            if (q[:, 2] > 0.1).mean() < 0.9:
                poses = []
                break
            projected = cv2.projectPoints(q, np.zeros(3), np.zeros(3), camera.K, camera.dist)[0].reshape(-1, 2)
            error = np.linalg.norm(projected - observations.pixels[use, index], axis=1)
            score += float(np.median(error) ** 2)
            poses.append({"R": R, "position": position})
        if poses:
            candidates.append((score, [{**pose, "delta": float(delta)} for pose in poses]))
    if not candidates:
        raise ValueError("No shared lag produced a positive-depth PnP initialization for every camera.")
    return sorted(candidates, key=lambda item: item[0])


def _so3_exp(vector):
    squared = ca.dot(vector, vector)
    angle = ca.sqrt(squared + 1e-24)
    x, y, z = vector[0], vector[1], vector[2]
    S = ca.vertcat(ca.horzcat(0, -z, y), ca.horzcat(z, 0, -x), ca.horzcat(-y, x, 0))
    a = ca.if_else(squared < 1e-8, 1 - squared / 6, ca.sin(angle) / angle)
    b = ca.if_else(squared < 1e-8, 0.5 - squared / 24, (1 - ca.cos(angle)) / (squared + 1e-24))
    return ca.SX.eye(3) + a * S + b * (S @ S)


def _distortion_terms(dist: np.ndarray) -> list[float]:
    values = np.zeros(14, dtype=float)
    values[:len(dist)] = dist
    return values.tolist()


def _distort(x, y, coefficients):
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, _, _ = coefficients
    r2 = x*x + y*y
    r4, r6 = r2*r2, r2*r2*r2
    radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x) + s1*r2 + s2*r4
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y + s3*r2 + s4*r4
    return xd, yd


def _bearing(pixel, fx, fy, K, dist):
    skew_px, cx, cy = float(K[0, 1]), float(K[0, 2]), float(K[1, 2])
    yd = (pixel[1] - cy) / fy
    xd = (pixel[0] - cx - skew_px*yd) / fx
    x, y = xd, yd
    coefficients = _distortion_terms(dist)
    for _ in range(7):
        projected_x, projected_y = _distort(x, y, coefficients)
        x += xd - projected_x
        y += yd - projected_y
    bearing = ca.vertcat(x, y, 1)
    return bearing / ca.norm_2(bearing)


def _project(point_camera, fx, fy, K, dist):
    x, y = point_camera[0] / point_camera[2], point_camera[1] / point_camera[2]
    xd, yd = _distort(x, y, _distortion_terms(dist))
    return ca.vertcat(fx*xd + float(K[0, 1])*yd + float(K[0, 2]),
                      fy*yd + float(K[1, 2]))


def _initial_focal(camera: JointCamera) -> float:
    """Collapse small checkerboard/factory fx/fy differences to one pixel focal length."""
    return float(0.5 * (camera.K[0, 0] + camera.K[1, 1]))


class _GeometryModel:
    def __init__(self, cameras, base_poses, pixels, visible):
        self.camera_count = len(cameras)
        self.sample_count = len(pixels)
        self.parameter_count = PARAMS_PER_CAMERA*self.camera_count + 1
        x = ca.SX.sym("joint_parameters", self.parameter_count)
        z = ca.SX.sym("joint_pixels", 2*self.camera_count)
        w = ca.SX.sym("joint_visibility", self.camera_count)
        rotations, positions, focals = [], [], []
        for index, (camera, pose) in enumerate(zip(cameras, base_poses)):
            start = PARAMS_PER_CAMERA*index
            rotations.append(_so3_exp(x[start:start+3]) @ ca.DM(pose["R"]))
            positions.append(ca.DM(pose["position"]) + x[start+3:start+6])
            focals.append(_initial_focal(camera) * ca.exp(x[start+6]))
        identity = ca.SX.eye(3)
        A, rhs = ca.SX.zeros(3, 3), ca.SX.zeros(3)
        for index, camera in enumerate(cameras):
            bearing_camera = _bearing(z[2*index:2*index+2], focals[index], focals[index], camera.K, camera.dist)
            bearing_world = rotations[index].T @ bearing_camera
            projector = identity - bearing_world @ bearing_world.T
            A += w[index] * projector
            rhs += w[index] * projector @ positions[index]
        point = ca.solve(A, rhs)
        reprojections, depths = [], []
        for index, camera in enumerate(cameras):
            point_camera = rotations[index] @ (point - positions[index])
            reprojections.append(w[index] *
                                 (_project(point_camera, focals[index], focals[index], camera.K, camera.dist)
                                  - z[2*index:2*index+2]))
            depths.append(point_camera[2])
        sample = ca.Function("joint_geometry_sample", [x, z, w],
                             [point, ca.vertcat(*reprojections), ca.vertcat(*depths)])
        mapped = sample.map(self.sample_count)
        parameters = ca.MX.sym("mapped_parameters", self.parameter_count)
        safe_pixels = np.asarray(pixels, dtype=float).copy()
        for index, camera in enumerate(cameras):
            safe_pixels[~visible[:, index], index] = [camera.K[0, 2], camera.K[1, 2]]
        Z = ca.DM(safe_pixels.reshape(self.sample_count, -1).T)
        W = ca.DM(np.asarray(visible, dtype=float).T)
        points, reprojections, depths = mapped(
            ca.repmat(parameters, 1, self.sample_count), Z, W,
        )
        output = ca.vertcat(ca.vec(points), ca.vec(reprojections), ca.vec(depths))
        self._function = ca.Function("joint_geometry", [parameters],
                                     [output, ca.jacobian(output, parameters)])

    def evaluate(self, parameters):
        values, jacobian = self._function(parameters)
        values, jacobian = np.asarray(values).ravel(), np.asarray(jacobian)
        n, c = self.sample_count, self.camera_count
        point_end = 3*n
        reprojection_end = point_end + 2*n*c
        return (
            values[:point_end].reshape(n, 3),
            values[point_end:reprojection_end].reshape(n, c, 2),
            values[reprojection_end:].reshape(n, c),
            jacobian[:point_end].reshape(n, 3, -1),
            jacobian[point_end:reprojection_end].reshape(n, c, 2, -1),
            jacobian[reprojection_end:].reshape(n, c, -1),
        )


def _parameter_values(parameters, cameras, base_poses):
    values = []
    for index, (camera, pose) in enumerate(zip(cameras, base_poses)):
        start = PARAMS_PER_CAMERA*index
        R = Rotation.from_rotvec(parameters[start:start+3]).as_matrix() @ pose["R"]
        position = pose["position"] + parameters[start+3:start+6]
        focal = _initial_focal(camera)*np.exp(parameters[start+6])
        values.append({
            "name": camera.name,
            "R": R,
            "position": position,
            "fx": float(focal),
            "fy": float(focal),
            "K": camera.K.copy(),
            "dist": camera.dist.copy(),
        })
    return values


def _residual_and_jacobian(model, parameters, path, relative_times, cameras, base_poses,
                           visible, focal_sigma, track_sigma, pixel_sigma):
    points, reprojection, depth, J_points, J_reprojection, J_depth = model.evaluate(parameters)
    delta = float(parameters[-1])
    target = path.evaluate(relative_times - delta)
    velocity = path.evaluate(relative_times - delta, derivative=1)
    track = (points - target) / track_sigma
    J_track = J_points / track_sigma
    J_track[:, :, -1] += velocity / track_sigma

    residuals = [track.ravel(), (reprojection / pixel_sigma).ravel()]
    jacobians = [J_track.reshape(-1, model.parameter_count),
                 (J_reprojection / pixel_sigma).reshape(-1, model.parameter_count)]

    depth_hinge = np.maximum(0.0, 0.1-depth) * visible * 100.0
    J_hinge = np.zeros_like(J_depth)
    active = (depth < 0.1) & visible
    J_hinge[active] = -100.0 * J_depth[active]
    residuals.append(depth_hinge.ravel())
    jacobians.append(J_hinge.reshape(-1, model.parameter_count))

    if np.isfinite(focal_sigma) and focal_sigma > 0:
        prior_residual, prior_jacobian = [], []
        for index in range(len(cameras)):
            column = PARAMS_PER_CAMERA*index + 6
            value = np.exp(parameters[column])
            row = np.zeros(model.parameter_count)
            row[column] = value/focal_sigma
            prior_residual.append((value-1)/focal_sigma)
            prior_jacobian.append(row)
        residuals.append(np.asarray(prior_residual))
        jacobians.append(np.asarray(prior_jacobian))

    for index, (camera, values) in enumerate(zip(cameras, _parameter_values(parameters, cameras, base_poses))):
        start = PARAMS_PER_CAMERA*index
        priors = camera.priors
        if priors.position is not None:
            residuals.append((priors.position-values["position"])/priors.position_sigma)
            J = np.zeros((3, model.parameter_count))
            J[:, start+3:start+6] = -np.diag(1/priors.position_sigma)
            jacobians.append(J)
        for world, observed, sigma in priors.directions:
            predicted = values["R"] @ world
            residuals.append((observed-predicted)/sigma)
            J = np.zeros((3, model.parameter_count))
            J[:, start:start+3] = skew(predicted) @ left_jacobian(parameters[start:start+3]) / sigma
            jacobians.append(J)
    return np.concatenate(residuals), np.vstack(jacobians)


def _fit_seed(path, cameras, observations, seed_poses, delta_bounds, focal_sigma,
              track_sigma, pixel_sigma, rejection_gate_px, max_iterations):
    camera_count = len(cameras)
    parameter_count = PARAMS_PER_CAMERA*camera_count+1
    parameters = np.zeros(parameter_count)
    parameters[-1] = seed_poses[0]["delta"]
    free = np.ones(parameter_count, dtype=bool)
    if focal_sigma == 0:
        for index in range(camera_count):
            free[PARAMS_PER_CAMERA*index+6] = False
    if delta_bounds[0] == delta_bounds[1]:
        free[-1] = False
    lower, upper = np.full(parameter_count, -np.inf), np.full(parameter_count, np.inf)
    for index in range(camera_count):
        start = PARAMS_PER_CAMERA*index
        lower[start:start+3], upper[start:start+3] = -0.7, 0.7
        lower[start+6], upper[start+6] = -np.log(4), np.log(4)
    lower[-1], upper[-1] = delta_bounds
    active_visible = observations.visible[observations.training].copy()
    pixels = observations.pixels[observations.training]
    times = observations.epochs[observations.training] - path.origin
    base_poses = [{"R": pose["R"], "position": pose["position"]} for pose in seed_poses]
    total_nfev = 0
    solved = None
    for robust_pass in range(4):
        if np.any(active_visible.sum(axis=0) < 12) or np.any(active_visible.sum(axis=1) < 2):
            raise ValueError("Robust rejection left insufficient camera support.")
        model = _GeometryModel(cameras, base_poses, pixels, active_visible)

        def expand(free_values):
            full = parameters.copy()
            full[free] = free_values
            return full

        def residual(free_values):
            return _residual_and_jacobian(
                model, expand(free_values), path, times, cameras, base_poses,
                active_visible, focal_sigma, track_sigma, pixel_sigma,
            )[0]

        def jacobian(free_values):
            return _residual_and_jacobian(
                model, expand(free_values), path, times, cameras, base_poses,
                active_visible, focal_sigma, track_sigma, pixel_sigma,
            )[1][:, free]

        solved = least_squares(
            residual, parameters[free], jac=jacobian,
            bounds=(lower[free], upper[free]), x_scale="jac", loss="huber", f_scale=1.0,
            max_nfev=max_iterations, ftol=1e-9, xtol=1e-9, gtol=1e-8,
        )
        total_nfev += solved.nfev
        parameters[free] = solved.x
        if not solved.success or not np.isfinite(parameters).all():
            raise ValueError(f"Joint least-squares solve failed: {solved.message}")
        _, reprojection, _, _, _, _ = model.evaluate(parameters)
        errors = np.linalg.norm(reprojection, axis=2)
        valid_errors = errors[active_visible]
        sigma = 1.4826*np.median(np.abs(valid_errors-np.median(valid_errors))) if len(valid_errors) else 0.0
        gate = max(rejection_gate_px if robust_pass == 0 else 4*sigma, 8.0)
        masked = np.where(active_visible, errors, -1.0)
        worst = masked.argmax(axis=1)
        rows = np.flatnonzero(masked.max(axis=1) > gate)
        if not len(rows):
            break
        changed = False
        for row in rows:
            if active_visible[row].sum() > 2:
                active_visible[row, worst[row]] = False
                changed = True
        if not changed:
            break
    assert solved is not None
    final_model = _GeometryModel(cameras, base_poses, pixels, active_visible)
    final_residual, final_jacobian = _residual_and_jacobian(
        final_model, parameters, path, times, cameras, base_poses, active_visible,
        focal_sigma, track_sigma, pixel_sigma,
    )
    return {
        "parameters": parameters,
        "base_poses": base_poses,
        "active_visible": active_visible,
        "cost": float(final_residual @ final_residual),
        "jacobian": final_jacobian[:, free],
        "free": free,
        "nfev": total_nfev,
    }


def _metrics(path, cameras, observations, fit, track_sigma, pixel_sigma):
    model = _GeometryModel(cameras, fit["base_poses"], observations.pixels, observations.visible)
    points, reprojection, depth, _, _, _ = model.evaluate(fit["parameters"])
    delta = float(fit["parameters"][-1])
    target = path.evaluate(observations.epochs-path.origin-delta)
    track_error = np.linalg.norm(points-target, axis=1)
    ray_error = np.linalg.norm(reprojection, axis=2)
    positive = depth > 0.1

    def split(mask):
        observed = observations.visible & mask[:, None]
        values = ray_error[observed]
        return {
            "frames": int(mask.sum()),
            "track_rmse_m": float(np.sqrt(np.mean(track_error[mask]**2))),
            "track_median_m": float(np.median(track_error[mask])),
            "ray_rms_px": float(np.sqrt(np.mean(values**2))),
            "ray_median_px": float(np.median(values)),
            "positive_depth_fraction": float(positive[observed].mean()),
            "observations": int(observed.sum()),
        }
    return split(observations.training), split(observations.validation), ray_error, positive


def fit_joint_calibration(
    path: SplinePath,
    cameras: list[JointCamera],
    *,
    delta_center: float = 0.22,
    search_radius: float = 2.0,
    search_step: float = 0.25,
    sample_count: int = 600,
    focal_sigma: float = np.inf,
    ransac_px: float = 8.0,
    rejection_gate_px: float = 30.0,
    track_sigma_m: float = TRACK_SIGMA_M,
    pixel_sigma_px: float = PIXEL_SIGMA_PX,
    max_iterations: int = 150,
) -> dict:
    """Estimate all camera poses, one common fx=fy per camera, and one shared lag.

    ``delta`` follows ``log_time = camera_time - delta``. The delta is a
    calibration-only association parameter and is not a tracking timestamp
    correction.
    """
    numeric = [delta_center, search_radius, search_step, ransac_px,
               rejection_gate_px, track_sigma_m, pixel_sigma_px]
    if (not np.isfinite(numeric).all() or search_radius < 0 or
            min(search_step, ransac_px, rejection_gate_px, track_sigma_m, pixel_sigma_px) <= 0):
        raise ValueError("Invalid joint calibration solver settings.")
    if not (focal_sigma == 0 or np.isinf(focal_sigma) or 0 < focal_sigma < 1):
        raise ValueError("focal_sigma must be 0, inf, or a relative prior width in (0, 1).")
    delta_bounds = (delta_center-search_radius, delta_center+search_radius)
    cameras, observations = prepare_joint_observations(path, cameras, delta_bounds, sample_count)
    world = path.evaluate(observations.epochs[observations.training]-path.origin-delta_center)
    spread = np.linalg.svd(world-world.mean(axis=0), compute_uv=False)
    if spread[0] < 0.1 or spread[1]/spread[0] < 1e-3:
        raise ValueError("Calibration flight is stationary or effectively a straight line.")

    candidates = _initial_poses(path, observations, cameras, delta_bounds, search_step, ransac_px)
    selected = []
    for candidate in candidates:
        lag = candidate[1][0]["delta"]
        if not selected or all(abs(lag-other[1][0]["delta"]) >= min(0.1, search_step) for other in selected):
            selected.append(candidate)
        if len(selected) == 2:
            break
    if len(selected) < min(2, len(candidates)):
        for candidate in candidates:
            if all(candidate is not existing for existing in selected):
                selected.append(candidate)
            if len(selected) == min(2, len(candidates)):
                break

    fits, failures = [], []
    for _, poses in selected:
        try:
            fits.append(_fit_seed(
                path, cameras, observations, poses, delta_bounds, focal_sigma,
                track_sigma_m, pixel_sigma_px, rejection_gate_px, max_iterations,
            ))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    if not fits:
        raise ValueError("No joint calibration initialization converged. " + "; ".join(failures))
    fits.sort(key=lambda item: item["cost"])
    best = fits[0]
    values = _parameter_values(best["parameters"], cameras, best["base_poses"])
    delta = float(best["parameters"][-1])
    for other in fits[1:]:
        if other["cost"] > best["cost"]*1.01+1e-6:
            continue
        other_values = _parameter_values(other["parameters"], cameras, other["base_poses"])
        moved = max(np.linalg.norm(a["position"]-b["position"]) for a, b in zip(values, other_values))
        if abs(float(other["parameters"][-1])-delta) > 0.05 or moved > 0.5:
            raise ValueError("Multiple similarly good camera/lag solutions make calibration ambiguous.")

    training, validation, ray_error, positive = _metrics(
        path, cameras, observations, best, track_sigma_m, pixel_sigma_px,
    )
    for index, camera in enumerate(cameras):
        for label, mask in (("training", observations.training), ("validation", observations.validation)):
            observed = observations.visible[:, index] & mask
            accepted = observed & (ray_error[:, index] <= ransac_px) & positive[:, index]
            if accepted.sum() < max(3, int(np.ceil(0.6*observed.sum()))):
                raise ValueError(
                    f"{camera.name}: fewer than 60% of {label} observations have positive depth "
                    f"and <= {ransac_px:g}px ray residual ({accepted.sum()}/{observed.sum()})."
                )
    if min(training["positive_depth_fraction"], validation["positive_depth_fraction"]) < 0.95:
        raise ValueError("More than 5% of calibration observations triangulate behind a camera.")
    if search_radius > 0 and min(delta-delta_bounds[0], delta_bounds[1]-delta) < 1e-4:
        raise ValueError("Shared camera/log lag reached the search boundary.")
    for camera, value in zip(cameras, values):
        ratio = value["fx"]/_initial_focal(camera)
        if ratio <= 0.251 or ratio >= 3.99:
            raise ValueError(f"{camera.name}: focal length reached its physical search bound.")

    J = best["jacobian"]
    scales = np.linalg.norm(J, axis=0)
    singular = np.linalg.svd(J/np.maximum(scales, 1e-15), compute_uv=False)
    if singular[-1]/singular[0] < 1e-7:
        raise ValueError("Joint camera/lag Jacobian is rank deficient; use a more varied calibration flight.")
    covariance = np.linalg.pinv(J.T@J)
    variance = max(1.0, best["cost"]/max(1, len(J)-J.shape[1]))
    covariance *= variance
    full_std = np.full(len(best["parameters"]), np.nan)
    full_std[best["free"]] = np.sqrt(np.maximum(np.diag(covariance), 0))
    lag_std = float(full_std[-1])
    warnings = []
    if lag_std > 0.05:
        warnings.append("Shared camera/log lag has more than 50 ms local uncertainty.")
    if not any(camera.priors.names for camera in cameras):
        warnings.append("No verified phone sensor priors were enabled.")
    warnings.append("Flight-log positions are fused onboard estimates, not independent ground truth.")
    for index, (camera, value) in enumerate(zip(cameras, values)):
        start = PARAMS_PER_CAMERA*index
        value["sd_position_m"] = full_std[start+3:start+6].tolist()
        focal_std = float(value["fx"]*full_std[start+6]) if np.isfinite(full_std[start+6]) else None
        value["sd_focal_px"] = focal_std
        # Keep the component fields for calibration JSON compatibility.
        value["sd_fx_px"] = focal_std
        value["sd_fy_px"] = focal_std

    return {
        "cameras": values,
        "delta_s": delta,
        "diagnostics": {
            "method": "joint_triangulation_track_ray_shared_lag_common_focal_casadi_scipy",
            "intrinsics_mode": "fixed_common_focal" if focal_sigma == 0 else ("free_common_focal" if np.isinf(focal_sigma) else "soft_common_focal_prior"),
            "focal_sigma": "inf" if np.isinf(focal_sigma) else float(focal_sigma),
            "shared_clock_delta_s": delta,
            "shared_clock_delta_std_s_local": lag_std,
            "time_convention": "log_time = camera_time - shared_clock_delta_s",
            "tracking_time_correction_s": 0.0,
            "sampled_frames": int(len(observations.epochs)),
            "training": training,
            "validation": validation,
            "iterations": int(best["nfev"]),
            "jacobian_singular_values": singular.tolist(),
            "warnings": warnings,
        },
    }

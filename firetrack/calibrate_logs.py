"""Estimate static camera calibration from synchronized video and a flight log.

The no-mocap calibration stage jointly fits all camera poses, one common fx=fy
focal length per camera, and one calibration-only camera/log lag. Tracking videos remain on
their own synchronized camera timeline and do not require a flight log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .flight_reference import read_flight_track, to_calibration_frame

DEFAULT_DIST = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DroneTrack:
    epoch_s: np.ndarray
    points_world: np.ndarray
    frame_name: dict
    gps_epoch_s: np.ndarray
    clock_spread_s: float = 0.0


@dataclass(frozen=True)
class CameraClip:
    name: str
    folder: Path
    start_epoch_s: float
    K_source: np.ndarray
    dist: np.ndarray
    source_resolution: tuple[int, int]
    sensor_orientation: int = 0
    centroids: np.ndarray | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_indices: np.ndarray | None = None


def _manifest_row(folder: Path) -> dict | None:
    manifest = folder.parent / "manifest.json"
    if not manifest.exists():
        return None
    try:
        rows = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("label") == folder.name:
            return row
    return None


def load_drone_track(log_path: Path) -> DroneTrack:
    """Use the same fused POS, GPS-to-UTC conversion, and ENU frame as Results."""
    track = read_flight_track(log_path)
    origin = track.lat_lon_alt[0]
    frame = {"axes": "ENU", "origin_lat_deg": float(origin[0]),
             "origin_lon_deg": float(origin[1]), "altitude_reference": "AMSL"}
    return DroneTrack(track.epochs, to_calibration_frame(track.lat_lon_alt, frame),
                      frame, track.epochs, track.clock_spread_s)


def _camera_json_to_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    data = json.loads(path.read_text())
    if "K" in data:
        K = np.asarray(data["K"], dtype=float)
        size = data.get("resolution")
        if not isinstance(size, list) or len(size) != 2 or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 or int(v) != v for v in size):
            raise ValueError(f"{path}: explicit K requires decoded-video resolution [width, height].")
        if "dist" not in data:
            raise ValueError(f"{path}: provide fixed OpenCV distortion coefficients in dist (zeros if verified undistorted).")
        return K, np.asarray(data["dist"], float), tuple(int(v) for v in size), 0
    hardware = data.get("hardware") if isinstance(data.get("hardware"), dict) else data
    size = hardware.get("sensorPixelArraySize")
    if isinstance(size, dict):
        width = int(size.get("width", 0))
        height = int(size.get("height", 0))
    elif isinstance(size, list) and len(size) >= 2:
        width = int(size[0])
        height = int(size[1])
    else:
        width, height = 0, 0
    vals = hardware.get("factoryIntrinsicCalibration")
    if not (isinstance(vals, list) and len(vals) >= 4):
        raise ValueError(f"{path} does not contain factoryIntrinsicCalibration")
    fx, fy, cx, cy = (float(vals[i]) for i in range(4))
    skew = float(vals[4]) if len(vals) > 4 else 0.0
    K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    raw_dist = hardware.get("factoryLensDistortion") or []
    dist = np.array(_android_dist_to_opencv(raw_dist), dtype=np.float64)
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: missing sensor calibration resolution; refusing to guess image dimensions.")
    try:
        sensor_orientation = int(hardware.get("sensorOrientation") or 0) % 360
    except (TypeError, ValueError):
        sensor_orientation = 0
    return K, dist, (width, height), sensor_orientation


def _android_dist_to_opencv(values: list[float]) -> list[float]:
    vals = [float(v) for v in values]
    if len(vals) >= 5:
        return [vals[0], vals[1], vals[3], vals[4], vals[2]]
    if len(vals) == 4:
        return vals
    return list(DEFAULT_DIST)


def _metadata_start_epoch_s(path: Path) -> float:
    data = json.loads(path.read_text())
    if "startTime" not in data:
        raise ValueError(f"{path} does not contain startTime")
    return float(data["startTime"]) * 1e-6


def load_camera_clips(run_root: Path, detections_root: Path | None = None) -> list[CameraClip]:
    clips: list[CameraClip] = []
    folders = sorted(p for p in run_root.iterdir() if p.is_dir() and (p / "video.mp4").exists())
    missing = [f"{folder.name}/{name}" for folder in folders for name in ("camera.json", "metadata.json")
               if not (folder / name).is_file()]
    if missing:
        raise ValueError("Missing calibration inputs: " + ", ".join(missing) +
                         ". Each camera needs camera.json with fixed intrinsics and metadata.json with timestamps. "
                         "The loaded output calibration JSON is not automatically used as an intrinsic input.")
    for folder in folders:
        camera_json = folder / "camera.json"
        metadata_json = folder / "metadata.json"
        K, dist, resolution, sensor_orientation = _camera_json_to_intrinsics(camera_json)
        clip = CameraClip(
            name=folder.name,
            folder=folder,
            start_epoch_s=_metadata_start_epoch_s(metadata_json),
            K_source=K,
            dist=dist,
            source_resolution=resolution,
            sensor_orientation=sensor_orientation,
        )
        if detections_root is not None:
            clip = _with_detections(clip, detections_root, run_root.name)
        clips.append(clip)
    if not clips:
        raise RuntimeError(f"No camera folders with camera.json and metadata.json found under {run_root}")
    return clips


def _with_detections(clip: CameraClip, detections_root: Path, run_name: str) -> CameraClip:
    path = _find_centroids(detections_root, run_name, clip.name)
    if path is None:
        return clip
    with np.load(path) as z:
        centroids = np.asarray(z["centroids"], dtype=float)
        fps = float(z["fps"])
        indices = np.asarray(z["frame_indices"] if "frame_indices" in z else np.arange(len(centroids)), dtype=float)
        if (not np.isfinite(fps) or fps <= 0 or len(centroids) < 1 or centroids.shape != (len(centroids), 2)
                or indices.shape != (len(centroids),) or not np.isfinite(indices).all()
                or np.any(indices < 0) or np.any(indices != np.floor(indices)) or np.any(np.diff(indices) <= 0)
                or int(z["width"]) <= 0 or int(z["height"]) <= 0):
            raise ValueError(f"{path}: invalid centroid frame indices or FPS.")
        return CameraClip(
            name=clip.name,
            folder=clip.folder,
            start_epoch_s=clip.start_epoch_s,
            K_source=clip.K_source,
            dist=clip.dist,
            source_resolution=clip.source_resolution,
            sensor_orientation=clip.sensor_orientation,
            centroids=centroids,
            fps=fps,
            frame_indices=indices,
            width=int(z["width"]),
            height=int(z["height"]),
        )


def _find_centroids(detections_root: Path, run_name: str, camera_name: str) -> Path | None:
    candidates = [
        detections_root / run_name / camera_name / "centroids.npz",
        detections_root / run_name / f"{run_name}_{camera_name}" / "centroids.npz",
    ]
    if detections_root.name == run_name:
        candidates.append(detections_root / camera_name / "centroids.npz")
    for path in candidates:
        if path.exists():
            return path
    return None


def summarize_log_and_run(log_path: Path, run_root: Path) -> dict:
    track = load_drone_track(log_path)
    log_start = float(track.epoch_s[0])
    log_end = float(track.epoch_s[-1])
    cameras = []
    for metadata in sorted(run_root.glob("*/metadata.json")):
        if not (metadata.parent / "video.mp4").exists():
            continue
        start = _metadata_start_epoch_s(metadata)
        end = _metadata_end_epoch_s(metadata)
        if end <= start:
            cap = cv2.VideoCapture(str(metadata.parent / "video.mp4"))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if cap.isOpened() and fps > 0:
                    end = start + cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            finally:
                cap.release()
        overlap = max(0.0, min(log_end, end) - max(log_start, start))
        cameras.append({
            "camera": metadata.parent.name,
            "camera_start_utc": _iso(start),
            "camera_end_utc": _iso(end),
            "overlap_s": overlap,
        })
    return {
        "log": str(log_path),
        "run": str(run_root),
        "track_frame": track.frame_name,
        "log_start_utc": _iso(log_start),
        "log_end_utc": _iso(log_end),
        "position_samples": int(len(track.epoch_s)),
        "cameras": cameras,
    }


def _metadata_end_epoch_s(path: Path) -> float:
    data = json.loads(path.read_text())
    if "endTime" in data:
        return float(data["endTime"]) * 1e-6
    start = float(data["startTime"]) * 1e-6
    return start


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()


def calibrate_from_log(
    *,
    log_path: Path,
    run_root: Path,
    detections_root: Path,
    out_json: Path,
    sample_count: int = 600,
    ransac_reproj_px: float = 8.0,
    time_offset_s: float | None = None,
    clock_lag_s: float = 0.22,
    offset_search_radius_s: float = 2.0,
    offset_search_step_s: float = 0.25,
    smoothing_m: float = 0.1,
    huber_px: float = 3.0,
    focal_sigma: float = float("inf"),
) -> dict:
    from .calibration_solver import SplinePath
    from .calibration_priors import load_priors
    from .joint_calibration_solver import JointCamera, fit_joint_calibration

    track = load_drone_track(log_path)
    path = SplinePath(track.epoch_s, track.points_world, smoothing_m=smoothing_m)
    clips = load_camera_clips(run_root, detections_root)
    if len(clips) < 2:
        raise RuntimeError("Joint calibration needs at least two cameras.")
    missing_detections = [clip.name for clip in clips
                          if clip.centroids is None or clip.fps is None or clip.width is None or clip.height is None]
    if missing_detections:
        raise RuntimeError("Calibration not saved: all cameras must have detections; missing " +
                           ", ".join(missing_detections) + ". Run Detect first.")
    joint_cameras = []
    for clip in clips:
        assert clip.centroids is not None and clip.fps is not None and clip.width is not None and clip.height is not None
        K_image = _image_space_K(clip, clip.width, clip.height)
        frame_indices = clip.frame_indices if clip.frame_indices is not None else np.arange(len(clip.centroids))
        epochs = clip.start_epoch_s + frame_indices / clip.fps
        priors = load_priors(clip.folder, track.frame_name, float(epochs[0]), float(epochs[-1]))
        joint_cameras.append(JointCamera(
            name=clip.name, epochs=epochs, pixels=clip.centroids,
            K=K_image, dist=clip.dist, priors=priors,
        ))

    # Backward-compatible API: the former option represented a value added to
    # camera time, which is the negative of camera_time - log_time.
    lag_center = -float(time_offset_s) if time_offset_s is not None else float(clock_lag_s)
    mode = "fixed common focal" if focal_sigma == 0 else ("free common focal" if np.isinf(focal_sigma) else f"common focal prior {focal_sigma:g}")
    print(f"Joint calibration: {len(clips)} cameras, {mode}, shared lag initialized at {lag_center:+.3f}s", flush=True)
    try:
        solved = fit_joint_calibration(
            path, joint_cameras, delta_center=lag_center,
            search_radius=offset_search_radius_s, search_step=offset_search_step_s,
            sample_count=sample_count, focal_sigma=focal_sigma,
            ransac_px=ransac_reproj_px,
            pixel_sigma_px=huber_px,
        )
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"Calibration not saved: {exc}") from exc

    delta = solved["delta_s"]
    diagnostics = solved["diagnostics"]
    cameras, results = [], []
    by_name = {camera.name: camera for camera in joint_cameras}
    clips_by_name = {clip.name: clip for clip in clips}
    for value in solved["cameras"]:
        name = value["name"]
        clip, camera = clips_by_name[name], by_name[name]
        K_image = camera.K.copy()
        K_image[0, 0], K_image[1, 1] = value["fx"], value["fy"]
        warnings_out = list(diagnostics["warnings"])
        if "K" not in json.loads((clip.folder / "camera.json").read_text()):
            warnings_out.append("Legacy factory intrinsics initialized focal length and fixed distortion; verify the capture mode assumptions.")
        source = {
            "method": diagnostics["method"], "log": str(log_path), "run": str(run_root),
            "world_frame": track.frame_name,
            "time_offset_s": 0.0,
            "calibration_clock_delta_s": delta,
            "time_convention": diagnostics["time_convention"],
            "calibration_lag_applies_to_tracking": False,
            "window_unix": [float(camera.epochs[0]-delta), float(camera.epochs[-1]-delta)],
            "log_clock_spread_s": track.clock_spread_s,
            "spline_smoothing_m": smoothing_m,
            "intrinsics_mode": diagnostics["intrinsics_mode"],
            "sd_position_m": value["sd_position_m"],
            "sd_focal_px": value["sd_focal_px"],
            "sd_fx_px": value["sd_fx_px"], "sd_fy_px": value["sd_fy_px"],
            "warnings": warnings_out,
        }
        R, position = value["R"], value["position"]
        cameras.append({
            "video": name, "K": K_image.tolist(), "dist": camera.dist.tolist(),
            "resolution": [int(clip.width), int(clip.height)],
            "R": R.tolist(), "t": (-R @ position).tolist(), "position": position.tolist(),
            "pose_convention": "w2c", "start_epoch_s": float(clip.start_epoch_s), "source": source,
        })
        results.append({
            "camera": name, "status": "calibrated", "fx": value["fx"], "fy": value["fy"],
            "shared_clock_delta_s": delta,
        })
    report = {
        "out_json": str(out_json), "track_frame": track.frame_name,
        "shared_clock_delta_s": delta, "results": results,
        "solver": diagnostics,
    }
    document = {
        "world_frame": track.frame_name,
        "calibration_clock": {
            "camera_minus_log_s": delta,
            "std_s_local": diagnostics["shared_clock_delta_std_s_local"],
            "applies_to_tracking": False,
            "convention": diagnostics["time_convention"],
        },
        "cameras": cameras, "diagnostics": report,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temp = out_json.with_name(out_json.name + ".tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(out_json)
    print(f"Joint calibration complete: shared camera/log lag {delta:+.6f}s; "
          f"held-out track RMSE {diagnostics['validation']['track_rmse_m']:.3f}m; "
          f"ray median {diagnostics['validation']['ray_median_px']:.2f}px", flush=True)
    for warning in diagnostics["warnings"]:
        print(f"Warning: {warning}", flush=True)
    return report


def _scaled_K(K: np.ndarray, resolution: tuple[int, int], width: int, height: int) -> np.ndarray:
    src_w, src_h = resolution
    out = K.copy()
    out[0, 0] *= width / float(src_w)
    out[0, 2] *= width / float(src_w)
    out[1, 1] *= height / float(src_h)
    out[1, 2] *= height / float(src_h)
    return out


def _aspect_fill_K(K: np.ndarray, resolution: tuple[int, int], width: int, height: int) -> np.ndarray:
    """Map sensor intrinsics into a video stream using center-crop aspect fill.

    Android factory calibration is reported in active sensor-array coordinates,
    but the video stream often has a different aspect ratio. For a normal camera
    preview/video stream, the sensor image is scaled uniformly until it fills the
    requested stream, with the excess cropped symmetrically. That preserves square
    pixels and avoids the nonphysical focal lengths produced by independent x/y
    scaling.
    """
    src_w, src_h = resolution
    scale = max(width / float(src_w), height / float(src_h))
    crop_w = width / scale
    crop_h = height / scale
    crop_x = (float(src_w) - crop_w) / 2.0
    crop_y = (float(src_h) - crop_h) / 2.0
    out = K.copy()
    out[0, 0] *= scale
    out[0, 1] *= scale
    out[0, 2] = (out[0, 2] - crop_x) * scale
    out[1, 1] *= scale
    out[1, 2] = (out[1, 2] - crop_y) * scale
    return out


def _rotate_K(K: np.ndarray, width: int, height: int, steps_ccw: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a standard camera matrix after image-plane 90-degree rotations.

    ``ffmpeg transpose=2`` rotates pixels 90 degrees counter-clockwise:
    ``u' = v`` and ``v' = width - 1 - u``. OpenCV's ``cameraMatrix`` cannot be
    an arbitrary homography, so we express the rotated image with positive focal
    lengths and swapped principal point coordinates. The resulting PnP extrinsic
    is therefore in the formatted-video camera coordinate system, which is also
    what no-mocap tracking detections use.
    """
    steps = steps_ccw % 4
    out = K.copy()
    w, h = int(width), int(height)
    for _ in range(steps):
        fx = float(out[0, 0])
        fy = float(out[1, 1])
        cx = float(out[0, 2])
        cy = float(out[1, 2])
        skew = float(out[0, 1])
        if abs(skew) > 1e-9:
            # Android factory calibration normally has zero skew. Preserve the
            # dominant pinhole terms if a device reports a tiny non-zero value.
            skew = 0.0
        out = np.array(
            [[fy, skew, cy], [0.0, fx, float(w - 1) - cx], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        w, h = h, w
    return out, (w, h)


def _sensor_to_decoded_K(clip: CameraClip, width: int, height: int) -> np.ndarray:
    src_w, src_h = clip.source_resolution
    orientation = int(clip.sensor_orientation) % 360
    portrait_from_landscape_sensor = src_w > src_h and height > width
    landscape_from_portrait_sensor = src_h > src_w and width > height
    needs_quarter_turn = orientation in (90, 270) and (
        portrait_from_landscape_sensor or landscape_from_portrait_sensor
    )
    if not needs_quarter_turn:
        return _aspect_fill_K(clip.K_source, clip.source_resolution, width, height)

    pre_w, pre_h = height, width
    pre_K = _aspect_fill_K(clip.K_source, clip.source_resolution, pre_w, pre_h)
    steps_ccw = 3 if orientation == 90 else 1
    rotated_K, (out_w, out_h) = _rotate_K(pre_K, pre_w, pre_h, steps_ccw)
    if out_w != width or out_h != height:
        rotated_K = _scaled_K(rotated_K, (out_w, out_h), width, height)
    return rotated_K


def _image_space_K(clip: CameraClip, width: int, height: int) -> np.ndarray:
    """Return intrinsics in the decoded video coordinate system."""
    if "K" in json.loads((clip.folder / "camera.json").read_text()):
        if clip.source_resolution != (width, height):
            raise ValueError(f"{clip.name}: supplied intrinsic resolution differs from detections; no automatic scaling or rotation is allowed for explicit K.")
        return clip.K_source.copy()
    row = _manifest_row(clip.folder)
    if not row:
        return _sensor_to_decoded_K(clip, width, height)
    raw_w = int(row.get("source_width") or width)
    raw_h = int(row.get("source_height") or height)
    steps = int(row.get("rotation_steps_ccw") or 0)
    raw_K = _sensor_to_decoded_K(clip, raw_w, raw_h)
    rotated_K, (out_w, out_h) = _rotate_K(raw_K, raw_w, raw_h, steps)
    if out_w != width or out_h != height:
        rotated_K = _scaled_K(rotated_K, (out_w, out_h), width, height)
    return rotated_K

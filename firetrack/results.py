"""Read-only access to pipeline outputs for the Results view.

Surfaces what the dashboard needs to *show* the work: per-camera 2D detections
(centroids over video frames) and 3D reconstructions (triangulated trajectories).
All paths are validated against their root to prevent traversal from query params.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .triangulate_527 import score_trajectory
from .sources import LEGACY_TRACKING_UPLOAD_SUBDIR, TRACKING_UPLOAD_SUBDIR, tracking_result_path
from .flight_reference import load_reference

MAX_TRAJ_POINTS = 3000


def _safe_subdir(root: Path, rel: str) -> Path:
    root = root.resolve()
    public = Path(tracking_result_path(rel))
    target = (root / public).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes root")
    if not target.exists() and public.parts and public.parts[0] == TRACKING_UPLOAD_SUBDIR:
        legacy = (root / LEGACY_TRACKING_UPLOAD_SUBDIR / Path(*public.parts[1:])).resolve()
        if root not in legacy.parents:
            raise ValueError("path escapes root")
        if legacy.exists():
            return legacy
    return target


def _source_of(rel: str) -> str:
    normalized = tracking_result_path(str(rel))
    if normalized == "mocap_raw" or normalized.startswith("mocap_raw/"):
        return "mocap raw"
    if normalized == "tracking_uploads" or normalized.startswith("tracking_uploads/"):
        return "tracking flight"
    if normalized == "uploads" or normalized.startswith("uploads/"):
        return "calibration flight"
    return "dataset"


def _run_label(rel: Path) -> str:
    rel = Path(tracking_result_path(str(rel)))
    parts = rel.parts
    if parts and parts[0] == "mocap_raw":
        return f"Mocap Raw / {parts[1]}" if len(parts) > 1 else "Mocap Raw"
    name = rel.name or str(rel)
    if name == "tracking_uploads":
        return "default"
    if name == "uploads":
        return "Calibration Flight"
    return name


def _detection_label(rel: str, fallback: str) -> str:
    normalized = tracking_result_path(str(rel))
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 3 and parts[0] == "mocap_raw":
        return f"Mocap Raw / {parts[1]} / {fallback}"
    if len(parts) >= 3 and parts[0] == "tracking_uploads":
        return f"{parts[1]} / {fallback}"
    return fallback


def _run_name(rel: str) -> str:
    normalized = tracking_result_path(str(rel))
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return ""
    if parts[0] == "mocap_raw":
        return f"Mocap Raw / {parts[1]}" if len(parts) > 1 else "Mocap Raw"
    if parts[0] == "tracking_uploads":
        return parts[1] if len(parts) > 1 else "default"
    if parts[0] == "uploads":
        return "Calibration Flight"
    return parts[0]


def list_detections(detections_root: Path) -> list[dict]:
    root = Path(detections_root)
    if not root.exists():
        return []
    rows: list[dict] = []
    for summ in sorted(root.glob("**/summary.json")):
        try:
            s = json.loads(summ.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rel = s.get("relative_dir")
        if not rel or not (summ.parent / "centroids.npz").exists():
            continue
        rel = tracking_result_path(str(rel))
        rows.append({
            "dir": str(rel),
            "label": _detection_label(str(rel), s.get("label", str(rel))),
            "run": _run_name(str(rel)),
            "source": _source_of(str(rel)),
            "n_frames": s.get("n_frames"),
            "n_detected": s.get("n_detected"),
            "detection_rate": s.get("detection_rate"),
        })
    return rows


def list_trajectories(triangulation_root: Path) -> list[dict]:
    root = Path(triangulation_root)
    if not root.exists():
        return []
    rows: list[dict] = []
    for traj in sorted(root.glob("**/trajectory.npz")):
        rel = Path(tracking_result_path(str(traj.parent.relative_to(root))))
        rows.append({
            "dir": str(rel),
            "run": _run_label(rel),
            "source": _source_of(str(rel)),
        })
    return rows


def _clean2d(arr: np.ndarray) -> list:
    a = np.asarray(arr, dtype=np.float64)
    return [[None if not math.isfinite(v) else float(v) for v in row] for row in a]


def _clean1d(arr: np.ndarray) -> list:
    return [None if not math.isfinite(float(v)) else float(v) for v in np.asarray(arr, dtype=np.float64)]


@lru_cache(maxsize=64)
def video_path_for(detections_root: str, rel: str) -> str:
    path = _safe_subdir(Path(detections_root), rel) / "centroids.npz"
    with np.load(path) as z:
        return str(z["video_path"])


def load_centroids(detections_root: Path, rel: str) -> dict:
    path = _safe_subdir(Path(detections_root), rel) / "centroids.npz"
    with np.load(path) as z:
        centroids = np.array(z["centroids"], dtype=np.float64)
        start_epoch_s = None
        try:
            metadata = json.loads(Path(str(z["video_path"])).with_name("metadata.json").read_text())
            start_epoch_s = float(metadata["startTime"]) * 1e-6
        except (OSError, ValueError, KeyError):
            pass
        return {
            "start_epoch_s": start_epoch_s,
            "centroids": _clean2d(centroids),
            "width": int(z["width"]),
            "height": int(z["height"]),
            "fps": float(z["fps"]),
            "n_frames": int(len(centroids)),
            "n_detected": int(np.isfinite(centroids[:, 0]).sum()),
        }


def apply_centroid_edits(detections_root: Path, rel: str, edits: list[dict]) -> dict:
    """Apply manual per-frame corrections to a detection's centroids and save.

    Each edit is ``{"frame": i, "x": ..., "y": ...}`` to set a point, or
    ``{"frame": i, "clear": true}`` to mark the frame as no-detection. All other
    npz fields and the summary's metadata are preserved.
    """
    path = _safe_subdir(Path(detections_root), rel) / "centroids.npz"
    data = dict(np.load(path, allow_pickle=True))  # preserve every stored field
    centroids = np.array(data["centroids"], dtype=np.float64)
    n = len(centroids)
    for edit in edits:
        frame = int(edit["frame"])
        if frame < 0 or frame >= n:
            raise ValueError(f"frame {frame} out of range [0, {n})")
        if edit.get("clear"):
            centroids[frame] = [np.nan, np.nan]
            continue
        x, y = float(edit["x"]), float(edit["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("x and y must be finite numbers")
        centroids[frame] = [x, y]

    data["centroids"] = centroids
    np.savez(path, **data)
    n_detected = int(np.isfinite(centroids[:, 0]).sum())

    summary = path.parent / "summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            s["n_detected"] = n_detected
            s["detection_rate"] = n_detected / n if n else 0.0
            summary.write_text(json.dumps(s, indent=2, sort_keys=True))
        except (json.JSONDecodeError, OSError):
            pass
    return {"n_frames": n, "n_detected": n_detected}


def trajectory_download_name(rel: str, filename: str) -> str:
    parts = tracking_result_path(rel).split("/")
    if parts[0] == TRACKING_UPLOAD_SUBDIR:
        parts[0] = "tracking"
    return "_".join(parts).strip("_") + "_" + filename


def trajectory_file(triangulation_root: Path, rel: str, fmt: str) -> Path:
    """Resolve a downloadable trajectory artifact (csv or npz), traversal-safe."""
    name = {"csv": "trajectory.csv", "npz": "trajectory.npz", "comparison": "flight_comparison.csv"}.get(fmt)
    if name is None:
        raise ValueError("fmt must be 'csv' or 'npz'")
    path = _safe_subdir(Path(triangulation_root), rel) / name
    if not path.exists():
        raise FileNotFoundError(name)
    if fmt == "comparison":
        trajectory = path.with_name("trajectory.npz")
        with np.load(trajectory, allow_pickle=False) as data:
            reference = load_reference(trajectory, data["epoch_times_s"])
        if not reference or reference.get("error"):
            raise FileNotFoundError("no current flight-log comparison")
    return path


def load_trajectory(triangulation_root: Path, rel: str) -> dict:
    path = _safe_subdir(Path(triangulation_root), rel) / "trajectory.npz"
    with np.load(path, allow_pickle=True) as z:
        raw = np.array(z["trajectory_raw"], dtype=np.float64)
        smooth = np.array(z["trajectory_smooth"], dtype=np.float64)
        gt = np.array(z["gt_drone"], dtype=np.float64)
        n_views = np.array(z["n_views"], dtype=np.int64)
        reproj = np.array(z["reproj_errors_px"], dtype=np.float64)
        epochs = np.array(z["epoch_times_s"], dtype=np.float64) if "epoch_times_s" in z else None
    summary_path = path.with_name("summary.json")
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        summary = {}
    n = len(raw)
    stride = max(1, math.ceil(n / MAX_TRAJ_POINTS))
    sl = slice(None, None, stride)
    has_gt = bool(np.isfinite(gt[:, 0]).any())
    reference = load_reference(path, epochs) if _source_of(rel) == "tracking flight" else None
    if reference and "points" in reference:
        reference["points"] = _clean2d(reference["points"][sl])
    return {
        "reference": reference,
        "raw": _clean2d(raw[sl]),
        "smooth": _clean2d(smooth[sl]),
        "gt": _clean2d(gt[sl]) if has_gt else None,
        "n_views": [int(v) for v in n_views[sl]],
        "reproj": _clean1d(reproj[sl]),
        "n_frames": n,
        "stride": stride,
        "epoch_times_s": _clean1d(epochs[sl]) if epochs is not None else None,
        "camera_start_epochs_s": summary.get("camera_start_epochs_s", {}),
        "camera_time_offsets_s": summary.get("camera_time_offsets_s", {}),
        "metrics": {
            "n_triangulated": int(np.isfinite(raw[:, 0]).sum()),
            "median_reproj_px": float(np.nanmedian(reproj)) if np.isfinite(reproj).any() else None,
            "has_gt": has_gt,
            "rmse_smooth_m": score_trajectory(smooth, gt)["rmse_m"] if has_gt else None,
        },
    }

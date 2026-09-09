"""Export upload-mode camera calibration from a mocap-annotated run."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .mocap import load_mocap_6d
from .triangulate_527 import (
    DRONE_BODY,
    build_correspondences,
    camera_center_world,
    find_calibration_time_offset,
    make_session,
    mocap_path,
)
from .triangulate_uploads import validate_calibration


def _manifest_by_label(formatted_root: Path) -> dict[str, dict]:
    manifest = formatted_root / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        rows = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {str(row.get("label")): row for row in rows if isinstance(row, dict)}


def _default_upload_label(session_label: str, manifest_row: dict | None) -> str:
    source_uuid = str((manifest_row or {}).get("source_uuid") or "")
    if source_uuid.startswith("camera") and source_uuid[6:].isdigit():
        return f"cam{source_uuid[6:]}"
    return source_uuid or session_label


def _parse_label_map(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Label map must be source=target, got {value!r}")
        source, target = value.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Label map must be source=target, got {value!r}")
        out[source] = target
    return out


def export_mocap_calibration(
    *,
    raw_root: Path,
    formatted_root: Path,
    detections_root: Path,
    out_json: Path,
    run: str,
    sample_count: int = 160,
    ransac_reproj_px: float = 8.0,
    label_map: list[str] | None = None,
) -> dict:
    """Solve camera extrinsics from mocap 3D points and export upload JSON.

    Output cameras use ``R`` and ``t`` as direct world-to-camera extrinsics,
    matching :mod:`firetrack.triangulate_uploads`.
    """
    mocap = load_mocap_6d(mocap_path(raw_root, run))
    if DRONE_BODY not in mocap.bodies:
        raise RuntimeError(f"Mocap file for {run!r} has no {DRONE_BODY!r} body")

    labels = _parse_label_map(label_map)
    manifest = _manifest_by_label(formatted_root)
    cameras = []
    results = []
    run_dir = formatted_root / run
    sessions = sorted(p for p in run_dir.glob(f"{run}_*/video.mp4"))
    if not sessions:
        raise RuntimeError(f"No formatted videos found under {run_dir}")

    for video in sessions:
        session_label = video.parent.name
        phone = session_label.rsplit("_", 1)[-1]
        session = make_session(formatted_root, detections_root, run, phone)
        offset, corr, fit = find_calibration_time_offset(
            session,
            mocap,
            sample_count=sample_count,
            radius_s=5.0,
            coarse_step_s=0.1,
            fine_step_s=0.01,
            ransac_reproj_px=ransac_reproj_px,
        )
        R, _ = cv2.Rodrigues(fit.rvec)
        row = manifest.get(session.label)
        source_uuid = str((row or {}).get("source_uuid") or "")
        upload_label = labels.get(source_uuid) or labels.get(session.label) or _default_upload_label(session.label, row)
        med = float(np.median(fit.errors_px))
        center = camera_center_world(fit.rvec, fit.tvec)
        cameras.append({
            "video": upload_label,
            "K": session.intrinsics.K.tolist(),
            "resolution": [int(session.width), int(session.height)],
            "dist": session.intrinsics.dist_coeffs.tolist(),
            "R": R.tolist(),
            "t": fit.tvec.tolist(),
            "source": {
                "type": "mocap_pnp",
                "run": run,
                "formatted_label": session.label,
                "phone": phone,
                "source_uuid": source_uuid,
                "mocap_path": str(mocap_path(raw_root, run)),
                "world_frame": "mocap",
                "time_offset_s": float(offset),
                "pnp_inliers": int(len(fit.inlier_indices)),
                "correspondences": int(len(corr.points_world)),
                "median_reproj_px": med,
                "camera_center_world": center.tolist(),
            },
        })
        results.append({
            "video": upload_label,
            "formatted_label": session.label,
            "phone": phone,
            "status": "calibrated",
            "time_offset_s": float(offset),
            "correspondences": int(len(corr.points_world)),
            "inliers": int(len(fit.inlier_indices)),
            "median_reproj_px": med,
            "camera_center_world": center.tolist(),
        })

    payload = {"cameras": cameras}
    validate_calibration(payload)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"out_json": str(out_json), "run": run, "results": results}

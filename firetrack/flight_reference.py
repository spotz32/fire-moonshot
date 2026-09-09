"""Flight-log references for georeferenced no-mocap reconstructions.

References are separate artifacts: no pose, spatial alignment, or clock shift
is fitted to the reconstructed path, and reconstruction arrays are untouched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FlightTrack:
    epochs: np.ndarray
    lat_lon_alt: np.ndarray
    clock_spread_s: float


def enu_frame(calibration: dict) -> dict:
    """Accept explicit frame metadata and the colleague's documented legacy form."""
    frames = []
    for cam in calibration.get("cameras", []):
        source = cam.get("source") or {}
        value = source.get("world_frame") or source.get("frame") or calibration.get("world_frame") or calibration.get("frame")
        if isinstance(value, dict):
            frame = dict(value)
        elif isinstance(value, str):
            match = re.fullmatch(
                r"ENU about lat ([+-]?[\d.]+) lon ([+-]?[\d.]+), U = AMSL(?:;.*)?", value
            )
            if not match:
                raise ValueError("Flight-log comparison requires an ENU calibration with a geographic origin and AMSL altitude.")
            frame = {"axes": "ENU", "origin_lat_deg": float(match[1]), "origin_lon_deg": float(match[2]), "altitude_reference": "AMSL"}
        else:
            raise ValueError("Calibration is missing its geographic coordinate frame; cannot align the flight log safely.")
        lat, lon = float(frame.get("origin_lat_deg", np.nan)), float(frame.get("origin_lon_deg", np.nan))
        if frame.get("axes") != "ENU" or frame.get("altitude_reference") != "AMSL" or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Unsupported calibration frame: expected ENU, a valid latitude/longitude origin, and AMSL altitude.")
        frames.append({"axes": "ENU", "origin_lat_deg": lat, "origin_lon_deg": lon, "altitude_reference": "AMSL"})
    if not frames or any(frame != frames[0] for frame in frames[1:]):
        raise ValueError("All calibrated cameras must use the same geographic frame.")
    return frames[0]


def read_flight_track(path: Path) -> FlightTrack:
    from pymavlink import DFReader
    from pymavlink.mavextra import gps_time_to_epoch

    reader = DFReader.DFReader_binary(str(path))
    positions, clocks = [], []
    try:
        while True:
            msg = reader.recv_match(type=["GPS", "POS"])
            if msg is None:
                break
            if not hasattr(msg, "TimeUS"):
                continue
            boot = float(msg.TimeUS) * 1e-6
            if msg.get_type() == "GPS":
                if getattr(msg, "I", 0) != 0 or getattr(msg, "Status", 0) < 3 or getattr(msg, "GWk", 0) <= 0:
                    continue
                # pymavlink converts GPS week/time-of-week to UTC, including
                # the GPS-UTC leap-second offset for these modern recordings.
                clocks.append(gps_time_to_epoch(msg.GWk, msg.GMS) - boot)
            else:
                row = (boot, float(msg.Lat), float(msg.Lng), float(msg.Alt))
                if all(np.isfinite(row)) and -90 <= row[1] <= 90 and -180 <= row[2] <= 180 and (row[1] or row[2]):
                    positions.append(row)
    finally:
        reader.close()
    if len(clocks) < 2 or len(positions) < 2:
        raise ValueError(f"{path.name}: needs valid GPS time and fused POS position messages.")
    offsets = np.asarray(clocks)
    offset = float(np.median(offsets))
    spread = float(np.percentile(offsets, 95) - np.percentile(offsets, 5))
    if spread > 0.5:
        raise ValueError(f"{path.name}: GPS clock mapping is unstable ({spread:.3f} s spread).")
    rows = np.asarray(positions)
    if np.any(np.diff(rows[:, 0]) < 0):
        raise ValueError(f"{path.name}: boot clock resets inside the log; split the recording first.")
    _, index = np.unique(rows[:, 0], return_index=True)
    rows = rows[index]
    return FlightTrack(rows[:, 0] + offset, rows[:, 1:], spread)


def to_calibration_frame(lat_lon_alt: np.ndarray, frame: dict) -> np.ndarray:
    """WGS84 local horizontal tangent coordinates; retain absolute AMSL height."""
    lat, lon = np.radians(lat_lon_alt[:, 0]), np.radians(lat_lon_alt[:, 1])
    lat0, lon0 = np.radians([frame["origin_lat_deg"], frame["origin_lon_deg"]])
    a, e2 = 6378137.0, 6.6943799901413165e-3

    def surface_xyz(phi, lam):
        n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        return np.array([n * np.cos(phi) * np.cos(lam), n * np.cos(phi) * np.sin(lam), n * (1 - e2) * np.sin(phi)])

    delta = surface_xyz(lat, lon).T - surface_xyz(lat0, lon0)
    east = delta @ np.array([-np.sin(lon0), np.cos(lon0), 0])
    north = delta @ np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    return np.column_stack([east, north, lat_lon_alt[:, 2]])


def sample_track(track: FlightTrack, epochs: np.ndarray, frame: dict, *, max_gap_s: float = 1.0) -> np.ndarray:
    points = to_calibration_frame(track.lat_lon_alt, frame)
    right = np.searchsorted(track.epochs, epochs, side="left")
    right = np.clip(right, 0, len(track.epochs) - 1)
    left = np.maximum(0, right - 1)
    exact = np.abs(track.epochs[right] - epochs) <= 1e-6
    valid = (epochs >= track.epochs[0]) & (epochs <= track.epochs[-1])
    valid &= exact | ((track.epochs[right] - track.epochs[left]) <= max_gap_s)
    result = np.column_stack([np.interp(epochs, track.epochs, points[:, k]) for k in range(3)])
    result[~valid] = np.nan
    return result


def calibration_clock_delta(calibration: dict) -> float:
    """Return the calibration-only camera_time - log_time lag.

    This value is used only to select a log position for an optional reference
    comparison. It must never shift synchronized tracking cameras relative to
    one another.
    """
    clock = calibration.get("calibration_clock")
    if isinstance(clock, dict) and isinstance(clock.get("camera_minus_log_s"), (int, float)):
        value = float(clock["camera_minus_log_s"])
        if not np.isfinite(value):
            raise ValueError("Calibration camera/log lag is not finite.")
        return value
    values = []
    for camera in calibration.get("cameras", []):
        source = camera.get("source") or {}
        value = source.get("calibration_clock_delta_s")
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 0.0
    if not np.isfinite(values).all() or max(values)-min(values) > 1e-6:
        raise ValueError("Cameras disagree about the shared calibration camera/log lag.")
    return float(np.mean(values))


def comparison_metrics(reconstructed: np.ndarray, reference: np.ndarray) -> dict:
    valid = np.isfinite(reconstructed).all(axis=1) & np.isfinite(reference).all(axis=1)
    delta = reconstructed[valid] - reference[valid]
    distances = np.linalg.norm(delta, axis=1)
    return {
        "n_compared": int(valid.sum()),
        "n_reference": int(np.isfinite(reference).all(axis=1).sum()),
        "rmse_m": float(np.sqrt(np.mean(distances ** 2))) if len(delta) else None,
        "median_error_m": float(np.median(distances)) if len(delta) else None,
        "horizontal_rmse_m": float(np.sqrt(np.mean(np.sum(delta[:, :2] ** 2, axis=1)))) if len(delta) else None,
        "vertical_rmse_m": float(np.sqrt(np.mean(delta[:, 2] ** 2))) if len(delta) else None,
    }


def attach_reference(trajectory_dir: Path, logs: list[Path], calibration_path: Path) -> dict:
    trajectory_path = trajectory_dir / "trajectory.npz"
    fingerprint = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    with np.load(trajectory_path, allow_pickle=False) as data:
        epochs = np.array(data["epoch_times_s"], dtype=float)
        smooth = np.array(data["trajectory_smooth"], dtype=float)
        raw = np.array(data["trajectory_raw"], dtype=float)
    if len(epochs) < 2 or not np.isfinite(epochs).all() or np.any(np.diff(epochs) <= 0):
        raise ValueError("Reconstruction needs increasing absolute timestamps for flight-log comparison.")
    summary_path = trajectory_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    calibration = summary.get("calibration_snapshot")
    frame_source = "reconstruction calibration snapshot"
    if not calibration:
        calibration = json.loads(calibration_path.read_text())
        frame_source = "current calibration (older reconstruction has no calibration snapshot)"
    frame = enu_frame(calibration)
    clock_delta_s = calibration_clock_delta(calibration)
    log_query_epochs = epochs - clock_delta_s
    candidates, failures = [], []
    for path in sorted(logs):
        try:
            track = read_flight_track(path)
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
            continue
        if track.epochs[-1] < log_query_epochs[0] or track.epochs[0] > log_query_epochs[-1]:
            continue
        reference = sample_track(track, log_query_epochs, frame)
        if np.isfinite(reference).all(axis=1).sum() >= 2:
            candidates.append((path, track, reference))
    if not candidates:
        raise ValueError("No flight log overlaps the reconstruction in UTC with usable positions." + (" " + "; ".join(failures) if failures else ""))
    if len(candidates) != 1:
        raise ValueError("Multiple flight logs overlap this run; the tracked drone's log cannot be selected automatically.")
    path, track, reference = candidates[0]
    metric = comparison_metrics(smooth, reference)
    warnings = []
    if "older" in frame_source:
        warnings.append("Frame taken from the current calibration; confirm it is the calibration used for this reconstruction.")
    sources = [cam.get("source") or {} for cam in calibration.get("cameras", [])]
    windows = [s.get("window_unix") for s in sources]
    if any(isinstance(w, list) and len(w) == 2 and max(log_query_epochs[0], w[0]) <= min(log_query_epochs[-1], w[1]) for w in windows):
        warnings.append("This recording overlaps the calibration flight; comparison measures calibration consistency, not independent validation.")
    metadata = {
        "label": "Flight-log reference", "position_source": "ArduPilot POS (fused onboard estimate)",
        "log": path.name, "world_frame": frame, "frame_source": frame_source,
        "time_basis": "log_time = camera_time - calibration camera/log lag",
        "calibration_clock_delta_s": clock_delta_s,
        "clock_spread_s": track.clock_spread_s, "max_interpolation_gap_s": 1.0,
        "trajectory_sha256": fingerprint, "metrics": metric, "warnings": warnings,
        "raw_metrics": comparison_metrics(raw, reference),
    }
    # The hash and timestamps prevent a later reconstruction from silently
    # displaying a reference prepared for an earlier result.
    if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("Reconstruction changed while preparing the reference; try again.")
    with (trajectory_dir / "flight_reference.npz.tmp").open("wb") as f:
        np.savez(f, epoch_times_s=epochs, points_world=reference)
    (trajectory_dir / "flight_reference.npz.tmp").replace(trajectory_dir / "flight_reference.npz")
    temp = trajectory_dir / "flight_reference.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    temp.replace(trajectory_dir / "flight_reference.json")
    with (trajectory_dir / "flight_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "epoch_s", "reconstructed_x", "reconstructed_y", "reconstructed_z", "reference_x", "reference_y", "reference_z", "error_m"])
        for i, (epoch, rec, ref) in enumerate(zip(epochs, smooth, reference)):
            values = [epoch, *rec, *ref, np.linalg.norm(rec - ref)]
            writer.writerow([i, *[float(v) if np.isfinite(v) else "" for v in values]])
    print(f"Flight-log reference: {path.name}; {metric['n_reference']}/{len(epochs)} frames covered")
    if metric["rmse_m"] is not None:
        print(f"Reconstruction vs flight-log reference: RMSE {metric['rmse_m']:.3f} m; median {metric['median_error_m']:.3f} m")
    for warning in warnings:
        print(f"Note: {warning}")
    return metadata


def load_reference(trajectory_path: Path, epochs: np.ndarray | None) -> dict | None:
    metadata_path = trajectory_path.with_name("flight_reference.json")
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("error"):
            return {"error": metadata["error"]}
        if metadata.get("trajectory_sha256") != hashlib.sha256(trajectory_path.read_bytes()).hexdigest():
            return {"error": "Reconstruction changed. Rerun Triangulate to update the flight-log reference."}
        with np.load(trajectory_path.with_name("flight_reference.npz"), allow_pickle=False) as data:
            times, points = data["epoch_times_s"], data["points_world"]
        if epochs is None or not np.array_equal(times, epochs) or points.shape != (len(times), 3):
            return {"error": "Reference timestamps do not match this reconstruction."}
        return {**metadata, "points": points}
    except (OSError, ValueError, KeyError):
        return {"error": "Flight-log reference could not be loaded. Rerun Triangulate to regenerate it."}

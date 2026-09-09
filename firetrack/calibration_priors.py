"""Optional, explicitly verified phone priors for the no-mocap solver."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .calibration_solver import Priors
from .flight_reference import to_calibration_frame


def _unit(value):
    v = np.asarray(value, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError("Sensor prior needs a finite nonzero 3D direction.")
    return v / np.linalg.norm(v)


def _sigma(value, size=1):
    v = np.broadcast_to(np.asarray(value, float), (size,)).copy()
    if not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("Prior uncertainties must be finite and positive.")
    return v


def _samples(path, start, end, fields):
    samples = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            t = float(row['timestamp_us']) * 1e-6
            if start <= t <= end:
                values = [float(row[k]) for k in fields]
                if np.isfinite(values).all():
                    samples.append(values)
    if not samples:
        raise ValueError(f"{path.name}: no finite sensor samples in the calibration video window.")
    return np.asarray(samples)


def load_priors(folder: Path, frame: dict, start: float, end: float) -> Priors:
    data = json.loads((folder / 'camera.json').read_text())
    config = data.get('calibration_priors')
    if not config:
        return Priors()
    if not isinstance(config, dict) or config.get('conventions_verified') is not True:
        raise ValueError("calibration_priors requires conventions_verified=true; phone axes/altitude must be verified first.")
    priors = Priors()
    if config.get('use_phone_gps'):
        if config.get('gps_altitude_reference') != 'AMSL':
            raise ValueError("Phone GPS prior requires verified AMSL altitude; ellipsoid heights must be converted first.")
        values = _samples(folder / 'gps.csv', start, end, ['latitude', 'longitude', 'altitude', 'accuracy_m'])
        if np.any(np.abs(values[:, 0]) > 90) or np.any(np.abs(values[:, 1]) > 180):
            raise ValueError("Phone GPS coordinates are outside valid latitude/longitude ranges.")
        priors.position = np.median(to_calibration_frame(values[:, :3], frame), axis=0)
        priors.position_sigma = _sigma(config.get('gps_sigma_m', 2.0), 3)
        priors.position_sigma[:2] = np.maximum(priors.position_sigma[:2], np.median(values[:, 3]))
        priors.names.append('phone_GPS')
    if config.get('use_gravity') or config.get('use_magnetometer'):
        device_to_camera = np.asarray(config.get('device_to_camera'), float)
        if (device_to_camera.shape != (3, 3) or not np.isfinite(device_to_camera).all()
                or not np.allclose(device_to_camera.T @ device_to_camera, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(device_to_camera), 1, atol=1e-6)):
            raise ValueError("device_to_camera must be a verified proper rotation into decoded-video camera axes.")
        if config.get('use_gravity'):
            if config.get('accelerometer_convention') != 'specific_force':
                raise ValueError("Gravity prior requires accelerometer_convention=specific_force (stationary reading points up).")
            values = _samples(folder / 'imu.csv', start, end, ['accel_x', 'accel_y', 'accel_z'])
            if not 8 < np.median(np.linalg.norm(values, axis=1)) < 11:
                raise ValueError("Accelerometer prior is not consistent with stationary specific force in m/s^2.")
            observed = _unit(device_to_camera @ np.median(values, axis=0))
            sigma = float(np.radians(_sigma(config.get('gravity_sigma_deg', 1.7))[0]))
            priors.directions.append((np.array([0., 0., 1.]), observed, sigma))
            priors.names.append('gravity')
        if config.get('use_magnetometer'):
            # A verified world magnetic direction includes declination/inclination;
            # the device magnetometer cannot be assumed to point to true north.
            world = _unit(config.get('magnetic_field_world'))
            values = _samples(folder / 'imu.csv', start, end, ['mag_x', 'mag_y', 'mag_z'])
            observed = _unit(device_to_camera @ np.median(values, axis=0))
            sigma = float(np.radians(_sigma(config.get('magnetometer_sigma_deg', 20.0))[0]))
            priors.directions.append((world, observed, sigma))
            priors.names.append('magnetometer_direction')
    return priors

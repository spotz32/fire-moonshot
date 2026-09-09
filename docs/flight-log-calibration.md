# Flight-log calibration (no mocap)

The Calibrate button and `firetrack calibrate-from-log` jointly estimate static
camera poses, one common `fx = fy` focal length per camera, and one shared
camera-to-log lag. Neither mocap pipeline is changed.

## Inputs

Upload a calibration ZIP with a folder per camera containing `video.mp4`,
`metadata.json`, and `camera.json`. Include the matching ArduPilot `.BIN` log(s)
anywhere in the ZIP. Then Annotate, Detect, and Calibrate. All cameras must pass;
a failure does not replace an existing calibration. At least two cameras are
required. Clear calibration first when replacing an entire calibration flight.

`metadata.json` must contain `startTime` in Unix UTC microseconds. Detection NPZ
files provide centroids, frame indices, FPS, and image dimensions. Frame time is
start time plus frame index / FPS. The calibration clips must already be
synchronized with one another. The fitted shared lag relates the synchronized
camera clock to the calibration flight log; it is not a camera-to-camera repair.

Recommended `camera.json` format (illustrative numbers, not a calibration):

```json
{
  "K": [[1400, 0, 540], [0, 1400, 960], [0, 0, 1]],
  "dist": [0, 0, 0, 0, 0],
  "resolution": [1080, 1920]
}
```

K and OpenCV distortion coefficients must describe the actual decoded video
pixels. Explicit K is never scaled, cropped, or rotated. The supplied `fx` and
`fy` are averaged to initialize one common focal length, which is optimized by
default and written as `fx = fy`; principal point, skew, and supported OpenCV
distortion coefficients remain fixed.
Resolution mismatches are rejected. Zeros for distortion are appropriate only
if verified.
Legacy Android factory camera.json files still work through the existing sensor
mapping, with an explicit warning that the mapping assumptions need verification.
An infinity-focus calibration may not describe autofocus recordings correctly.

The log reader uses fused ArduPilot POS latitude/longitude/AMSL altitude and GPS
week/time-of-week mapped to UTC via pymavlink. Logs without these messages or with
unstable clock mapping are rejected; raw GPS positions are not silently used as
a replacement. All cameras use one ENU horizontal frame centered at the first
valid POS latitude/longitude, with absolute AMSL height as U. Results uses the
same frame convention. Multiple overlapping uploaded logs are ambiguous: select
the tracked drone's log explicitly using the CLI.

## Solver

- Piecewise cubic smoothing splines provide the calibration drone's position.
  Log gaps longer than one second are not interpolated, and no extrapolation is
  allowed. Observations must remain supported over the whole offset search range.
- Multi-start PnP RANSAC initializes every named camera over a coarse shared-lag
  grid. At each synchronized camera time, the solver intersects the available
  camera rays. It minimizes both the 3-D difference between that point and the
  flight-log spline and the pixel reprojection disagreement between cameras.
- Continuous robust nonlinear least squares jointly refines every world-to-camera
  rotation, camera position, one common `fx = fy`, and one shared lag. Rotations use SO(3)
  exponential coordinates. CasADi differentiates through ray intersection;
  SciPy supplies the bounded robust solve.
- Iterated consistency gating can reject the worst camera observation at a time
  when at least two other views remain. It cannot choose an alternative drone
  because the NPZ contains one centroid per frame.
- Two of ten temporal blocks (20%) are held out and never used in the fit.
  Export includes their all-observation and inlier-only errors and coverage.
- At least 30 synchronized samples are needed. Stationary/straight flights,
  ambiguous minima, rank-deficient fits, nonconvergence, boundary lag/focal
  estimates, insufficient per-camera support, invalid depth, and under 60%
  training or held-out consistency cause rejection.
- Diagnostics include Jacobian singular values, conditional local timing
  uncertainty, enabled priors, and warnings. These are not absolute accuracy
  guarantees: log error, autofocus mismatch, rolling shutter, and clock drift are
  not estimated. A calibration-flight comparison is not independent ground truth.

## Timing and output

The convention is `log_time = camera_time - shared_clock_delta_s`. The default
initial value is 0.22 seconds, but the solver estimates it within the configured
bounds. It is one shared calibration parameter because the input camera clips
are already synchronized.

The exported JSON stores this under `calibration_clock.camera_minus_log_s` and
as camera provenance. `source.time_offset_s` remains zero. Therefore, the lag is
used to associate calibration frames with calibration-log positions, but it does
not shift future tracking videos. Tracking runs need synchronized cameras and
the saved calibration; they do not need an ArduPilot log. If a tracking log is
available only for evaluation, Results uses the saved lag when sampling that log.

R is world-to-camera; `t = -R @ position`. The usual `cameras` array is retained,
including K/dist, resolution, source frame, and diagnostics. The app still
downloads this JSON after successful calibration. Camera poses and intrinsics
remain reusable only while the physical cameras and capture mode remain unchanged.

CLI example from the app directory:

```bash
python -m firetrack.cli calibrate-from-log \
  --log /path/to/flight.BIN \
  --run-root /path/to/work/uploads \
  --detections-root /path/to/work/detections \
  --out-json /path/to/work/uploads_calibration.json
```

Defaults: `--sample-count 600`, `--smoothing-m 0.1` (per-axis RMS allowance),
`--huber-px 3`, `--ransac-reproj-px 8`, `--clock-lag-s 0.22`,
`--offset-search-radius-s 2`, `--offset-search-step-s 0.25`, and
`--focal-sigma inf`. The radius bounds continuous shared-lag refinement; the
step controls initialization only. Radius zero fixes the lag. Focal sigma `0`
fixes `fx = fy` to the mean of the supplied values; `inf` estimates the common
focal length; a value such as `0.02` applies a 2% soft prior. Do not raise the pixel gate merely
to make poor calibration pass.

## Optional phone priors

Priors are OFF by default. Add `calibration_priors` to each camera.json only after
verifying its sensor conventions. Unverified configured priors cause an error.
The field is accepted by the existing ZIP upload; no additional UI form is needed.

Supported fields:

| Field | Meaning |
| --- | --- |
| `conventions_verified: true` | Explicit acknowledgement of the coordinate and sensor conventions below. |
| `use_phone_gps: true` | Read timestamp_us, latitude, longitude, altitude, accuracy_m from gps.csv. |
| `gps_altitude_reference: "AMSL"` | Required for GPS; convert ellipsoid altitude before enabling. |
| `gps_sigma_m` | Positive scalar or XYZ uncertainties in meters, default 2. Horizontal uncertainty is at least the median reported accuracy. |
| `device_to_camera` | Proper 3x3 rotation from phone sensor axes to decoded-video camera axes (x right, y down, z forward). No orientation is inferred. |
| `use_gravity: true` | Use stationary accelerometer samples in imu.csv. |
| `accelerometer_convention: "specific_force"` | Required for gravity: stationary accelerometer points upward, in m/s^2. |
| `gravity_sigma_deg` | Direction uncertainty, default 1.7 degrees. |
| `use_magnetometer: true` | Use mag_x, mag_y, mag_z samples from imu.csv. |
| `magnetic_field_world` | Required ENU magnetic-field direction, including verified declination/inclination. Not simply true north. |
| `magnetometer_sigma_deg` | Direction uncertainty, default 20 degrees. |

Gravity and magnetic priors use unit-vector chord residuals scaled by angular
uncertainty in radians (small-angle approximation). Samples are selected in the
phone video timestamp window and summarized by their median. Device motion,
magnetic interference, and sensor-to-camera misalignment can invalidate these
priors even when the CSV fields exist. Leave them disabled unless verified.

## Validation

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Tests cover joint pose/common-focal/shared-lag recovery, enforced `fx = fy`, camera
identity, distortion validation, missing data/outliers, degenerate motion, spline
gaps, sensor conventions, atomic replacement, run-scoped detection lookup,
explicit-intrinsic handling, tracking timestamp isolation, and optional
flight-log reference timing.

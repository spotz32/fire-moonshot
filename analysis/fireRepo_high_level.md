# fireRepo: FireTrack overview

FireTrack reconstructs a drone's **3D flight path from videos recorded by two or more stationary cameras**. It first finds the drone in each image, then uses camera geometry and synchronized times to estimate its position. It can compare the result with motion capture or an ArduPilot flight log, and optionally classify a completed tracking trajectory using an external model. It is a research-oriented processing pipeline with a browser dashboard and a command-line interface. It does not detect fires or control a drone. ([README.md](fireRepo/README.md):3; [cli.py](fireRepo/firetrack/cli.py):176; [classification.py](fireRepo/firetrack/classification.py):130.)

This report describes the checkout inspected on 2026-09-28, commit `1bed945d11f5d54c46969634be9f7e02c6518df6`. For algorithms, setup commands, examples, findings, and inspection coverage, read [the detailed analysis](fireRepo_detailed_analysis.md).

## Inputs, outputs, and workflow

1. **Prepare videos and their metadata.** The older workflow understands a particular multi-phone dataset called “5-27.” The newer No-Mocap workflow accepts camera folders with videos, camera parameters, and timestamps.
2. **Find the drone.** SAM3, a machine-learning segmentation model that marks an object's pixels, follows a text prompt and optionally a user click. The average pixel position of the resulting mask becomes a 2D detection. Missing detections remain marked as missing.
3. **Calibrate the cameras.** Calibration describes each camera's lens and its position and orientation. Supply it manually, derive it from motion capture, or estimate it from a separate calibration flight with an ArduPilot `.BIN` log.
4. **Reconstruct and smooth the path.** The code aligns camera observations in time, triangulates positions from multiple views, rejects inconsistent observations, and smooths the result using a motion model.
5. **Inspect and export.** The browser shows video frames, editable detections, and a rotatable trajectory. Results include CSV tables, NumPy `.npz` array archives, and JSON summaries. Optional flight-log comparisons and firmware classification produce additional files.

The implementation spans [detect.py](fireRepo/firetrack/detect.py):218, [calibrate_logs.py](fireRepo/firetrack/calibrate_logs.py):246, [triangulate_uploads.py](fireRepo/firetrack/triangulate_uploads.py):262, and [results_page.py](fireRepo/firetrack/results_page.py):128. Classification calls an external package; its internal model and training procedure are outside this checkout.

| Workflow | Where camera calibration comes from | Important distinction |
|---|---|---|
| No-Mocap calibration and tracking | Calibration videos plus a flight log, or supplied calibration JSON | Reuses the calibrated stationary camera arrangement for separate named tracking runs. |
| Mocap / dataset | Qualisys motion-capture recordings and dataset-specific conventions | Depends on known phone models, run names, filenames, and coordinate transforms. |
| Mocap Raw | Original dataset videos with mocap calibration | Avoids the older video's rotation/transcoding path. |

“No-Mocap” means no external motion-capture system is required; automatic calibration still needs a known reference trajectory from the flight log. Accurate reconstruction depends on consistent camera labels, valid calibration, and reliable timing. ([webui.py](fireRepo/firetrack/webui.py):640; [triangulate_527.py](fireRepo/firetrack/triangulate_527.py):908.)

## Technology and layout

The backend is **Python**, packaged as `docker-fire-tracking` with the `firetrack` command. The browser interface is plain **HTML, CSS, and JavaScript embedded in Python strings**, served using Python's built-in HTTP server; there is no React or Flask project. NumPy and SciPy handle numerical work; OpenCV handles videos and camera geometry; CasADi provides derivatives for calibration; pymavlink reads flight logs; PyTorch and SAM3 perform detection; Hugging Face supplies model downloads. Docker adds FFmpeg and a CUDA-enabled PyTorch environment. ([pyproject.toml](fireRepo/pyproject.toml):5–35; [Dockerfile](fireRepo/Dockerfile):1–42.)

`firetrack/` contains the application, `tests/` contains 61 test methods in 10 files, `docs/` explains flight-log calibration, `scripts/` prepares the optional classifier, and `vendor/` contains small package markers and a pinned model manifest. Runtime recordings, model checkpoints, and results are not included.

## Concise file inventory

Paths below are relative to `fireRepo/`; every listed file was read in full for this analysis.

| File | Purpose |
|---|---|
| `README.md` | Introduces FireTrack and documents Docker and the older CLI workflow. |
| `pyproject.toml` | Defines package metadata, dependencies, command entry point, and pytest discovery. |
| `requirements.txt` | Repeats the runtime dependencies used by Docker. |
| `Dockerfile` | Builds the GPU runtime and prepares classifier assets. |
| `.dockerignore` | Excludes local data, caches, and artifacts from Docker's build context. |
| `.gitignore` | Excludes generated outputs, environments, recordings, and model binaries from Git. |
| `docs/flight-log-calibration.md` | Explains log-based calibration, timing, assumptions, and validation. |
| `firetrack/__init__.py` | Declares the package version. |
| `firetrack/__main__.py` | Implements `python -m firetrack`. |
| `firetrack/cli.py` | Parses commands and dispatches processing stages. |
| `firetrack/webui.py` | Coordinates uploads, HTTP routes, jobs, calibration, and tracking runs. |
| `firetrack/dashboard.py` | Contains the main dashboard's HTML, CSS, and JavaScript. |
| `firetrack/clicks.py` | Stores click annotations and serves the annotation interface. |
| `firetrack/jobs.py` | Runs one background job and captures progress and logs. |
| `firetrack/sources.py` | Discovers video sources and preserves older tracking-directory names. |
| `firetrack/dataset_527.py` | Defines dataset runs, phone models, and video discovery. |
| `firetrack/format_527.py` | Converts dataset videos and records normalization metadata. |
| `firetrack/detect.py` | Runs SAM3 and saves per-frame 2D centroids and optional masks. |
| `firetrack/mocap.py` | Reads Qualisys positions, rotations, and recording times. |
| `firetrack/calibrate_mocap.py` | Exports mocap-derived poses in the upload calibration format. |
| `firetrack/calibrate_logs.py` | Reads ArduPilot logs and orchestrates camera calibration. |
| `firetrack/calibration_priors.py` | Validates optional measured camera-position and attitude constraints. |
| `firetrack/calibration_solver.py` | Provides the earlier single-camera pose and timing solver. |
| `firetrack/joint_calibration_solver.py` | Fits all camera poses, focal lengths, and one shared timing lag. |
| `firetrack/triangulate_527.py` | Implements shared geometry and dataset-specific reconstruction. |
| `firetrack/triangulate_uploads.py` | Reconstructs uploaded runs using supplied camera calibration. |
| `firetrack/ekf.py` | Smooths positions using a linear Kalman filter and backward pass. |
| `firetrack/flight_reference.py` | Compares reconstruction with flight logs without fitting it to them. |
| `firetrack/results.py` | Loads result artifacts, applies centroid edits, and resolves downloads. |
| `firetrack/results_page.py` | Displays synchronized camera views and an interactive trajectory plot. |
| `firetrack/classify.py` | Validates and loads the pinned external classifier model. |
| `firetrack/classification.py` | Manages per-run classification and invalidates stale predictions. |
| `scripts/prepare_classifier.py` | Installs the classifier wheel and caches its verified model. |
| `tests/test_calibration_solver.py` | Checks the single-camera solver, derivatives, and difficult trajectories. |
| `tests/test_joint_calibration_solver.py` | Checks joint pose, lag, focal recovery, and camera validation. |
| `tests/test_calibration_integration.py` | Checks calibration export, timing, input errors, logs, and priors. |
| `tests/test_calibration_clear.py` | Checks scoped calibration deletion, cache reset, and busy-job protection. |
| `tests/test_camera_repair.py` | Checks per-camera clearing and redetection without removing other inputs. |
| `tests/test_flight_reference.py` | Checks coordinates, timing, comparisons, and stale-reference detection. |
| `tests/test_automatic_flight_reference.py` | Checks automatic comparison and reconstruction-preserving error handling. |
| `tests/test_tracking_classification.py` | Checks classification persistence, invalidation, and model reuse with mocks. |
| `tests/test_tracking_names.py` | Checks compatibility between old and new tracking-directory names. |
| `tests/test_tracking_output_clear.py` | Checks selected-run output deletion and path protections. |
| `vendor/__init__.py` | Makes runtime asset directories importable as a package. |
| `vendor/sam3_assets/__init__.py` | Marks the expected SAM3 vocabulary asset package. |
| `vendor/moonshot_classifier_assets/__init__.py` | Marks the classifier metadata package. |
| `vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json` | Pins the model repository, filename, revision, and checksum. |

## Where to start and what remains unverified

Read `cli.py`, then `webui.py`'s stage dispatch, then `detect.py` and `triangulate_uploads.py`; use the [detailed report's reading order](fireRepo_detailed_analysis.md#6-recommended-reading-order) for the calibration mathematics.

The inspection covered all **47 application, configuration, documentation, test, and asset-metadata files**. All **39 Python files compiled in memory**; TOML parsing and classifier-manifest validation passed. A standard-library check reproduced a job-log issue in which the dashboard console stops receiving new lines after 5,000. The pipeline and repository tests were **not run** because dependencies, recordings, and models are absent. Python 3.14 also emitted an invalid-escape warning in `dashboard.py:462`. File hashes confirm the repository was not modified.

The Dockerfile requires `vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz`, which is missing from this checkout. The detailed report also documents source-confirmed problems with dataset clearing, cross-camera edits, fixed-lag calibration export, and stale outputs. These are material considerations before using the dashboard with valuable working results. Two practical rules follow from the code:
- Tracking-flight camera folders must use exactly the same names as the calibration cameras (for example `cam1` and `cam2`).
- Every tracking clip needs its own `metadata.json` start time; otherwise it silently inherits the calibration flight's time.


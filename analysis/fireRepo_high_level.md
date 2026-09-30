# fireRepo: FireTrack overview

**FireTrack reconstructs a drone's 3D flight path from videos taken by two or more stationary cameras**, typically phones on tripods. It finds the drone in every frame with Meta's SAM3 segmentation model and works out where each camera is and how it points ("calibration"). It then lines the cameras up in time, intersects their lines of sight to get 3D positions, and smooths the path. A browser dashboard drives the whole process. Optional extras compare the result with the drone's own ArduPilot flight log and guess whether the drone ran ArduPilot or PX4 firmware using an external classifier. It is a research pipeline, not a product. Despite the name, it does not detect fires or control drones; "fire" appears to be a project name (the repository is `fire-moonshot`).

This overview describes `main` at `769561474c793ae7a4a7813bc520a9241e6224a6`, analyzed on 2026-09-29. The application code is unchanged since commit `1bed945` (2026-09-09). For algorithms, commands, evidence, and line-level citations, see **[the detailed analysis](fireRepo_detailed_analysis.md)**.

## What goes in, what comes out

| | Details |
|---|---|
| **Inputs** | Per camera: `video.mp4`, `metadata.json` (clip start time in UTC microseconds), and `camera.json` (lens parameters of the decoded video). Plus one of: an ArduPilot `.BIN` flight log of a separate calibration flight, a Qualisys motion-capture recording from the lab "5-27" dataset, or camera poses typed into a JSON file. Optionally, one click on the drone per camera. |
| **Outputs** | Per camera: `centroids.npz` (the drone's pixel position per frame). A reusable `uploads_calibration.json` (camera positions, orientations, focal lengths). Per flight: `trajectory.npz`/`.csv` (raw and smoothed 3D positions per frame) and `summary.json`. Optionally, a flight-log comparison (`flight_reference.json`, `flight_comparison.csv`) and `prediction.json` (ArduPilot vs PX4). |
| **Capabilities** | Text- or click-prompted segmentation; automatic multi-camera calibration from a flight log; triangulation that ignores a disagreeing camera; Kalman smoothing; synchronized 2D/3D playback; manual correction of detections; scoped clean-up and re-processing of single cameras. |

## How it works, in plain language

1. **Upload** camera clips (as a ZIP) through the dashboard. The older dataset mode also re-encodes videos with FFmpeg.
2. **Find the drone.** SAM3 follows the text prompt "drone" or the user's click through the video. The average pixel position of the drone's outline in each frame becomes that frame's 2D detection.
3. **Calibrate the cameras** once, from a *calibration flight*. The drone's flight log says where it was, and the solver finds the camera positions, orientations, focal lengths, and one clock offset that make the camera views agree with the log. Motion capture or hand-entered poses are the alternatives.
4. **Track later flights** with the cameras left in place. For every frame, the code looks up each camera's detection at the same moment (using each clip's start time), intersects the viewing rays, rejects a camera that disagrees, and smooths the path.
5. **Review and export.** The browser shows the camera frames with detections and a rotatable 3D path. Detections can be corrected, after which the flight must be re-triangulated. Files can be downloaded, and the optional comparison and classification can be run.

~~~mermaid
flowchart LR
  A[Camera clips] --> B[SAM3 detection] --> C[2D drone positions]
  L[Calibration flight log] --> D[Camera calibration]
  C --> D
  C --> E[Triangulate + smooth]
  D --> E --> F[3D trajectory files + browser view]
  F -.-> G[Optional: log comparison, firmware classifier]
~~~

| Workflow (dashboard tab) | Where camera calibration comes from | Notes |
|---|---|---|
| **No-Mocap** (default) | A calibration flight plus its ArduPilot log, or a supplied calibration JSON | Calibration is reused for any number of named tracking runs; no log is needed for tracking. |
| **Mocap** | Qualisys motion capture of the specific "5-27" dataset | Hard-wired to three known phone models and fixed run names. |
| **Mocap Raw** | Same, but using the original videos without re-encoding | Diagnostic variant of the above. |

## Technology

- **Language.** Python (package `firetrack`, distribution `docker-fire-tracking` 0.1.0, command `firetrack`).
- **Browser UI.** Plain HTML/CSS/JavaScript stored as strings inside Python modules, served by Python's built-in `http.server`. No web framework, no build step.
- **Main libraries.** NumPy/SciPy (math), OpenCV (video frames, camera geometry), CasADi (derivatives for calibration), pymavlink (ArduPilot logs), PyTorch + `sam3==0.1.2` (detection), huggingface-hub (model downloads).
- **Deployment.** A CUDA/PyTorch Docker image adds FFmpeg and prefetches the classifier.

## Folders and major components

| Folder | Purpose |
|---|---|
| `firetrack/` | The application: CLI and web server (`cli.py`, `webui.py`, `jobs.py`), browser pages (`dashboard.py`, `results_page.py`, `clicks.py`), detection (`detect.py`), calibration (`calibrate_logs.py`, `joint_calibration_solver.py`, …), reconstruction (`triangulate_uploads.py`, `triangulate_527.py`, `ekf.py`), and results/classification. |
| `tests/` | 61 unit tests in 10 files using synthetic data and mocks. |
| `docs/` | A guide to flight-log calibration. |
| `scripts/` | Classifier installation helper. |
| `vendor/` | Package markers and the pinned classifier-model manifest (no third-party code). |
| `analysis/` | These two reports. |

Recordings, model weights, and results are not in the repository. At runtime everything is stored under a `--work-root` folder.

## File inventory

Paths are relative to the repository root; every file was read in full.

| File | Purpose |
|---|---|
| [analysis/fireRepo_high_level.md](fireRepo_high_level.md) | This standalone overview. |
| [analysis/fireRepo_detailed_analysis.md](fireRepo_detailed_analysis.md) | Full technical analysis with evidence, verification results, and findings. |
| [README.md](../README.md) | Introduces FireTrack, the Docker commands, and the older dataset CLI workflow. |
| [pyproject.toml](../pyproject.toml) | Declares the package, its 8 dependencies, the `firetrack` command, package data, and pytest settings. |
| [requirements.txt](../requirements.txt) | Lists the same runtime dependencies for the Docker build. |
| [Dockerfile](../Dockerfile) | Builds the GPU image, installs FFmpeg, prefetches the classifier, and copies the SAM3 vocabulary file. |
| [.dockerignore](../.dockerignore) | Keeps local data, caches, and outputs out of the Docker build context. |
| [.gitignore](../.gitignore) | Keeps environments, outputs, recordings, weights, and the SAM3 vocabulary out of Git. |
| [docs/flight-log-calibration.md](../docs/flight-log-calibration.md) | Explains calibration inputs, the solver, timing conventions, options, and optional phone sensor priors. |
| [firetrack/\_\_init\_\_.py](../firetrack/__init__.py) | Declares the package version. |
| [firetrack/\_\_main\_\_.py](../firetrack/__main__.py) | Lets `python -m firetrack` run the CLI. |
| [firetrack/cli.py](../firetrack/cli.py) | Defines the nine command-line subcommands and forwards each to its module. |
| [firetrack/webui.py](../firetrack/webui.py) | Runs the web server: uploads, stage dispatch, status, clean-up/repair, and every HTTP route. |
| [firetrack/dashboard.py](../firetrack/dashboard.py) | Holds the main dashboard page (HTML/CSS/JavaScript). |
| [firetrack/results_page.py](../firetrack/results_page.py) | Holds the results page: synchronized camera frames, the 3D view, and the detection editor. |
| [firetrack/clicks.py](../firetrack/clicks.py) | Stores click annotations, serves video frames, and provides the annotator page. |
| [firetrack/jobs.py](../firetrack/jobs.py) | Runs one background job at a time and captures its log and progress. |
| [firetrack/sources.py](../firetrack/sources.py) | Finds videos for each input source and maps old folder names to new ones. |
| [firetrack/dataset_527.py](../firetrack/dataset_527.py) | Encodes the "5-27" dataset's phone models and folder layout. |
| [firetrack/format_527.py](../firetrack/format_527.py) | Rotates and re-encodes dataset videos with FFmpeg. |
| [firetrack/detect.py](../firetrack/detect.py) | Runs SAM3 on each video and saves per-frame drone centroids. |
| [firetrack/flight_reference.py](../firetrack/flight_reference.py) | Reads ArduPilot logs, defines the local map frame, and compares reconstructions with a log. |
| [firetrack/calibrate_logs.py](../firetrack/calibrate_logs.py) | Loads calibration inputs, runs the joint solver, and saves the calibration JSON. |
| [firetrack/joint_calibration_solver.py](../firetrack/joint_calibration_solver.py) | Fits all camera poses, focal lengths, and a shared clock lag to the flight log. |
| [firetrack/calibration_solver.py](../firetrack/calibration_solver.py) | Provides the flight-path spline and the older single-camera solver. |
| [firetrack/calibration_priors.py](../firetrack/calibration_priors.py) | Adds optional phone GPS, gravity, and compass constraints to calibration. |
| [firetrack/mocap.py](../firetrack/mocap.py) | Reads Qualisys motion-capture TSV files. |
| [firetrack/calibrate_mocap.py](../firetrack/calibrate_mocap.py) | Converts a mocap-based camera calibration into the generic calibration JSON. |
| [firetrack/triangulate_527.py](../firetrack/triangulate_527.py) | Holds the shared 3D geometry and the mocap-dataset reconstruction. |
| [firetrack/triangulate_uploads.py](../firetrack/triangulate_uploads.py) | Reconstructs No-Mocap flights from a calibration JSON. |
| [firetrack/ekf.py](../firetrack/ekf.py) | Smooths 3D paths with a Kalman filter and backward pass. |
| [firetrack/results.py](../firetrack/results.py) | Lists and loads results, applies manual detection edits, and resolves downloads. |
| [firetrack/classify.py](../firetrack/classify.py) | Downloads and verifies the pinned classifier model and builds the predictor. |
| [firetrack/classification.py](../firetrack/classification.py) | Manages per-run classification results and hides stale ones. |
| [scripts/prepare_classifier.py](../scripts/prepare_classifier.py) | Installs the pinned classifier package and caches its model. |
| [tests/test_calibration_solver.py](../tests/test_calibration_solver.py) | Tests the older single-camera solver and the flight-path spline. |
| [tests/test_joint_calibration_solver.py](../tests/test_joint_calibration_solver.py) | Tests joint recovery of poses, focal lengths, and clock lag. |
| [tests/test_calibration_integration.py](../tests/test_calibration_integration.py) | Tests calibration export, input validation, log selection, and priors end to end (log mocked). |
| [tests/test_flight_reference.py](../tests/test_flight_reference.py) | Tests coordinates, time conversion, and flight-log comparison. |
| [tests/test_automatic_flight_reference.py](../tests/test_automatic_flight_reference.py) | Tests the automatic comparison after triangulation and its failure handling. |
| [tests/test_calibration_clear.py](../tests/test_calibration_clear.py) | Tests that clearing calibration deletes only calibration data. |
| [tests/test_camera_repair.py](../tests/test_camera_repair.py) | Tests clearing and re-detecting a single camera. |
| [tests/test_tracking_output_clear.py](../tests/test_tracking_output_clear.py) | Tests clearing one tracking run's outputs. |
| [tests/test_tracking_names.py](../tests/test_tracking_names.py) | Tests compatibility between old and new tracking folder names. |
| [tests/test_tracking_classification.py](../tests/test_tracking_classification.py) | Tests classification persistence and staleness with a mocked model. |
| [vendor/\_\_init\_\_.py](../vendor/__init__.py) | Makes the vendor asset folders importable. |
| [vendor/sam3_assets/\_\_init\_\_.py](../vendor/sam3_assets/__init__.py) | Marks the folder where the (currently missing) SAM3 vocabulary belongs. |
| [vendor/moonshot_classifier_assets/\_\_init\_\_.py](../vendor/moonshot_classifier_assets/__init__.py) | Marks the classifier-manifest folder. |
| [vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json](../vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json) | Pins the classifier model's repository, file, revision, and checksum. |

## What was verified

- **Tests ran.** On Windows (Python 3.14) **53 of 61** passed; the 8 failures come from Windows symlink permissions, Windows path separators, or an optional package that isn't installed. On Linux (WSL) **60 of 61** passed; only the optional-classifier test failed. Every calibration and geometry test passed on both.
- **A synthetic end-to-end run** used the real code for calibration → reconstruction → smoothing → log comparison → a live local server. Only SAM3 and log-file parsing were replaced with known data. §3 of the detailed report has the numbers.
- **Not verified**: SAM3 detection (needs a GPU and gated model access), real recordings and flight logs, Docker, in-browser behavior, and classifier inference.

## Most important findings

Details, evidence, and line references are in [section 5 of the detailed report](fireRepo_detailed_analysis.md#5-findings-and-open-questions).

1. **Calibration can be confidently wrong.** On synthetic data, the default log smoothing (`--smoothing-m 0.1`) shifted cameras by ~1 m and focal lengths by ~10% while every acceptance check passed. A flight 10 m higher then came out ~2 m off. The Calibrate button always uses this default. Validate calibrations against an independent measurement.
2. **Use Linux, WSL, or Docker, not native Windows.** On Windows the dashboard sees one flight log as two and refuses to calibrate (reproduced). The No-Mocap results panel also filters out every reconstruction; that follows from the code and the API output but was not observed in a browser.
3. **Building the Docker image fails**, because a SAM3 vocabulary file was deleted from the repository. It can be restored from Git history (commit `66c31bc`).
4. **"Clear dataset" in the Mocap tabs also deletes No-Mocap detections and reconstructions.**
5. **Tracking clips need their own `metadata.json` start time and the same camera-folder names as calibration.** Otherwise cameras are silently mis-timed (0 frames reconstructed in a test) or dropped.
6. **Put every flight log in the calibration ZIP.** Logs inside tracking ZIPs are discarded, and a dataset ZIP must contain only one run.
7. **Fixing the clock lag (`--offset-search-radius-s 0`) always fails to save.**
8. **In the results editor, save before switching cameras**, and re-run Triangulate after edits; results don't update themselves.
9. **The optional classifier needs Python 3.10–3.12**, and locally the same `FIRETRACK_CLASSIFIER_CACHE` must be set for its prep script and the server.

## Where to start

Read `cli.py`, then `_stage_fn` and `_handle_run` in `webui.py` to see how buttons become function calls. Next read `detect.py`, `triangulate_uploads.py`, and the shared geometry in `triangulate_527.py` (lines 514–582). Leave the calibration mathematics (`calibrate_logs.py`, `joint_calibration_solver.py`) for last. The [recommended reading order](fireRepo_detailed_analysis.md#6-recommended-reading-order) explains each step, and [section 4](fireRepo_detailed_analysis.md#4-how-to-run-and-modify-the-repository) shows the commands that were verified to work.

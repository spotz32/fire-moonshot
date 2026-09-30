# fireRepo: detailed technical analysis

FireTrack reconstructs a drone's 3D flight path from videos recorded by two or more stationary cameras (typically phones). It finds the drone in every frame with a segmentation model (SAM3), works out where each camera is and how it is oriented, lines the cameras up in time, intersects their viewing rays to get 3D positions, smooths the result, and shows everything in a browser dashboard. Optional extras compare the reconstruction with the drone's own ArduPilot flight log and classify the flight as ArduPilot or PX4 firmware with an external model. The Python package is named **firetrack**; the repository folder is **fireRepo**. Despite the name, nothing in the code detects fire. "Fire" appears to be a project name: the Git remote is a repository called `fire-moonshot` and the classifier is `fire-moonshot-classifier`. That reading of the name is an inference.

Companion overview: [fireRepo_high_level.md](fireRepo_high_level.md).

**Snapshot.** The analysis date is **2026-09-29**, on branch `main`, HEAD **769561474c793ae7a4a7813bc520a9241e6224a6** ("Added my analysis", 2026-09-28). That commit only added the two analysis reports. Application code last changed in **1bed945** (2026-09-09, "Add no-mocap calibration and trajectory classification"). In the working tree, only these two reports differ from HEAD.

**Evidence labels used throughout**

| Label | Meaning |
|---|---|
| **Ran** | Executed during this analysis: the test suite, synthetic pipeline scripts, a live local server. |
| **Reproduced** | A specific defect was triggered by running the repository's own code. |
| **Source-confirmed** | Follows directly from reading the code; not executed. |
| **Inferred** | A likely consequence or interpretation that was not proven. |
| **Unverified** | Not checked, because it needs a GPU, real recordings, Docker, a browser, or similar. |

**Existing analysis checked first.** The `analysis/` folder held exactly two documents. `fireRepo_high_level.md` was read first and `fireRepo_detailed_analysis.md` second; both had uncommitted edits relative to HEAD, and the edited on-disk versions were used. The repository contains no other analysis notes, `AGENTS.md`, or `CLAUDE.md`. Every claim was re-checked against the source, and most held. The earlier review could not run anything: NumPy and OpenCV were missing and Git was unavailable. This revision installed the lightweight dependencies in throwaway environments outside the repository. It ran all 61 tests on Windows and on Linux, ran a synthetic end-to-end pipeline, and inspected Git history. [Section 5.6](#56-significant-corrections-to-the-previous-analysis) lists what changed.

**Glossary.** The terms below are used throughout.

| Term | Meaning here |
|---|---|
| SAM3 | Meta's "Segment Anything Model 3", an external neural network that outlines (*segments*) a prompted object in every video frame. |
| Mask / centroid | A mask is the set of pixels SAM3 assigns to the drone; the centroid is the mean x/y pixel position of that mask. The centroid is FireTrack's 2D "detection". |
| Intrinsics (K, dist) | How a camera maps directions to pixels. `K` holds focal lengths fx/fy and principal point cx/cy; `dist` holds OpenCV lens-distortion coefficients. |
| Extrinsics (R, t, position) | Where a camera is and how it is rotated. FireTrack stores **world-to-camera**: `camera_point = R · world_point + t`, and `position = −Rᵀ·t`. |
| ENU / AMSL | East-North-Up local axes around a latitude/longitude origin. FireTrack's "Up" is absolute altitude **Above Mean Sea Level** rather than height above the origin. |
| Epoch time | Seconds since 1970-01-01 UTC. `metadata.json` stores the clip start as `startTime` in microseconds. |
| Triangulation / DLT | Recovering a 3D point from 2+ calibrated views. DLT (Direct Linear Transform) solves a small linear system with SVD. |
| Reprojection error | Pixel distance between an observed detection and where the reconstructed 3D point projects back into that image. |
| PnP / RANSAC | Perspective-n-Point estimates a camera pose from known 3D↔2D pairs. RANSAC repeats the fit on random subsets to resist bad pairs. |
| Spline | A smooth curve through logged positions that also provides velocity. |
| Kalman filter / RTS smoother | A forward "predict then correct" filter, followed by a backward (Rauch–Tung–Striebel) pass that uses future data to refine the past. |
| Huber loss | A least-squares variant that becomes linear for large residuals, so outliers pull less. |
| Jacobian | The matrix of derivatives of residuals with respect to unknowns, used by the optimizer. |
| NPZ | NumPy's zip archive of named arrays (`.npz`). |

---

## 1. Repository Structure and File Breakdown

### 1.1 Annotated directory tree

Line counts are from `wc -l`, and every file listed was read in full. `.git/` is collapsed.

~~~text
fireRepo/
├── .dockerignore (45)                 Docker build-context exclusions
├── .gitignore (60)                    Local/generated file exclusions (incl. the SAM3 vocab .gz)
├── Dockerfile (52)                    CUDA/PyTorch image, FFmpeg, classifier prefetch, vocab copy
├── pyproject.toml (35)                Package "docker-fire-tracking", deps, `firetrack` command, pytest config
├── requirements.txt (8)               Same 8 runtime deps, used by Docker
├── README.md (133)                    Docker usage and the older CLI workflow
├── analysis/
│   ├── fireRepo_high_level.md          Concise overview (companion)
│   └── fireRepo_detailed_analysis.md   This report
├── docs/
│   └── flight-log-calibration.md (153) No-Mocap calibration model, timing, options, priors
├── firetrack/                          The application package
│   ├── __init__.py (3)                 Version 0.1.0
│   ├── __main__.py (4)                 `python -m firetrack` → cli.main
│   ├── cli.py (295)                    9 subcommands; thin wrappers over modules
│   ├── webui.py (1609)                 HTTP server, uploads, stage dispatch, clears/repairs
│   ├── dashboard.py (840)              Main page: HTML/CSS/JS in one bytes constant
│   ├── results_page.py (490)           Results page: synced 2D frames + 3D canvas + editor
│   ├── clicks.py (217)                 Click-annotation storage, frame server, annotator page
│   ├── jobs.py (118)                   One-at-a-time background job runner with log capture
│   ├── sources.py (110)                Video discovery per source; legacy name mapping
│   ├── dataset_527.py (105)            "5-27" dataset: phone models, run/camera layout
│   ├── format_527.py (163)             FFmpeg rotate/re-encode of dataset videos
│   ├── detect.py (421)                 SAM3 sessions → per-frame centroids (+ optional masks)
│   ├── flight_reference.py (254)       ArduPilot log reader, ENU frame, reference comparison
│   ├── calibrate_logs.py (473)         No-Mocap calibration orchestration and export
│   ├── joint_calibration_solver.py (614) Active multi-camera pose/focal/lag solver (CasADi+SciPy)
│   ├── calibration_solver.py (301)     Spline path, priors record, older single-camera solver
│   ├── calibration_priors.py (85)      Optional phone GPS/gravity/magnetometer constraints
│   ├── mocap.py (227)                  Qualisys TSV loaders (3D markers, 6D rigid bodies)
│   ├── calibrate_mocap.py (142)        Mocap run → generic upload-calibration JSON
│   ├── triangulate_527.py (951)        Shared geometry + mocap/dataset reconstruction
│   ├── triangulate_uploads.py (365)    Reconstruction from a supplied calibration JSON
│   ├── ekf.py (205)                    Linear Kalman filter + RTS smoother
│   ├── results.py (266)                Result listing/loading, centroid edits, downloads
│   ├── classify.py (90)                Pinned classifier manifest, download, predictor
│   └── classification.py (151)         Per-run classification state and prediction files
├── scripts/
│   └── prepare_classifier.py (61)      Install pinned classifier wheel; cache its model
├── tests/                              10 unittest modules, 61 test methods (see §4.8)
│   ├── test_automatic_flight_reference.py (77)
│   ├── test_calibration_clear.py (102)
│   ├── test_calibration_integration.py (157)
│   ├── test_calibration_solver.py (121)
│   ├── test_camera_repair.py (147)
│   ├── test_flight_reference.py (155)
│   ├── test_joint_calibration_solver.py (86)
│   ├── test_tracking_classification.py (89)
│   ├── test_tracking_names.py (93)
│   └── test_tracking_output_clear.py (86)
└── vendor/                             Package-data namespaces (no third-party code)
    ├── __init__.py (1)
    ├── sam3_assets/__init__.py (1)     The expected vocab .gz is absent (see F3)
    └── moonshot_classifier_assets/
        ├── __init__.py (1)
        └── fire-moonshot-classifier-model.json (6)  Model repo, file, revision, SHA-256
~~~

The checkout contains no videos, flight logs, mocap files, model weights, build outputs, caches, or virtual environments.

### 1.2 Entry points, core logic, configuration, tests, and utilities

| Role | Files |
|---|---|
| **Entry points** | `firetrack` console script → [cli.py:288](../firetrack/cli.py#L288) (`main`), declared at [pyproject.toml:22-23](../pyproject.toml#L22-L23); `python -m firetrack` via [__main__.py](../firetrack/__main__.py); the web server `serve_webui` at [webui.py:875](../firetrack/webui.py#L875); the standalone annotator `serve_click_ui` at [clicks.py:144](../firetrack/clicks.py#L144). |
| **Core computation** | `detect.py` (2D), `calibrate_logs.py` + `joint_calibration_solver.py` (No-Mocap calibration), `triangulate_527.py` (shared geometry, mocap path), `triangulate_uploads.py` (No-Mocap reconstruction), `ekf.py` (smoothing), `flight_reference.py` (log reading and comparison). |
| **Orchestration and UI** | `webui.py`, `jobs.py`, `dashboard.py`, `results_page.py`, `clicks.py`. |
| **Data adapters and utilities** | `sources.py`, `dataset_527.py`, `format_527.py`, `mocap.py`, `results.py`, `calibration_priors.py`, `calibrate_mocap.py`, `classify.py`, `classification.py`. |
| **Configuration** | `pyproject.toml`, `requirements.txt`, `Dockerfile`, `.dockerignore`, `.gitignore`, `vendor/moonshot_classifier_assets/*.json`, environment variables (§4.7). |
| **Tests** | `tests/` (unittest style; synthetic fixtures, temp directories, mocks). |
| **Docs** | `README.md`, `docs/flight-log-calibration.md`. |

**How the modules import each other** (arrows mean "imports"; dashed arrows are imports that happen inside a function):

~~~mermaid
graph LR
  cli --> webui & detect & format_527 & triangulate_527 & calibrate_logs & calibrate_mocap & clicks
  webui --> clicks & jobs & dashboard & results_page & results & sources & detect & format_527
  webui --> triangulate_527 & triangulate_uploads & calibrate_logs & flight_reference & classification
  sources --> dataset_527 & detect
  clicks --> dataset_527
  detect --> dataset_527
  format_527 --> dataset_527
  triangulate_uploads --> triangulate_527 & ekf & sources
  triangulate_527 --> ekf & mocap & dataset_527
  calibrate_mocap --> mocap & triangulate_527 & triangulate_uploads
  calibrate_logs --> flight_reference
  calibrate_logs -.-> joint_calibration_solver & calibration_solver & calibration_priors
  joint_calibration_solver --> calibration_solver
  calibration_priors --> calibration_solver & flight_reference
  results --> triangulate_527 & sources & flight_reference
  classification --> classify & results
~~~

**Import weight (Ran).** Importing `firetrack.cli` loads NumPy and OpenCV but not SciPy, CasADi, pymavlink, PyTorch, SAM3, or huggingface_hub. `python -m firetrack --help` exited 0 in an environment with no PyTorch or SAM3 installed. The heavy libraries load only when needed: SciPy and CasADi when calibration runs ([calibrate_logs.py:262-264](../firetrack/calibrate_logs.py#L262-L264)), pymavlink when a log is read ([flight_reference.py:52-53](../firetrack/flight_reference.py#L52-L53)), PyTorch and SAM3 when detection runs ([detect.py:118](../firetrack/detect.py#L118), [229](../firetrack/detect.py#L229)), and huggingface_hub when the classifier loads ([classify.py:39](../firetrack/classify.py#L39)).

### 1.3 File-by-file breakdown

**Root configuration and documentation**

| File | Purpose and main contents | Connections |
|---|---|---|
| [README.md](../README.md) | Says FireTrack estimates drone trajectories from 2+ static cameras. Shows `docker run` for the published image `ashreeku/firetrack:latest` (bound to 127.0.0.1:8080), lists 6 CLI commands, a staged format→clicks→detect→triangulate flow, output paths, gated SAM3 access, and `pip install -e .`. | Describes the older dataset workflow. It omits log calibration, tracking runs, and classification (§5.4). |
| [pyproject.toml](../pyproject.toml) | Setuptools project `docker-fire-tracking` 0.1.0 targeting Python ≥3.9. Declares 8 dependencies (casadi, numpy, opencv-python-headless, pymavlink, scipy, torch, `sam3==0.1.2`, huggingface-hub <1.0), the `firetrack = firetrack.cli:main` script, packages `firetrack*`/`vendor*`, package data (`*.gz`, `*.json`), and pytest `testpaths`/`pythonpath`. | pip and Docker. There is no lockfile, and pytest is not a dependency. |
| [requirements.txt](../requirements.txt) | The same 8 runtime dependencies. | Installed by the Dockerfile before `pip install --no-deps .`. |
| [Dockerfile](../Dockerfile) | Builds from `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` and sets `HF_HOME`, `FIRETRACK_CLASSIFIER_CACHE`, `FIRETRACK_FFMPEG`. Installs FFmpeg/git/GL libraries and the package, runs `scripts/prepare_classifier.py` (network), copies the SAM3 BPE vocabulary into SAM3's `assets` folder (lines 40-42), and runs `firetrack webui` on port 8080. | The vocab copy fails in this checkout (F3). |
| [.dockerignore](../.dockerignore) | Keeps `.git`, caches, environments, work/output folders, weights, videos, NPZ, and archives out of the build context. | The Dockerfile copies only specific paths anyway. Unanchored patterns such as `*.mp4` match only at the context root under Docker's rules (not tested). |
| [.gitignore](../.gitignore) | Ignores caches, environments, `.env`, IDE folders, runtime outputs, click/calibration JSONs, weights, `vendor/sam3_assets/*.gz` (line 49), videos, and NPZ. | Line 49 is why the Docker vocab asset is absent. |
| [docs/flight-log-calibration.md](../docs/flight-log-calibration.md) | User guide to No-Mocap calibration: required ZIP contents, `metadata.json`/`camera.json` rules, log reading, solver description, timing convention, CLI defaults, optional phone priors, and the test command. | Mostly accurate; see §5.4 for gaps. |

**Package shell: entry points, server, and browser pages**

| File | Purpose and main contents | Connections |
|---|---|---|
| [firetrack/__init__.py](../firetrack/__init__.py) | `__version__ = "0.1.0"`. | Not used elsewhere. |
| [firetrack/__main__.py](../firetrack/__main__.py) | Calls `cli.main()`. | `python -m firetrack`. |
| [cli.py](../firetrack/cli.py) | `build_parser` ([176-285](../firetrack/cli.py#L176-L285)) defines `format`, `clicks serve/status`, `detect`, `triangulate`, `log-summary`, `calibrate-from-log`, `calibrate-from-mocap`, `run-all`, `webui`. Each `_cmd_*` handler ([31-173](../firetrack/cli.py#L31-L173)) forwards its arguments to one module function and prints a summary. | Imports most modules at load time ([9-15](../firetrack/cli.py#L9-L15)). |
| [webui.py](../firetrack/webui.py) | The application's hub. `WebConfig` derives every on-disk location from `--work-root` ([76-142](../firetrack/webui.py#L76-L142)). Also: upload path mapping ([145-240](../firetrack/webui.py#L145-L240)), tracking-run discovery/selection ([254-318](../firetrack/webui.py#L254-L318)), `build_status` for the dashboard ([335-399](../firetrack/webui.py#L335-L399)), calibration-log selection ([456-482](../firetrack/webui.py#L456-L482)), validated clear/repair targets ([496-597](../firetrack/webui.py#L496-L597)), stage dispatch `_stage_fn` ([640-694](../firetrack/webui.py#L640-L694)), stage wrappers ([697-833](../firetrack/webui.py#L697-L833)), and the nested HTTP `Handler` with every route ([888-1601](../firetrack/webui.py#L888-L1601)). | Calls almost every other module; the dashboard, results, and annotator pages call its routes. |
| [dashboard.py](../firetrack/dashboard.py) | `DASHBOARD_HTML`: one bytes constant with CSS, markup, and JavaScript. Three tabs (No-Mocap default, Mocap Raw, Mocap). Stage cards, upload buttons with progress, run/clear/repair buttons, a console that polls the job log, and an embedded Results iframe. Refreshes `/api/status` every 2.5 s ([838](../firetrack/dashboard.py#L838)). | Served at `/`; uses 20+ routes. |
| [results_page.py](../firetrack/results_page.py) | `RESULTS_HTML`: a camera grid with centroid overlays, a centroid editor, a hand-written 3D canvas projection (no 3D library), and synchronized image-by-image playback. | Served at `/results`; embedded by the dashboard with `?mode=nomocap&embedded=1&run=…`. |
| [clicks.py](../firetrack/clicks.py) | `build_manifest_from_videos` joins saved clicks to current clips ([32-60](../firetrack/clicks.py#L32-L60)). `FrameCache` seeks and JPEG-encodes a frame with OpenCV ([63-85](../firetrack/clicks.py#L63-L85)). `ClicksService` stores clicks and approval ([88-128](../firetrack/clicks.py#L88-L128)). Also holds the annotator page `HTML_PAGE` ([131-141](../firetrack/clicks.py#L131-L141)), the standalone server ([144-207](../firetrack/clicks.py#L144-L207)), and `click_status` counts ([210-217](../firetrack/clicks.py#L210-L217)). | Used by the `clicks` CLI and by `webui._ClicksRegistry`. |
| [jobs.py](../firetrack/jobs.py) | `JobRunner` runs one function at a time in a daemon thread. It captures `print` output into a 5,000-line list and exposes `status()`/`log_since()`. | Every dashboard stage runs through it. |

**Inputs and discovery**

| File | Purpose and main contents | Connections |
|---|---|---|
| [sources.py](../firetrack/sources.py) | Folder-name constants (`uploads`, `tracking_uploads`, legacy `actual_uploads`, `mocap_raw`). `tracking_result_path` rewrites legacy prefixes and backslashes to the public name ([26-31](../firetrack/sources.py#L26-L31)). Discovery functions return `VideoSpec` (label, relative output dir, video path) for each source ([50-110](../firetrack/sources.py#L50-L110)). | Detection output paths come from `VideoSpec.relative_dir`. |
| [dataset_527.py](../firetrack/dataset_527.py) | Hard-coded knowledge of the "5-27" phone dataset. Three phone models map to `pixel9`, `galaxyS8`, `galaxyS21`, each rotated one quarter-turn ([8-18](../firetrack/dataset_527.py#L8-L18)). Raw layout is `root/run/group/camera/video.mp4` and labels are `run_phone` ([48-73](../firetrack/dataset_527.py#L48-L73)); the formatted layout is `root/run/label/video.mp4` ([83-105](../firetrack/dataset_527.py#L83-L105)). | Used by format, detect, clicks, and the mocap reconstruction. |
| [format_527.py](../firetrack/format_527.py) | Rotates and re-encodes dataset videos with FFmpeg (`transpose=2`, libx264 CRF 18, first video stream only; [48-74](../firetrack/format_527.py#L48-L74)), retrying three frame-sync flag variants ([77-96](../firetrack/format_527.py#L77-L96)). Copies sidecar files, probes dimensions, and merges `manifest.json` ([99-163](../firetrack/format_527.py#L99-L163)). | Dataset/"Mocap" tab only. Needs FFmpeg. |

**Detection, calibration, and reconstruction**

| File | Purpose and main contents | Connections |
|---|---|---|
| [detect.py](../firetrack/detect.py) | Runs SAM3 per video, with text-only or click prompts. Converts masks to centroids and writes `centroids.npz` + `summary.json` (+ optional `masks.npz`). Explained in §2.4. | Produces the input for calibration, reconstruction, and results. |
| [flight_reference.py](../firetrack/flight_reference.py) | Reads ArduPilot DataFlash `.BIN` files with pymavlink ([51-89](../firetrack/flight_reference.py#L51-L89)). Converts lat/lon/alt to the ENU+AMSL frame ([92-105](../firetrack/flight_reference.py#L92-L105)). Samples the log at reconstruction times ([108-118](../firetrack/flight_reference.py#L108-L118)), computes comparison metrics, and writes/loads reference sidecars with trajectory hashes ([161-254](../firetrack/flight_reference.py#L161-L254)). | Used by calibration (log reading) and the optional comparison. |
| [calibrate_logs.py](../firetrack/calibrate_logs.py) | File-facing No-Mocap calibration. Loads the log, camera folders (`camera.json`, `metadata.json`), and detections; converts intrinsics into decoded-video pixels; calls the joint solver; writes `uploads_calibration.json` atomically. Explained in §2.6. | Called by the Calibrate button and `calibrate-from-log`. |
| [joint_calibration_solver.py](../firetrack/joint_calibration_solver.py) | The active calibration engine. Jointly fits every camera's rotation, position, and one focal length, plus one shared camera-vs-log clock lag, by intersecting camera rays and matching the log path. Explained in §2.6. | Uses `SplinePath`, `Priors`, `skew`, `left_jacobian` from `calibration_solver.py`. |
| [calibration_solver.py](../firetrack/calibration_solver.py) | `SplinePath` (log path → smooth, differentiable curve; [31-68](../firetrack/calibration_solver.py#L31-L68)) and the `Priors` record ([71-77](../firetrack/calibration_solver.py#L71-L77)), both used by the active solver. Also holds the older single-camera solver `fit_pose_time` ([184-301](../firetrack/calibration_solver.py#L184-L301)), which only its tests call. | Shared helpers; the older solver is legacy. |
| [calibration_priors.py](../firetrack/calibration_priors.py) | Optional per-camera constraints from phone GPS, accelerometer (gravity), and magnetometer, accepted only with `conventions_verified: true` ([42-85](../firetrack/calibration_priors.py#L42-L85)). | Feeds `Priors` into the solver. |
| [mocap.py](../firetrack/mocap.py) | Parses Qualisys TSV exports: header, numeric block, mm→m, 3D marker and 6D rigid-body loaders. Wall-clock parsing assumes UTC−4 ([59-75](../firetrack/mocap.py#L59-L75), [183-227](../firetrack/mocap.py#L183-L227)). | Mocap/dataset reconstruction and mocap export. |
| [calibrate_mocap.py](../firetrack/calibrate_mocap.py) | Uses one mocap run to solve each formatted camera's pose (PnP) and exports it in the generic upload-calibration format, with a per-camera time offset ([56-142](../firetrack/calibrate_mocap.py#L56-L142)). | Bridge from the mocap world to `triangulate_uploads`. |
| [triangulate_527.py](../firetrack/triangulate_527.py) | Two things in one file. First, **shared geometry**: DLT, reprojection, subset selection, centroid interpolation, undistortion, CSV writer ([514-582](../firetrack/triangulate_527.py#L514-L582), [822-836](../firetrack/triangulate_527.py#L822-L836)). Second, the **dataset-specific mocap pipeline**: run/phone constants, PnP calibration and time-offset searches, and per-run reconstruction scored against mocap ground truth. | Shared helpers are reused by `triangulate_uploads.py` and `calibrate_mocap.py`. |
| [triangulate_uploads.py](../firetrack/triangulate_uploads.py) | Validates a calibration JSON, resolves pose conventions, aligns camera timelines from `metadata.json`, triangulates each reference-camera frame, smooths, and writes outputs ([48-365](../firetrack/triangulate_uploads.py#L48-L365)). | The No-Mocap tracking reconstruction. |
| [ekf.py](../firetrack/ekf.py) | Constant-velocity **linear** Kalman filter plus RTS smoother with view-count/reprojection-weighted noise ([15-205](../firetrack/ekf.py#L15-L205)). | Called by both reconstruction paths. |

**Results and classification**

| File | Purpose and main contents | Connections |
|---|---|---|
| [results.py](../firetrack/results.py) | Path-safe access to outputs (`_safe_subdir`), result listings, centroid/trajectory loaders that turn NaN into JSON `null`, manual centroid edits, and download resolution ([23-266](../firetrack/results.py#L23-L266)). | Backs `/api/results`, `/api/centroids`, `/api/trajectory`, and downloads. |
| [classify.py](../firetrack/classify.py) | Validates the pinned model manifest, downloads the model from Hugging Face and checks its SHA-256, and builds the external `TrajectoryPredictor` ([15-90](../firetrack/classify.py#L15-L90)). | Used by `classification.py` and `prepare_classifier.py`. |
| [classification.py](../firetrack/classification.py) | `ClassificationService`: preloads the CPU model if cached, checks trajectory readiness, runs prediction, and stores `prediction.json` with trajectory and model hashes ([41-151](../firetrack/classification.py#L41-L151)). | Classify stage and status. |
| [scripts/prepare_classifier.py](../scripts/prepare_classifier.py) | Downloads the pinned classifier wheel from GitHub, verifies its SHA-256, runs `pip install --no-deps`, then caches the model ([14-57](../scripts/prepare_classifier.py#L14-L57)). | Run by the Dockerfile; optional locally. |
| [vendor/…](../vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json) | Three one-line package markers plus the model manifest: `repo_id`, `filename`, a 40-hex `revision`, and a SHA-256. These are public identifiers, not credentials. | `classify.load_model_manifest`. |

The ten test modules are covered in §4.8.

### 1.4 Data contracts: what goes in and what comes out

**Inputs a user supplies**

| Input | Required content | Consumed by |
|---|---|---|
| Camera folder (No-Mocap) | `video.mp4`, `metadata.json` with `startTime` (UTC microseconds), and `camera.json`. Optional `gps.csv`/`imu.csv` for priors. | `calibrate_logs.load_camera_clips`, `triangulate_uploads._with_upload_metadata` |
| `camera.json` (recommended form) | `{"K": 3×3, "dist": [OpenCV coeffs], "resolution": [width, height]}` describing the **decoded** video pixels. Missing `dist` or `resolution` is rejected ([calibrate_logs.py:75-82](../firetrack/calibrate_logs.py#L75-L82)). | Calibration |
| `camera.json` (legacy Android form) | `sensorPixelArraySize`, `factoryIntrinsicCalibration` [fx, fy, cx, cy, skew], `factoryLensDistortion`, `sensorOrientation`, mapped with crop/rotation assumptions ([83-107](../firetrack/calibrate_logs.py#L83-L107), [381-473](../firetrack/calibrate_logs.py#L381-L473)) | Calibration |
| ArduPilot `.BIN` | GPS messages (instance 0, 3D fix, valid week) and fused **POS** lat/lon/alt messages ([flight_reference.py:58-78](../firetrack/flight_reference.py#L58-L78)) | Calibration; optional comparison |
| Dataset ZIP (Mocap tabs) | One run: camera folders with `video.mp4`, `camera.json` (`device.model` must be a known phone), `metadata.json`, `calibration.json` (`K-matrix`, `resolution`, `distCoeff`), plus `<run>_data_6D.tsv` | `dataset_527`, `triangulate_527` |
| Click rows (`*clicks.json`) | `label`, `click_x`, `click_y` (native pixels), `click_frame`, `approved`, plus probe metadata | `detect.load_clicks` |
| Manual calibration JSON | `{"cameras": [{video, K, rotation (R, euler, quat, or rvec), t or position, optional dist/resolution/start_epoch_s/pose_convention}]}` ([triangulate_uploads.py:48-100](../firetrack/triangulate_uploads.py#L48-L100)) | Tracking reconstruction |

**Artifacts FireTrack writes** (N is the number of frames or samples; missing values are NaN in NPZ, blank in CSV, and `null` in API JSON):

| Artifact | Contents | Producer → consumer |
|---|---|---|
| `detections/<rel>/centroids.npz` | `centroids` (N,2), `frame_indices`, `width`, `height`, `fps`, **absolute** `video_path`, `relative_dir`, `label`, `anchor_frames`, `text_prompt` ([detect.py:179-191](../firetrack/detect.py#L179-L191)) | detect → calibration, reconstruction, results |
| `detections/<rel>/summary.json` | Counts, detection rate, dimensions, paths, prompt/click ([196-214](../firetrack/detect.py#L196-L214)) | detect → status and result listings |
| `detections/<rel>/masks.npz` | Optional per-frame boolean masks | Diagnostic only |
| `uploads_calibration.json` | `world_frame`, `calibration_clock`, `cameras[]`, `diagnostics` (example below) | calibrate_logs, calibrate_mocap, or manual upload → triangulate_uploads |
| `triangulation/<rel>/trajectory.npz` | `times_s`, `epoch_times_s`, `trajectory_raw`/`trajectory_smooth` (N,3), `gt_drone` (NaN without mocap), `n_views`, `reproj_errors_px`, `used_cameras`; the dataset path adds `mocap_times` (keys verified by running) | Reconstruction → viewer, comparison, classifier |
| `triangulation/<rel>/trajectory.csv` | Per frame: time, view count, reprojection error, cameras used, raw/smooth/ground-truth xyz (4 decimals) ([triangulate_527.py:822-836](../firetrack/triangulate_527.py#L822-L836)) | Download |
| `triangulation/<rel>/summary.json` | Counts, reference camera, camera start epochs, skipped cameras, a **calibration snapshot**, median reprojection | Status, comparison, classifier readiness |
| `flight_reference.json/.npz`, `flight_comparison.csv` | Reference metadata, metrics, trajectory SHA-256, sampled log points, per-frame errors | flight_reference → results |
| `classification/tracking_uploads/<run>/prediction.json` | External prediction plus `trajectory_sha256`/`model_sha256` | classification → dashboard/results |
| `triangulation/calibration.json`, `mocap_raw_calibration.json` | Mocap-path camera cache keyed `phone/config` (a different schema from `uploads_calibration.json`) | triangulate_527 |

**Abridged `uploads_calibration.json` written by the synthetic run in §3.1.** Values are real but rounded; `…` marks omitted content. The true cam1 position was (1, −2, 138) and the true focal length 905 (see F1).

~~~json
{
  "world_frame": {"axes": "ENU", "origin_lat_deg": 40.0, "origin_lon_deg": -86.0, "altitude_reference": "AMSL"},
  "calibration_clock": {"camera_minus_log_s": 0.2408, "std_s_local": 0.0068, "applies_to_tracking": false,
                        "convention": "log_time = camera_time - shared_clock_delta_s"},
  "cameras": [{
    "video": "cam1",
    "K": [[1002.9, 0, 540], [0, 1002.9, 960], [0, 0, 1]], "dist": [0, 0, 0, 0, 0], "resolution": [1080, 1920],
    "R": ["…3×3…"], "t": ["…"], "position": [0.729, -2.155, 136.91], "pose_convention": "w2c",
    "start_epoch_s": 1787950003.0,
    "source": {"method": "joint_triangulation_track_ray_shared_lag_common_focal_casadi_scipy",
               "time_offset_s": 0.0, "calibration_clock_delta_s": 0.2408,
               "calibration_lag_applies_to_tracking": false, "window_unix": [1787950002.759, 1787950014.709],
               "spline_smoothing_m": 0.1, "sd_position_m": [0.13, 0.16, 0.58], "sd_focal_px": 47.9, "warnings": ["…"]}
  }],
  "diagnostics": {"shared_clock_delta_s": 0.2408, "results": ["…"],
                  "solver": {"training": {"…": "…"}, "validation": {"frames": 43, "track_rmse_m": 0.206, "ray_median_px": 0.35}}}
}
~~~

### 1.5 The work root: all persistent state

The server has no database; everything lives under `--work-root` (default `/work`). Locations come from `WebConfig` ([webui.py:76-142](../firetrack/webui.py#L76-L142)):

~~~text
<work-root>/
├── dataset_uploads/<run>/<run>/<camera>/…   Raw 5-27 recordings + <run>_data_6D.tsv (Mocap tabs)
├── formatted/<run>/<run>_<phone>/video.mp4  FFmpeg-normalized dataset clips + manifest.json
├── uploads/<camera>/                        Calibration-flight clips and sidecars (+ manifest.json)
├── tracking_uploads/<run>/<camera>/         Named tracking-flight clips (+ manifest.json)
├── calibration_logs/*.BIN                   Logs from the calibration ZIP (also used for comparison)
├── tracking_logs/<run>/*.BIN                Older per-run reference logs (route unused by the UI)
├── detections/<source>/…/<camera>/          centroids.npz, summary.json, optional masks.npz
├── triangulation/<source>/…/                trajectory.npz/.csv, summary.json, reference sidecars
├── classification/tracking_uploads/<run>/prediction.json
├── uploads_calibration.json                 Current No-Mocap calibration
├── clicks.json, mocap_raw_clicks.json       Dataset annotations
├── uploads_clicks.json                      Calibration annotations
├── tracking_uploads_<run>_clicks.json       Tracking annotations
├── webui_state.json                         Selected tracking run
└── .cache/huggingface/hub                   Default local classifier cache (server side)
~~~

Legacy compatibility. Older installations used `actual_uploads` and a flat `tracking_uploads/<camera>` layout exposed as run `default`. `sources.tracking_result_path` and `webui._tracking_root` accept both without moving files ([sources.py:26-36](../firetrack/sources.py#L26-L36), [webui.py:254-318](../firetrack/webui.py#L254-L318)).

---

## 2. In-Depth Code Analysis

### 2.1 The big picture

~~~mermaid
flowchart TD
    V[Camera videos + metadata.json + camera.json] --> C[Click annotation - optional]
    V --> D[SAM3 detection]
    C --> D
    D --> P[2D centroids per camera and frame]
    L[Calibration-flight ArduPilot .BIN] --> S[Spline of logged positions]
    P --> J[Joint calibration: poses, focal, shared lag]
    S --> J
    M[Qualisys mocap or manual poses] --> K[Camera calibration JSON]
    J --> K
    P --> A[Align camera clocks with startTime]
    K --> A
    A --> T[Triangulate every camera subset; pick best]
    T --> E[Kalman filter + RTS smoother]
    E --> O[trajectory.npz / .csv / summary.json]
    O --> R[Browser results + manual centroid edits]
    R -. edits change centroids; rerun needed .-> P
    O --> F[Optional: compare with a flight log]
    O --> X[Optional: ArduPilot vs PX4 classifier]
~~~

In the No-Mocap workflow, **two different flights** matter. The calibration flight is recorded while the drone's log provides known 3D positions, and the solver learns where the cameras are. Tracking flights are later recordings with the cameras left in place; reconstruction uses the saved camera parameters and each clip's own timestamps. A flight log is not needed for tracking.

**Three clocks and three spaces.** Keeping these straight is the key to reading the code:

| Quantity | Convention | Where defined |
|---|---|---|
| Camera frame time | `startTime_µs / 1e6 + frame_index / fps` (constant FPS assumed) | [calibrate_logs.py:281](../firetrack/calibrate_logs.py#L281), [triangulate_uploads.py:304](../firetrack/triangulate_uploads.py#L304) |
| Log time | GPS week/ms → UTC through pymavlink, which hard-codes 18 leap seconds (verified in pymavlink 2.4.50) | [flight_reference.py:70](../firetrack/flight_reference.py#L70) |
| Shared lag δ | `log_time = camera_time − δ`, fitted during calibration only | [joint_calibration_solver.py:493-495](../firetrack/joint_calibration_solver.py#L493-L495) |
| Mocap time | Qualisys wall clock parsed as UTC−4, plus row / frequency | [mocap.py:59-75](../firetrack/mocap.py#L59-L75) |
| World (No-Mocap) | East/North metres from the first log fix; Up = absolute AMSL altitude | [flight_reference.py:92-105](../firetrack/flight_reference.py#L92-L105) |
| World (Mocap) | Qualisys lab frame in metres | [mocap.py:209](../firetrack/mocap.py#L209) |
| Camera axes | x right, y down, z forward (OpenCV) | [docs:129](../docs/flight-log-calibration.md#L129) |

### 2.2 Entry points: CLI, web server, jobs

**CLI commands** ([cli.py:176-285](../firetrack/cli.py#L176-L285))

| Command | Calls | Notes |
|---|---|---|
| `format` | `format_527.normalize_dataset` | Dataset only; needs FFmpeg. |
| `clicks serve` / `clicks status` | `clicks.serve_click_ui` / `click_status` | Formatted dataset clips only. |
| `detect` | `detect.run_detection` | Formatted dataset layout; `--clicks-json` optional (text-only without it). |
| `triangulate` | `triangulate_527.run_triangulation` | **Mocap dataset only**, even when `--calibration-json` is given; that file is the mocap cache format. |
| `log-summary` | `calibrate_logs.summarize_log_and_run` | Prints log/camera UTC overlap. |
| `calibrate-from-log` | `calibrate_logs.calibrate_from_log` | Exposes solver options (§2.6). |
| `calibrate-from-mocap` | `calibrate_mocap.export_mocap_calibration` | Mocap → upload-calibration JSON. |
| `run-all` | format → detect → mocap triangulate | No log calibration or classification. |
| `webui` | `webui.serve_webui` | Default bind `0.0.0.0:8080`, work root `/work`. |

No CLI command runs No-Mocap tracking reconstruction, flight-log comparison, or classification; those are reachable only through the dashboard or by importing the functions. Dataset triangulation catches per-run errors and records them in `summary.json`, and the CLI returns 0 even when some runs failed ([triangulate_527.py:889-905](../firetrack/triangulate_527.py#L889-L905), [cli.py:74-85](../firetrack/cli.py#L74-L85)).

**The web server.** `serve_webui` creates the work folders, a `JobRunner`, a `ClassificationService` (preloading the model if cached), a `_ClicksRegistry`, and a `FrameCache`, then serves a nested `BaseHTTPRequestHandler` on the standard library's `ThreadingHTTPServer`, one thread per request ([webui.py:875-1609](../firetrack/webui.py#L875-L1609)). Uploads are raw request bodies read in 1 MiB chunks, not multipart forms.

| Route (method) | What it does | Called by bundled UI? |
|---|---|---|
| `/`, `/clicks`, `/results` (GET) | Embedded pages | yes |
| `/api/status` (GET) | Everything the dashboard renders: sources, runs, calibration, clicks, classification, environment chips, job | yes (every 2.5 s) |
| `/api/log?since=` (GET) | Job log lines after a cursor | yes (every 1 s while running) |
| `/api/results`, `/api/centroids`, `/api/trajectory` (GET) | Result listings and payloads | yes |
| `/api/result-frame`, `/frame` (GET) | JPEG frames for results and the annotator | yes |
| `/api/trajectory/download` (GET) | CSV / NPZ / comparison CSV | yes |
| `/api/calibration` (GET/POST) | Download / replace calibration JSON | yes |
| `/api/run` (POST) | Validate and start a stage (202; 409 if busy) | yes |
| `/api/nomocap-upload-zip`, `/api/dataset-upload-zip` (POST) | ZIP import | yes |
| `/api/tracking-run/select`, `/clear-outputs`, `…/camera/{clear-outputs,detect}`, `/api/calibration/camera/{clear-outputs,detect}`, `/api/calibration/clear` (POST) | Selection and scoped clears/repairs (job-locked) | yes |
| `/api/dataset/clear`, `/api/centroids/edit`, `/api/clicks/select`, `/api/click`, `/api/skip` (POST) | Direct mutations | yes |
| `/api/upload`, `/api/calibration-log`, `/api/nomocap-format-zip`, `/api/nomocap-format`, `/api/tracking-run/reference-log`, `/api/tracking-run/reference`, `/api/calibration/camera` (exact), `/api/upload/remove`, `/api/upload-frame` | Older or scripted endpoints | **no** (text search of the three pages) |

**Background jobs** ([jobs.py](../firetrack/jobs.py)). `start` refuses a second job, resets state, and runs the function in a daemon thread under `redirect_stdout` ([79](../firetrack/jobs.py#L79)). A failure stores the message and the last 12 traceback lines. `log_since(n)` returns `lines[n:]` with `next=len(lines)`. Notes:

- `redirect_stdout` replaces `sys.stdout` for the **whole process**. While a job runs, prints from other threads land in the job log (**Reproduced**: a print from the test script's main thread appeared in the job log).
- The 5,000-line cap breaks the cursor protocol (F10). Job state is not persisted, and there is no cancel or resume.
- Only stages, scoped clears, and camera repairs use the runner. Uploads, dataset clear, calibration POSTs, and centroid edits do not (F4, §2.13).

**Stage dispatch** ([webui.py:640-694](../firetrack/webui.py#L640-L694)). The dashboard sends `{stage, source}`; `_handle_run` checks the stage against the allowed set for the source ([1123-1159](../firetrack/webui.py#L1123-L1159)):

| Source (tab) | Stages → functions |
|---|---|
| `upload` (No-Mocap calibration) | detect → `run_detection_on_specs(uploads)`; calibrate → `_best_calibration_log_path` + `calibrate_from_log` **with all defaults** ([761-774](../firetrack/webui.py#L761-L774)); triangulate → `triangulate_uploads(uploads)` |
| `tracking` (No-Mocap tracking run) | detect → per-run specs; triangulate → `triangulate_uploads(run)`, then the automatic reference attempt, then classification cleared ([654-660](../firetrack/webui.py#L654-L660), [807-819](../firetrack/webui.py#L807-L819)); classify → `ClassificationService.run` |
| `mocap_raw` | detect (raw videos); triangulate → `run_raw_triangulation` |
| `dataset` ("Mocap") | format; detect; triangulate → `run_triangulation`; run-all |

### 2.3 Getting videos in

- **No-Mocap ZIPs** (`/api/nomocap-upload-zip`, [webui.py:1306-1360](../firetrack/webui.py#L1306-L1360)). The body is staged to a temporary ZIP inside the work root. Every member named `video.mp4`, `metadata.json`, `camera.json`, `gps.csv`, `imu.csv`, or `rf_data.jsonl` is stored as `<its parent folder name>/<file>` (`nomocap_store_rel`, [202-213](../firetrack/webui.py#L202-L213)), after a path-containment check. So a ZIP must hold **one flight with unique camera-folder names**. `.BIN` files are kept **only for the calibration bucket** ([1336](../firetrack/webui.py#L1336)); a log inside a tracking ZIP is silently discarded (F11, Reproduced). Tracking runs are named from the ZIP file name. `normalize_nomocap_videos` only probes and indexes; it does not rotate or re-encode ([216-240](../firetrack/webui.py#L216-L240)). `/api/nomocap-format-zip` is a near-copy of this handler that the UI doesn't call.
- **Dataset ZIPs** (`/api/dataset-upload-zip`, [1210-1248](../firetrack/webui.py#L1210-L1248)). `infer_dataset_run` takes the run name from the **first** `*_data_6D.tsv` in the archive ([187-199](../firetrack/webui.py#L187-L199)). `dataset_store_rel` then maps every member under that run ([159-175](../firetrack/webui.py#L159-L175)). A ZIP with two runs is merged into the first one, and the second run's `camera1/video.mp4` overwrites the first's (F12, Reproduced).
- **Extraction** overwrites existing files, is not transactional, and does not run under the job lock.
- **Discovery** (`sources.py`) turns folders into `VideoSpec(label, relative_dir, video_path)`. `relative_dir` decides where detections are written, e.g. `tracking_uploads/run1/cam1`.
- **Dataset formatting** (`format_527.py`) runs FFmpeg with the first video stream, the rotation filter, H.264 CRF 18, `yuv420p`, and fast-start. It tries `-fps_mode passthrough`, then `-vsync 0`, then the default. It warns, without failing, if the frame count changes, and copies sidecars unchanged, so `calibration.json` still describes the unrotated sensor. The mocap path then rescales it by the new resolution ([triangulate_527.py:126-136](../firetrack/triangulate_527.py#L126-L136)). Whether that matches the rotated frames depends on the dataset's conventions (open question 5.5-4).

### 2.4 Annotation and SAM3 detection

**Clicks.** Opening the annotator (`/api/clicks/select`) builds a `ClicksService` for the chosen source and **immediately writes** its JSON, with rows defaulting to `approved: true` and no click ([clicks.py:95-101](../firetrack/clicks.py#L95-L101)). The page converts a browser click from displayed to native pixels ([clicks.py:140](../firetrack/clicks.py#L140), `saveClick`). `_effective_clicks` treats a click file as meaningful only if it holds at least one approved click ([webui.py:321-332](../firetrack/webui.py#L321-L332)). The consequences:

- **No approved clicks anywhere** → text-only detection for every clip, so the README's statement that clicks are required ([README.md:46](../README.md#L46)) is too strict.
- **At least one approved click** → clips *without* a click are skipped ([detect.py:363-366](../firetrack/detect.py#L363-L366)).
- `approved: false` rows are ignored, and rows with no `approved` field count as approved ([detect.py:105](../firetrack/detect.py#L105)).

**Detection** (`run_sam3_on_video`, [detect.py:218-340](../firetrack/detect.py#L218-L340)), in execution order:

| Lines | What happens and why it matters |
|---|---|
| 229 | Imports PyTorch, even when outputs will be reused, so re-running Detect always needs PyTorch installed. |
| 231-243 | Reuses outputs if `centroids.npz` and `summary.json` exist. This checks existence only, not whether the video, click, or prompt changed. |
| 245-255 | Reads frame count, size, and FPS with OpenCV. Validates the click frame, or picks text anchors at 35%/60%/85% of the clip ([77-81](../firetrack/detect.py#L77-L81)). |
| 257-264 | Builds a SAM3 video predictor, passing the vendored BPE vocabulary if it exists ([117-122](../firetrack/detect.py#L117-L122)), and opens a session on the video file. |
| 266-279 | Click mode: adds the text prompt at the click frame, then runs one full propagation whose output is discarded to prime the session. |
| 280-287 | Adds the user's positive point (normalized by width/height) as object id **9999**. |
| 288-295 | Text mode: adds the text prompt at each anchor frame. |
| 297-317 | Propagates both directions. Each frame's mask becomes `[mean x, mean y]` of its pixels; frames outside `[0, n_frames)` or with empty masks stay NaN. Click mode uses only object 9999; **text mode takes the first reported object**, which may not be the same drone in every frame when several objects match ([137-155](../firetrack/detect.py#L137-L155)). |
| 306-319 | Progress callback every 8 frames. |
| 320-325 | Always closes the session and frees CUDA memory. |
| 327-338 | Writes NPZ + JSON. Frames are numbered `0..N−1` with a constant FPS. |

`run_detection_on_specs` ([343-400](../firetrack/detect.py#L343-L400)) builds one predictor per video, or shares one when `FIRETRACK_REUSE_SAM_PREDICTOR=1`, and rewrites CUDA out-of-memory errors into advice. It fails if there are no videos or nothing was detected.

**Assumptions to keep in mind (Inferred / Unverified).**
- A mask centroid is not a fixed point on the airframe; it shifts as the silhouette changes.
- The code assumes SAM3 decodes the same frames, in the same orientation, as OpenCV reports. Phone videos with rotation metadata, or containers with approximate frame counts, could break that without any error; this was not testable here.
- Holding all masks (`--save-masks`) keeps them in RAM until the end.

### 2.5 Flight logs: time and world frame

**Reading the log** (`read_flight_track`, [flight_reference.py:51-89](../firetrack/flight_reference.py#L51-L89)):
1. Iterates GPS and POS messages with pymavlink's `DFReader_binary`.
2. For every GPS message from instance 0 with a 3D fix and a valid week, records `UTC(GPS week, ms) − boot_time`. pymavlink subtracts a fixed 18 leap seconds, which is correct since 2017.
3. Uses the **median** of those offsets as the boot→UTC mapping, and rejects the log if the 5th–95th percentile spread exceeds 0.5 s.
4. Takes positions from fused **POS** Lat/Lng/Alt, not raw GPS. It rejects boot-clock resets and removes duplicate timestamps.

GPS reporting latency is inside that median offset. The fitted calibration lag δ will absorb any constant latency between the log's UTC and the phones' clocks (Inferred; the default initial δ is 0.22 s).

**World frame** (`to_calibration_frame`, [92-105](../firetrack/flight_reference.py#L92-L105)). The function computes ellipsoid-surface ECEF coordinates, subtracts the origin point, and projects the difference onto east/north unit vectors. Altitude is passed through unchanged, so *z = 150 means 150 m above sea level*. This is a local tangent plane: over ~100 m the curvature error is below a millimetre, but the frame is not a full geodetic ENU conversion. `enu_frame` accepts this frame as a dict or a legacy string, and requires all cameras to agree ([25-48](../firetrack/flight_reference.py#L25-L48)).

**Continuous path** (`SplinePath`, [calibration_solver.py:31-68](../firetrack/calibration_solver.py#L31-L68)). The log is split wherever consecutive samples are more than 1 s apart. Each segment with at least 4 samples gets one cubic smoothing spline per axis, with residual budget `s = n · smoothing_m²` ([47](../firetrack/calibration_solver.py#L47)). Queries outside a segment raise an error, so there is no extrapolation. **The smoothing default of 0.1 m is not neutral.** On a curved path the spline uses its whole budget and shrinks the motion. In a check on the synthetic calibration path (4 m and 3 m horizontal oscillations, ±2 m vertical), `smoothing_m=0.1` left 0.1 m RMS residual per axis and scaled the y and z oscillations to 98.5% and 97.8% (Ran). §3.1 and F1 show how that small shrinkage became a ~1 m calibration error.

**Clock convention.** δ is `camera_time − log_time`. It is used to associate calibration frames with log positions and, later, to query a log for optional comparison. It is **never** applied to tracking cameras: `source.time_offset_s` is exported as 0 and `calibration_clock.applies_to_tracking` is false ([calibrate_logs.py:320-324](../firetrack/calibrate_logs.py#L320-L324), [351-356](../firetrack/calibrate_logs.py#L351-L356)). The hidden legacy option `--time-offset-s` uses the opposite sign and is negated ([288-290](../firetrack/calibrate_logs.py#L288-L290)). The design assumes all phones already share a synchronized clock; per-camera offsets and clock drift are not estimated.

### 2.6 No-Mocap calibration

**Orchestration** (`calibrate_from_log`, [calibrate_logs.py:246-368](../firetrack/calibrate_logs.py#L246-L368)):

1. `load_drone_track` → ENU path + frame metadata ([63-70](../firetrack/calibrate_logs.py#L63-L70)); `SplinePath(..., smoothing_m)` ([267](../firetrack/calibrate_logs.py#L267)).
2. `load_camera_clips` finds every immediate subfolder with `video.mp4`, reports **all** missing `camera.json`/`metadata.json` files together ([126-153](../firetrack/calibrate_logs.py#L126-L153)), and attaches detections found only in run-specific paths. It never falls back to another run's detections ([156-195](../firetrack/calibrate_logs.py#L156-L195)).
3. It requires at least 2 cameras, and **every** camera must have detections ([269-275](../firetrack/calibrate_logs.py#L269-L275)).
4. `_image_space_K` converts intrinsics to decoded pixels ([457-473](../firetrack/calibrate_logs.py#L457-L473)). Explicit `K` must match the detection resolution exactly and is never scaled or rotated. Legacy Android intrinsics go through center-crop scaling (`_aspect_fill_K`) and quarter-turn logic (`_sensor_to_decoded_K`, `_rotate_K`), which are assumptions flagged by a warning.
5. Camera epochs are computed as `start + frame_index/fps` and optional priors are loaded ([277-286](../firetrack/calibrate_logs.py#L277-L286)).
6. `fit_joint_calibration(...)` runs; solver errors are re-raised as "Calibration not saved: …" ([293-302](../firetrack/calibrate_logs.py#L293-L302)).
7. It exports each camera with `R`, `t = −R·position`, `position`, `pose_convention: "w2c"`, the calibration clip's `start_epoch_s`, and provenance ([309-343](../firetrack/calibrate_logs.py#L309-L343)).
8. It writes to `<out>.tmp` with `allow_nan=False` and renames over the target ([359-362](../firetrack/calibrate_logs.py#L359-L362)), so a failure leaves the previous calibration intact. The JSON check is also why fixed-lag runs fail (F5).

**The joint solver** ([joint_calibration_solver.py](../firetrack/joint_calibration_solver.py)):

*Observations* (`prepare_joint_observations`, [85-127](../firetrack/joint_calibration_solver.py#L85-L127)). The camera with the most detections supplies the time grid. Other cameras are interpolated onto that grid, with exact frame matches allowed and gaps larger than 1.5 frame periods never bridged ([63-82](../firetrack/joint_calibration_solver.py#L63-L82)). A sample is kept if ≥2 cameras see the drone and the log covers the **whole** lag search window around it. At most `sample_count` (600) samples are kept, and at least 30 are required. The timeline is cut into 10 equal blocks, and blocks 4 and 9 (20%) are held out for validation. Each camera needs ≥12 training and ≥3 held-out observations.

*Unknowns.* Each of C cameras has 3 rotation increments (a local rotation vector applied to the seed), 3 position increments, and `log(f/f0)` with `f0` = mean of the supplied fx and fy. One shared lag δ is added, for **7C + 1** parameters. The fitted camera has `fx = fy = f0·e^a`; principal point, skew, and distortion stay fixed ([232-245](../firetrack/joint_calibration_solver.py#L232-L245), [294-310](../firetrack/joint_calibration_solver.py#L294-L310)).

*Initialization* (`_initial_poses`, [130-177](../firetrack/joint_calibration_solver.py#L130-L177)). Candidate lags are spaced at most `search_step` apart across the window. For each lag and each camera it runs EPnP inside RANSAC on (log position at camera time − δ, pixel) pairs. It needs ≥12 inliers and ≥90% of points in front of the camera, and scores by median reprojection². The best two candidates with distinct lags become seeds ([511-524](../firetrack/joint_calibration_solver.py#L511-L524)).

*The geometry model, line by line* (`_GeometryModel`, [232-291](../firetrack/joint_calibration_solver.py#L232-L291)). The model is written symbolically in CasADi, so exact derivatives come for free:

| Lines | Meaning |
|---|---|
| 237-245 | Symbolic parameter vector → per-camera rotation `exp(ω)·R_seed`, position `p_seed + Δp`, focal `f0·e^a`. |
| 249 | Pixel → unit viewing ray in camera axes. `_bearing` removes lens distortion with 7 fixed-point iterations ([206-217](../firetrack/joint_calibration_solver.py#L206-L217)). |
| 250 | Rotates the ray into world axes: `d = Rᵀ·bearing`. |
| 251 | `P = I − d·dᵀ` measures displacement perpendicular to that ray. |
| 252-254 | Sums `A = Σ wᵢPᵢ`, `b = Σ wᵢPᵢ·positionᵢ` over visible cameras (`wᵢ` = 1 if visible) and solves `A·X = b`. This is the point minimizing the summed squared distance to all rays. |
| 256-261 | Reprojects `X` into every camera (with distortion) and keeps pixel errors and depths. |
| 262-273 | Maps the same function over all samples; invisible pixels are replaced by the principal point and weighted 0. |
| 274-276 | Asks CasADi for the Jacobian of all outputs with respect to the parameters. |

Nearly parallel rays make `A` poorly conditioned. This ray-intersection solver is used **only during calibration**; tracking reconstruction uses DLT (§2.9).

*Objective* (`_residual_and_jacobian`, [313-360](../firetrack/joint_calibration_solver.py#L313-L360)):
- (a) **Track residual**: intersected point − spline position at (camera time − δ), divided by 0.20 m. Its derivative with respect to δ is the spline velocity, which lets the lag be refined continuously.
- (b) **Pixel residual**: reprojection error divided by `pixel_sigma_px`. The CLI's `--huber-px` feeds this value (default 3).
- (c) **Depth hinge**: penalizes points closer than 0.1 m or behind a camera.
- (d) An optional focal prior.
- (e) Optional sensor priors.

*Robust fitting* (`_fit_seed`, [363-449](../firetrack/joint_calibration_solver.py#L363-L449)). SciPy's `least_squares` uses `loss="huber"` with `f_scale=1`, which applies componentwise Huber at "1 sigma" to **every** residual block. The bounds are:
- rotation increments within ±0.7 rad,
- focal within ×¼ to ×4,
- δ inside the search window.

Setting `focal_sigma=0` freezes the focal parameters, and a zero search radius freezes δ ([369-374](../firetrack/joint_calibration_solver.py#L369-L374)). Up to 4 passes follow; after each, the worst camera at every sample whose error exceeds a gate (30 px first, then max(4·robust σ, 8 px)) is dropped, but only if ≥2 cameras remain. With exactly two cameras nothing can be dropped. If the 4th pass changes visibility, the reported cost and Jacobian use the new visibility without a re-solve (a minor inconsistency, Inferred).

*Acceptance checks* ([497-574](../firetrack/joint_calibration_solver.py#L497-L574)) reject:
- stationary or straight-line flights,
- two similarly good fits with lags more than 0.05 s apart or cameras more than 0.5 m apart,
- fewer than 60% of any camera's training or held-out observations within `ransac_px` with positive depth,
- more than 5% of points behind a camera,
- δ at the search boundary,
- focal at its bound,
- a rank-deficient scaled Jacobian (singular-value ratio < 1e-7).

**Two things are not checked.** No threshold is applied to the 3D track error. And the held-out "track RMSE" is measured against the **same smoothed spline** used for fitting ([456](../firetrack/joint_calibration_solver.py#L456)), so it cannot reveal bias the spline introduced (F1).

*Uncertainty* ([575-594](../firetrack/joint_calibration_solver.py#L575-L594)). The covariance is `pinv(JᵀJ) · max(1, cost/dof)`, computed from the raw (non-Huber) Jacobian. That makes it a local, conditional estimate, not an accuracy guarantee. Frozen parameters get NaN standard deviations; focal handles that by writing `None`, but the frozen lag's NaN reaches the JSON export (F5).

**Calibration options** ([cli.py:230-244](../firetrack/cli.py#L230-L244); the dashboard always uses the defaults, [webui.py:767-772](../firetrack/webui.py#L767-L772)):

| Option | Default | Effect |
|---|---:|---|
| `--sample-count` | 600 | Maximum synchronized samples (training + held-out). |
| `--clock-lag-s` | 0.22 | Center and initial value of δ. |
| `--offset-search-radius-s` | 2.0 | δ bounds; **0 freezes δ but the export then fails** (F5). |
| `--offset-search-step-s` | 0.25 | Maximum spacing of initialization lags (≤401 candidates). |
| `--ransac-reproj-px` | 8 | PnP inlier threshold and the 60%-agreement acceptance threshold. |
| `--smoothing-m` | 0.1 | Per-axis spline residual allowance; **0 interpolates the log exactly**. See F1. |
| `--huber-px` | 3 | Pixel residual scale, which with `f_scale=1` is also the per-component Huber transition. It also sets the weight of pixels relative to the 0.2 m track sigma. |
| `--focal-sigma` | inf | `inf` = free common focal; `0` = fixed at the mean of fx and fy; `(0,1)` = relative prior (0.02 = 2%). |

Not exposed on the CLI: `track_sigma_m` (0.20), `rejection_gate_px` (30), `max_iterations` (150).

### 2.7 Optional priors and the older solver

**Priors** ([calibration_priors.py:42-85](../firetrack/calibration_priors.py#L42-L85)) are off unless `camera.json` contains `calibration_priors` with `conventions_verified: true`:

| Prior | Inputs and checks | Constraint |
|---|---|---|
| Phone GPS | `gps.csv` samples in the clip window; `gps_altitude_reference: "AMSL"` required | Median camera position in the log frame; σ defaults to 2 m, and the horizontal σ is at least the median reported accuracy. |
| Gravity | `imu.csv` accelerometer, `accelerometer_convention: "specific_force"`, a proper `device_to_camera` rotation, median magnitude between 8 and 11 m/s² | The camera's view of "up", σ 1.7°. |
| Magnetometer | `imu.csv` magnetometer and an explicit world field direction | Orientation constraint, σ 20°; magnetic north is not assumed to be true north. |

These checks confirm that the declared conventions are consistent; they cannot confirm that the sensor data is accurate.

**Older single-camera solver** (`fit_pose_time`, [calibration_solver.py:184-301](../firetrack/calibration_solver.py#L184-L301)). It fits one camera's pose and lag with fixed intrinsics, using analytic Jacobians, a vector Huber loss (`huber_blocks`), iterative inlier gating, held-out blocks, and ambiguity/rank checks. Only its tests call it. It returns `None` for the uncertainty of a frozen lag, which is the behaviour the joint solver's export needs (F5).

### 2.8 The mocap (dataset) path

This path exists for one specific dataset, "5-27": three Android phones plus a Qualisys motion-capture system that tracks a rigid body named `drone`. The constants live in [triangulate_527.py:17-37](../firetrack/triangulate_527.py#L17-L37):

- **Runs**: `ardu_run1-3`, `px4_points1-3`, `px4_traj1-3`.
- **Calibration runs**: `ardu_run1` calibrates galaxyS8 and pixel9. galaxyS21 has separate "ardu" and "px4" configurations, calibrated from `ardu_run1` and `px4_points1`.
- **Exclusion**: `px4_traj1` excludes galaxyS21.

1. **Loading mocap** ([mocap.py](../firetrack/mocap.py)). Header fields, a numeric block with empty cells as NaN, mm→m, and rows whose first six values are all zero treated as untracked. Relative time is row/frequency. The start time is parsed as local time at a fixed **UTC−4** ([134](../firetrack/mocap.py#L134), [185](../firetrack/mocap.py#L185)), which is right only for US Eastern daylight time. Untracked rows get NaN position and Euler angles but keep zero rotation matrices, although the docstring promises NaN (F22, Reproduced; no current code reads rotations).
2. **Per-camera calibration** (`calibrate`, [343-387](../firetrack/triangulate_527.py#L343-L387)). For each configuration, the formatted session's centroids are paired with drone mocap positions at `video time + offset`, and OpenCV PnP-RANSAC plus refinement solves the pose. The offset is searched over ±5 s at 0.1 s, then ±0.1 s at 0.01 s, minimizing median reprojection ([291-325](../firetrack/triangulate_527.py#L291-L325)). The result is cached as `triangulation/calibration.json`, and later runs reuse the cache without checking whether inputs changed.
3. **Per-run reconstruction** (`triangulate_run`, [664-740](../firetrack/triangulate_527.py#L664-L740)). For each run, every non-excluded phone's pose is reused and **that run's own mocap** is used to re-search its time offset (±1.5 s at 0.02 s, [475-511](../firetrack/triangulate_527.py#L475-L511)). The reported accuracy therefore includes mocap-assisted synchronization. The run then triangulates with the shared helpers, smooths, samples the nearest mocap ground truth, and scores RMSE, mean, median, p90, and max.

**Practical consequences (Source-confirmed).**
- Triangulating *any* run requires the designated calibration runs to be uploaded and detected. `mocap_path` raises if `ardu_run1` is missing, and `px4_points1` is also needed when px4 runs are included.
- Every non-excluded phone of the target run needs detections, because a missing file fails the whole run ([676](../firetrack/triangulate_527.py#L676)).
- `discover_cameras` rejects unknown phone models. The mode is therefore not a general "bring your own mocap" workflow.

`calibrate_mocap.export_mocap_calibration` reuses steps 1–2 for one chosen run and writes the generic upload schema, with `source.time_offset_s` set to the per-camera offset. `triangulate_uploads` adds that offset to later clips' start times, unlike the log path where the offset is 0.

### 2.9 Reconstruction from calibrated cameras

**Calibration schema and pose conventions** ([triangulate_uploads.py:48-169](../firetrack/triangulate_uploads.py#L48-L169)):

| Field | Accepted forms |
|---|---|
| rotation | `R` (3×3), else `euler` {pitch, yaw, roll} (degrees by default, order default `ZYX` = Rz·Ry·Rx), else `quat` [x,y,z,w], else `rvec` (Rodrigues) |
| translation | `t` → the rotation **is** world→camera; or `position` → the rotation is camera→world (default `pose_convention: "c2w"`) and is transposed, unless `"w2c"` is given |
| optional | `dist` (≥4 values), `resolution` [w,h], `start_epoch_s`, `source` |

Validation checks structure only. `rotation_issue` (orthonormal, det = +1) is enforced only by the unused single-camera upsert route ([webui.py:600-610](../firetrack/webui.py#L600-L610)). A mirrored or non-orthonormal `R` in an uploaded file is accepted.

**Loading cameras and time** ([181-259](../firetrack/triangulate_uploads.py#L181-L259)):
1. For every calibration entry whose `video` **name matches a detected folder**, load `centroids.npz`. The match is by name only, so a renamed folder is skipped and swapped names swap poses (F7).
2. `K` is rescaled if the calibration `resolution` differs from the detection size, **independently in x and y and without a warning** (F20). Calibration, by contrast, rejects any mismatch.
3. `P = K·[R | t]`.
4. The start time comes from the **tracking clip's** `metadata.json` `startTime` (+ `source.time_offset_s`). If that file or field is missing, the calibration file's `start_epoch_s` is used, which for log calibrations is the calibration flight's start, or 0 (F6).

**Timeline.** The camera with the **fewest centroid rows** is the reference ([302-306](../firetrack/triangulate_uploads.py#L302-L306)). Output rows are that camera's frames, and `times_s` is measured from the earliest camera start. There is no image-based synchronization, and cameras whose valid timelines never overlap produce an all-NaN result without error.

**Interpolating the other cameras** (`interpolate_centroid`, [triangulate_527.py:565-576](../firetrack/triangulate_527.py#L565-L576)). `frame_f = (t − start)·fps`, then the code interpolates linearly between `floor(frame_f)` and the next frame, **requiring both to be detected**. Epoch times are ~1.8×10⁹ s, so even the reference camera's own frames are rarely exact: only 200 of 3,000 were exact integers in the check, and 1,400 fell just below (Ran). An isolated detection therefore never contributes, and a detection next to a gap may or may not survive (F14, Reproduced).

**Undistortion** (`undistort_pixel`, [579-582](../firetrack/triangulate_527.py#L579-L582)). `cv2.undistortPoints(..., P=K)` returns undistorted **pixel** coordinates compatible with `P`.

**Geometry used everywhere in reconstruction:**

~~~text
camera_point      = R · world_point + t          (world → camera)
camera_position   = −Rᵀ · t
homogeneous_pixel = K · camera_point  →  pixel = (hx/hz, hy/hz)
P                 = K · [R | t]
~~~

**DLT, line by line** (`triangulate_point_dlt`, [514-524](../firetrack/triangulate_527.py#L514-L524)). Each observed pixel (u, v) in a camera with projection matrix P gives two linear equations in the homogeneous point Xh: `(u·P₃ − P₁)·Xh = 0` and `(v·P₃ − P₂)·Xh = 0`.

| Line | Purpose |
|---|---|
| 515-516 | Needs at least two cameras. |
| 517 | Allocates 2 rows per camera × 4 unknowns. |
| 518-521 | Fills the two equations per camera. |
| 522-523 | SVD; the right singular vector of the smallest singular value is the least-squares solution. |
| 524 | Dehomogenizes (divides by W). There is no guard for W ≈ 0 and no coordinate normalization, and it minimizes algebraic rather than pixel error. |

**Choosing among cameras** (`select_best_triangulation`, [539-562](../firetrack/triangulate_527.py#L539-L562)):

| Lines | Behavior |
|---|---|
| 545-548 | Tries **every** subset of ≥2 cameras (2^C − C − 1 subsets). This is fine for 2–4 cameras and exponential beyond. |
| 549-550 | Rejects points behind any camera in the subset. |
| 551-557 | Computes errors for the subset and for all cameras; rejects the subset if any of its own errors exceeds `max_reproj_px` (30). |
| 558-562 | Picks the lowest **median error over all cameras**, then lowest subset mean, then more views. |

Because the ranking uses the median over all cameras, one wrong camera out of three is outvoted. That was Ran: with two consistent cameras and one deliberately wrong one, the chosen subset excluded the wrong camera and returned the exact point.

**Outputs** ([329-365](../firetrack/triangulate_uploads.py#L329-L365)). The trajectory is smoothed, then the NPZ, CSV, and summary are written. The summary records effective camera starts, skipped cameras, and a **calibration snapshot**, so a later comparison uses the geometry that actually produced the result. The mocap path writes the same keys plus `mocap_times` and fills `gt_drone`.

### 2.10 Smoothing (`ekf.py`)

Despite the file name and docstring ("Extended Kalman Filter"), this is a **linear** constant-velocity Kalman filter followed by an RTS smoother, run offline.

- **State** `[x, y, z, vx, vy, vz]`.
- **Transition**: position += velocity·dt ([15-25](../firetrack/ekf.py#L15-L25)).
- **Process noise**: white-noise acceleration, with per-axis terms q·dt³/3, q·dt²/2, q·dt and q = 2 ([28-46](../firetrack/ekf.py#L28-L46)).
- **Measurement noise**: `σ = 0.15 m × 2/max(n_views, 2) × (1 + 0.01 × reprojection_px)` ([54-68](../firetrack/ekf.py#L54-L68)). More views and smaller errors mean more trust; ray geometry is ignored.

| Lines | Behavior |
|---|---|
| 113-126 | Starts at the first detected point with zero velocity. Leading gaps stay NaN. |
| 142-154 | Predicts; replaces a non-positive dt with 1/30 s. |
| 156-164 | Missing observation: keeps predicting for up to **30 frames**, then resets until the next detection. |
| 166-179 | Standard Kalman update (innovation, S, gain, corrected state/covariance). |
| 185-198 | Backward RTS pass where neighboring states exist. |
| 201-205 | Returns positions only. |

Effect (Ran). A track that stops after 40 frames gets 30 extra constant-velocity smoothed points, 70 finite rows from 40 detections. In the synthetic tracking run, 17 smoothed rows had no triangulation behind them. `n_smooth_finite` is therefore not a measure of detection coverage.

### 2.11 Results API, viewer, and manual correction

- **Listings** ([results.py:86-124](../firetrack/results.py#L86-L124)). Detections come from every `summary.json` next to a `centroids.npz`, and trajectories from every `trajectory.npz`.
  - Detection `dir` strings are built with forward slashes. Trajectory `dir` strings go back through `Path`, so they contain **backslashes on Windows** (F13, Reproduced through the live API: `tracking_uploads\\demo_flight`).
  - In the flat legacy `default` layout, detection rows get the **camera name** as their run (F21, Reproduced).
- **Payloads.** `load_centroids` also reads `metadata.json` next to the **absolute video path stored in the NPZ**; moving the work root breaks that link (Source-confirmed). `load_trajectory` decimates to ≤3,000 points for plotting, adds the flight reference for tracking runs, and adds classification in `webui._load_trajectory`.
- **Edits** (`apply_centroid_edits`, [164-200](../firetrack/results.py#L164-L200)). The function sets or clears frames, rewrites `centroids.npz`, and updates the counts in `summary.json`. Coordinates are not bounds-checked, and triangulation is neither rerun nor invalidated; the page just says "re-run Triangulate". The module docstring still says "Read-only access" ([1](../firetrack/results.py#L1)).
- **Viewer** (`results_page.py`):
  - **2D grid.** Draws the camera grid with overlays.
  - **3D view.** A hand-made projection rotates around **world y** and puts y on the vertical screen axis ([264-271](../firetrack/results_page.py#L264-L271)). ENU data (z = altitude) therefore appears with north roughly "up" on screen, and altitude mixed into the view.
  - **Playback.** Finds each camera's start (the saved start from the summary, else metadata + offset), loads every camera's frame for a time before drawing, and finds the nearest 3D sample by binary search ([338-425](../firetrack/results_page.py#L338-L425)). Mocap summaries key `camera_time_offsets_s` by **phone**, but the page looks them up by the **camera label** ([345-352](../firetrack/results_page.py#L345-L352) vs [triangulate_527.py:731](../firetrack/triangulate_527.py#L731)). In the Mocap tabs, 2D frames and the 3D marker can therefore be offset by the per-camera sync offset (F25, Inferred; not browser-tested).
  - **No-Mocap filtering.** Items are filtered by the prefix `tracking_uploads/` or `uploads/` ([471-476](../firetrack/results_page.py#L471-L476)), which is where F13 bites.
  - **Editing.** The edit map is keyed by frame only and belongs to the active camera. Each camera pane has its own click handler that switches the active camera with `keepEdits=false` ([164](../firetrack/results_page.py#L164), [176-182](../firetrack/results_page.py#L176-L182)), and it fires *before* the grid-level edit handler ([227-235](../firetrack/results_page.py#L227-L235)). Clicking another camera while editing therefore **drops the previous camera's unsaved edits**. Those edited points **stay drawn**, because the in-memory centroid arrays were already changed ([212](../firetrack/results_page.py#L212), [234](../firetrack/results_page.py#L234)), until the page reloads (F8, Source-confirmed; not browser-run).

### 2.12 Flight-log comparison and classification

**Automatic comparison after tracking triangulation** (`_tracking_reference`, [webui.py:777-804](../firetrack/webui.py#L777-L804)). The function deletes old sidecars. It then uses **all** `.BIN` files in `calibration_logs/`; only if there are none does it fall back to `tracking_logs/<run>/`. Any failure becomes `{"error": …}` in `flight_reference.json` without failing the reconstruction.

`attach_reference` ([flight_reference.py:161-235](../firetrack/flight_reference.py#L161-L235)) does the comparison:
1. Hashes `trajectory.npz`.
2. Takes the frame and δ from the summary's calibration snapshot, or from the current calibration.
3. Queries each log at `epoch − δ` with interpolation, never across gaps over 1 s.
4. Requires **exactly one** overlapping log.
5. Computes metrics on the **smoothed** trajectory (plus `raw_metrics`).
6. Warns if the tracking window overlaps the calibration window.
7. Re-checks the hash and writes the NPZ/JSON atomically.

`load_reference` later rejects the reference if the trajectory hash or timestamps changed ([238-254](../firetrack/flight_reference.py#L238-L254)).

Practical rule: **put every relevant `.BIN` (calibration and tracking flights) in the calibration ZIP**. Logs in tracking ZIPs are discarded (F11), and a calibration-only log will not overlap a later flight.

**Classification.**
- `classify.load_model_manifest` validates the pinned manifest, and `_download_model` fetches the file from Hugging Face and checks its SHA-256 ([classify.py:15-59](../firetrack/classify.py#L15-L59)).
- `ClassificationService.preload` loads the CPU predictor with `local_files_only=True` at startup ([classification.py:54-69](../firetrack/classification.py#L54-L69)). A missing package or model only disables the button. The error text suggests "start the app once while online", but preload never downloads ([classify.py:52-53](../firetrack/classify.py#L52-L53)).
- `status` requires a successful triangulation with ≥1 raw point and ≥10 finite smoothed samples with distinct times. It only shows a saved prediction whose trajectory and model hashes still match ([92-125](../firetrack/classification.py#L92-L125)).
- `run` calls `predict_trajectory(trajectory.npz, measurement_type="vision")` and saves the result atomically.
- The dashboard labels the result "ArduPilot or PX4" ([dashboard.py:419](../firetrack/dashboard.py#L419)).

External package (not part of this repository; checked only lightly):
- The pinned wheel (88,990 bytes) and model `cnn_diversify_20260831_1429.inference.pt` (590,554 bytes) still download, and both SHA-256 values match the pins (Ran).
- The wheel's metadata requires **Python ≥3.10, <3.13**. `prepare_classifier.py` only relaxes this below 3.10, so its `pip install` is **refused on Python 3.13+** (F17, Reproduced as a dry run on 3.14.7).
- The predictor's signature is `predict_trajectory(csv_path, *, measurement_type="vision")`, and FireTrack passes the NPZ path. The model's accuracy, windowing, and training data were not reviewed.

### 2.13 Cross-cutting concerns

- **Configuration.** CLI flags plus environment variables (§4.7). No config file exists.
- **Error handling.**
  - Stage failures surface through the job status: a message plus the last 12 traceback lines.
  - HTTP handlers catch the expected `ValueError`, `FileNotFoundError`, and `KeyError` cases and return 400 or 404. There is **no central exception boundary**, though. Malformed JSON bodies ([webui.py:902-904](../firetrack/webui.py#L902-L904)), a non-integer `since` ([946](../firetrack/webui.py#L946)), wrong value types, or a malformed clicks file (F18) raise inside the handler. The connection is then closed without a response.
  - The dashboard swallows failed status fetches ([dashboard.py:626](../firetrack/dashboard.py#L626)), so the page just stops updating.
- **External services.**
  - Hugging Face: the gated `facebook/sam3` weights, fetched by SAM3 at runtime, and the classifier model.
  - GitHub: the classifier wheel.
  - Google Fonts, requested by the browser.
  - FFmpeg, for dataset formatting only.
- **Files and atomicity.**
  - **Atomic** (temp file + rename): calibration JSON, flight-reference JSON/NPZ, `prediction.json`, and the uploaded reference log.
  - **Direct writes**: detections, trajectories, CSVs, clicks, state, ZIP extraction, and manual calibration upload.
- **Concurrency.** Requests are multi-threaded, but only one job runs at a time. Uploads, dataset clear, calibration POST/upsert, upload removal, and centroid edits are **not** job-locked. The dashboard also keeps those buttons (`data-keep`) enabled while a job runs ([dashboard.py:634](../firetrack/dashboard.py#L634)).
- **Security posture.**
  - There is no authentication, CSRF protection, or TLS.
  - The CLI binds `0.0.0.0` by default ([cli.py:281](../firetrack/cli.py#L281)); the README's Docker example correctly publishes only on 127.0.0.1.
  - Path handling is careful: names are sanitized, targets are resolved and checked for containment, and symlinks are refused before deletes.
  - `apply_centroid_edits` loads NPZ files with `allow_pickle=True`, which is acceptable only because those files are the app's own output.

---

## 3. End-to-End Example

### 3.1 What was actually run: a synthetic No-Mocap flight (Ran)

To trace real behavior without GPU, videos, or logs, a script in the session scratchpad (outside the repository) drove the repository's own functions. SAM3 was replaced by projecting a known path into the cameras and adding noise; the log readers were replaced by mocks returning known positions. Everything else was the unmodified code, including a live server on 127.0.0.1.

**Setup (illustrative numbers, real code).**
- **Cameras.** Two portrait cameras (1080×1920, fx = fy = 905, no distortion) about 11–12 m below a drone flying at ~150 m AMSL, looking up. cam1 is at (1, −2, 138) and cam2 at (4, 2, 139) in ENU metres.
- **Calibration flight.** 20 s of curved motion (±4 m east, ±3 m north, ±2 m up), 240 frames at 20 fps per camera, 0.5 px pixel noise, every 17th frame missing, true lag δ = 0.237 s.
- **Tracking flight.** Taken 300 s later along a different path. cam1 recorded 200 frames at 20 fps; cam2 started 20 ms later at 30 fps. Pixel noise was 0.7 px, cam1 had a 10-frame gap containing one isolated detection, and cam2 had a 6-frame gap.

| Step | Code path | Observed result |
|---|---|---|
| 1. Calibrate (dashboard defaults) | `calibrate_from_log` → `fit_joint_calibration` | Accepted. δ = 0.2408 s (true 0.237). **Camera positions off by 113 cm and 82 cm, mostly along the viewing direction; focal 1003/985 vs true 905.** Held-out track RMSE 0.206 m, ray median 0.35 px, no warning about geometry. The exported local uncertainties did hint at weakness (cam1 σ ≈ 0.13/0.16/0.58 m, σ_focal ≈ 48 px), but the actual errors were about twice those values. |
| 2. Same, lag frozen (`offset_search_radius_s=0`) | same | `ValueError: Out of range float values are not JSON compliant: nan`; the previous file was left untouched (F5, Reproduced). |
| 3. Reconstruct tracking flight | `triangulate_uploads` → `interpolate_centroid` → `undistort_pixel` → `select_best_triangulation` → `ekf_smooth_trajectory` | Reference camera cam1; 182/200 frames triangulated. Raw RMSE vs truth **19.0 cm**, smoothed 19.3 cm, median reprojection 0.84 px. Frame 0 was lost because cam2 had not started; the 10-frame gap cost 11 frames, including the valid isolated frame and one valid gap-edge frame (F14); 199 rows were smoothed. |
| 4. Compare with a log | `attach_reference` (log reader mocked with the true lat/lon/alt) | 200/200 frames covered. RMSE 0.189 m smoothed (0.186 m raw), horizontal 0.131, vertical 0.137. |
| 5. Live server | `serve_webui` on 127.0.0.1 | `GET /` 200 (54,558 bytes). `/api/status` shows the calibration (2 cameras) and run `demo_flight`. `POST /api/run` → **202**; the job finished `done`. The job log contained a print from another thread (process-wide stdout). `/api/results` returned the trajectory `dir` as `tracking_uploads\\demo_flight` (F13). The automatic reference correctly reported "No stored flight logs". The CSV download returned 200. |
| 6. Remove one tracking `metadata.json` | `triangulate_uploads` | cam2 silently took the calibration clip's start time, **0/200 frames triangulated**, and nothing warned (F6, Reproduced). |

**What step 1 means.** Rerunning calibration with different settings isolated the cause as the default spline smoothing, not pixel noise (Ran):

| Variant | Camera position error | Focal (true 905) | Held-out track RMSE | Tracking RMSE, same heights | Tracking RMSE, 10 m higher |
|---|---:|---:|---:|---:|---:|
| Defaults (noise 0.5 px, smoothing 0.1, free focal) | 113 cm | 985–1003 | 0.206 m | 18.9 cm | **202 cm** |
| Noise 0.5 px, **smoothing 0** | 6 cm | 900–902 | 0.029 m | 3.3 cm | 9.0 cm |
| **No noise**, smoothing 0.1 | 122 cm | 990–1011 | 0.203 m | 18.6 cm | 201 cm |
| No noise, smoothing 0 | 0 cm | 905.0 | 0.000 m | 0.0 cm | 0.0 cm |
| Noise 0.5 px, smoothing 0.1, **fixed focal** | 33 cm | 905 | 0.214 m | 19.1 cm | 114 cm |
| Noise 0.5 px, smoothing 0.1, 2% focal prior | 32 cm | 911–917 | 0.213 m | 18.5 cm | 126 cm |

In this geometry, focal length and camera distance can almost trade against each other. The ~2% shrinkage of the smoothed log path was enough to push the solution about 10% along that trade-off. The biased calibration is self-consistent at the calibration heights, which is why the same-height tracking error looked fine and the held-out diagnostics looked excellent, but it extrapolates badly. This is one synthetic configuration. Real ArduPilot POS logs are noisier and the best smoothing value for them is unknown; the point is that the diagnostics cannot tell you.

### 3.2 The same flight with real data through the dashboard (illustrative)

Suppose two phones are fixed on tripods, record a calibration flight and later a tracking flight, and have synchronized clocks.

**Calibration ZIP.** One folder per camera, plus the log(s):

~~~text
calibration_flight.zip
├── cam1/ video.mp4  camera.json  metadata.json
├── cam2/ video.mp4  camera.json  metadata.json
├── calibration_flight.BIN
└── tracking_flight_1.BIN        ← include later flights' logs here too, for comparison
~~~

Example `camera.json` for a 1920×1080 decoded video (illustrative numbers):

~~~json
{"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], "dist": [0, 0, 0, 0, 0], "resolution": [1920, 1080]}
~~~

`metadata.json` needs `{"startTime": <UTC microseconds>}` with the true clip start.

| Step (dashboard, No-Mocap tab) | Code | Files written |
|---|---|---|
| Upload calibration zip | `uploadNoMocapZip` → `/api/nomocap-upload-zip?bucket=calibration` → `_handle_nomocap_upload_zip` → `normalize_nomocap_videos` | `uploads/cam*/…`, `uploads/manifest.json`, `calibration_logs/*.BIN` |
| Annotate (one click per camera) | `/api/clicks/select` → annotator page → `/api/click` → `ClicksService.apply_click` | `uploads_clicks.json` |
| Detect | `/api/run {detect, upload}` → `_upload_detect_fn` → `run_detection_on_specs` → `run_sam3_on_video` | `detections/uploads/cam*/centroids.npz`, `summary.json` |
| Calibrate | `/api/run {calibrate, upload}` → `_best_calibration_log_path` (exactly one overlapping log) → `calibrate_from_log` → `fit_joint_calibration` | `uploads_calibration.json` (auto-downloaded) |
| Upload tracking zip `tracking_flight_1.zip` | `/api/nomocap-upload-zip?bucket=tracking&run=tracking_flight_1` | `tracking_uploads/tracking_flight_1/cam*/…`, `webui_state.json` |
| Annotate + Detect the run | `_tracking_detect_fn` | `tracking_uploads_tracking_flight_1_clicks.json`, `detections/tracking_uploads/tracking_flight_1/cam*/…` |
| Triangulate | `_tracking_triangulate_fn` → `triangulate_uploads` → `_tracking_reference` → `attach_reference`; then `classifier.clear` | `triangulation/tracking_uploads/tracking_flight_1/trajectory.npz/.csv`, `summary.json`, `flight_reference.*`, `flight_comparison.csv` |
| Classify (optional) | `/api/run {classify, tracking}` → `ClassificationService.run` | `classification/tracking_uploads/tracking_flight_1/prediction.json` |
| Review / fix | Results iframe (`/results?mode=nomocap…`); edit centroids → `/api/centroids/edit`; re-run Triangulate | Updated `centroids.npz`/`summary.json` |

The tracking ZIP must reuse the **same camera folder names** as calibration (F7), and each clip needs its own `metadata.json` (F6). On native Windows, step "Calibrate" currently fails (F2).

### 3.3 One time sample, by hand (Ran with the repository functions)

Two identical cameras (the K above), both with identity rotation. Camera 1 is at the origin and camera 2 is 1 m to the east, so its `t = −R·position = (−1, 0, 0)`. The drone is 5 m in front, at (0, 0, 5):

- Camera 1: `K·(0, 0, 5)` → pixel (960, 540).
- Camera 2: camera point (−1, 0, 5) → u = 1000·(−1/5) + 960 = **760**, v = 540.
- DLT on those two pixels returns exactly (0, 0, 5) with 0 px reprojection. Adding a third, deliberately wrong camera still returns (0, 0, 5) using cam1 + cam2, because the all-camera median error is 0 for that pair.

### 3.4 How the clocks line up (illustrative arithmetic)

Say calibration found δ = 0.237 s. A tracking frame stamped T + 60.000 s by the phone stays at T + 60.000 s in `epoch_times_s`: reconstruction never shifts cameras by δ. Only the optional log comparison looks up the log at T + 59.763 s.

---

## 4. How to Run and Modify the Repository

### 4.1 What each stage needs

| Stage | Python packages | Other requirements |
|---|---|---|
| CLI help, server, uploads, annotation, results | numpy, opencv-python-headless | — |
| Calibration (No-Mocap) | + scipy, casadi, pymavlink | Real `.BIN` log, `camera.json`, `metadata.json`, detections |
| Reconstruction, smoothing, comparison | numpy, opencv (+ pymavlink for comparison) | Calibration JSON, detections |
| Detection | + torch, `sam3==0.1.2` | Access to gated `facebook/sam3` on Hugging Face (token or cache); practically an NVIDIA GPU |
| Dataset formatting | — | FFmpeg on PATH or `FIRETRACK_FFMPEG` |
| Classification | + external `fire-moonshot-classifier` (Python 3.10–3.12), torch, huggingface-hub | Cached pinned model |
| Tests | numpy, scipy, opencv, casadi, pymavlink, huggingface-hub | POSIX-like filesystem behaviour for 7 tests (see §4.8) |

### 4.2 Verified light setup (no GPU, no detection)

This is the setup used for this analysis. The venv lived in a scratch folder outside the repository, the test and `--help` lines were run exactly as shown, and the server was started in-process on 127.0.0.1 with a scratch work root. On Windows PowerShell with Python 3.14:

~~~powershell
python3.14 -m venv C:\path\outside\repo\ft-venv
C:\path\outside\repo\ft-venv\Scripts\python.exe -m pip install numpy scipy opencv-python-headless casadi pymavlink "huggingface-hub>=0.30,<1.0"
cd C:\Users\gmato\Desktop\Research\fireRepo
$env:PYTHONDONTWRITEBYTECODE = "1"
C:\path\outside\repo\ft-venv\Scripts\python.exe -B -m unittest discover -s tests -v      # 61 run, 53 pass
C:\path\outside\repo\ft-venv\Scripts\python.exe -B -m firetrack --help                   # works without torch/sam3
C:\path\outside\repo\ft-venv\Scripts\python.exe -B -m firetrack webui --work-root ..\firetrack-work --host 127.0.0.1 --port 8080
~~~

On Linux (WSL Ubuntu 26.04, Python 3.14.4), the same packages give **60/61** passing tests.

Versions used: NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0, CasADi 3.8.1, pymavlink 2.4.50, huggingface_hub 0.36.2. Running from the repository root makes `firetrack` and `vendor` importable without installing the package. `-B` keeps bytecode out of the repository.

### 4.3 Full local setup including detection (Unverified)

`pip install -e .` (README) also installs torch and `sam3==0.1.2`; whether those install on Windows with Python 3.14 was not attempted. Detection additionally needs an authorized Hugging Face login or token for `facebook/sam3`, and realistically a CUDA GPU. `_gpu_visible` checks Linux device paths only, so on native Windows the GPU chip always shows "off" ([webui.py:624-630](../firetrack/webui.py#L624-L630)).

To enable classification locally, use Python 3.10–3.12 and point both the prep script and the server at the same cache:

~~~bash
export FIRETRACK_CLASSIFIER_CACHE=/abs/path/classifier-cache   # same value for both commands
python scripts/prepare_classifier.py
python -m firetrack webui --work-root ../firetrack-work --host 127.0.0.1
~~~

Without that variable, the script caches under `<repo>/../firetrack-work/.cache/…`, or `<repo>/firetrack-work/.cache/…` if that sibling folder doesn't exist yet. The server looks under `<work-root>/.cache/…`. The two agree only by coincidence ([prepare_classifier.py:45-51](../scripts/prepare_classifier.py#L45-L51), [classification.py:47-50](../firetrack/classification.py#L47-L50)).

### 4.4 Docker

The published-image command from the README ([README.md:17-29](../README.md#L17-L29)) binds to localhost and mounts a Hugging Face cache and a work folder. Neither the published image nor its contents were checked.

~~~bash
docker run --rm -it --gpus all -p 127.0.0.1:8080:8080 -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf -v "$PWD/firetrack-work:/work" \
  ashreeku/firetrack:latest webui --work-root /work --host 0.0.0.0 --port 8080
~~~

**Building this checkout** (`docker build -t firetrack-local .`) will stop at [Dockerfile:40-42](../Dockerfile#L40-L42), because `vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz` is absent (F3). Git history shows the file was committed in `66c31bc` (1,356,917 bytes, SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`, first line `bpe_simple_vocab_16e6.txt#version: 0.2`). It was deliberately deleted in `3a0a659` "Ignore bundled SAM asset", and `.gitignore:49` ignores it. It can be restored locally from Git Bash or another POSIX shell:

~~~bash
git cat-file blob 66c31bc:vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz > vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz
sha256sum vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz   # expect 924691ac…6804a
~~~

Avoid Windows PowerShell 5.1 `>` for this, because it re-encodes binary output. Nothing was restored during this analysis. The build also downloads the classifier wheel and model; both pins resolve today. The base image's Python must be 3.10–3.12 for the wheel, which was not checked. No Docker build was run.

### 4.5 Command-line workflows

Mocap dataset staging:

~~~bash
firetrack format   --data-root /data/raw --out-root /work/formatted
firetrack clicks serve --data-root /work/formatted --clicks-json /work/clicks.json --host 127.0.0.1 --port 8080
firetrack detect   --data-root /work/formatted --out-root /work/detections --clicks-json /work/clicks.json
firetrack triangulate --raw-root /data/raw --formatted-root /work/formatted --detections-root /work/detections --out-root /work/triangulation
~~~

No-Mocap calibration from existing folders and detections (logs can be inspected first):

~~~bash
firetrack log-summary --log /work/calibration_logs/cal.BIN --run-root /work/uploads
firetrack calibrate-from-log --log /work/calibration_logs/cal.BIN --run-root /work/uploads \
  --detections-root /work/detections --out-json /work/uploads_calibration.json --smoothing-m 0
~~~

`--smoothing-m 0` is shown because of F1; choose it deliberately for your logs. Mocap → upload-calibration export:

~~~bash
firetrack calibrate-from-mocap --raw-root /data/raw --formatted-root /work/formatted \
  --detections-root /work/detections --run ardu_run1 --out-json /work/uploads_calibration.json --label-map camera1=cam1
~~~

### 4.6 Dashboard workflow (No-Mocap)

1. Upload the calibration ZIP (cameras + all relevant `.BIN` logs).
2. Annotate each camera, or skip clicks for text-only detection.
3. Detect.
4. Calibrate; the JSON downloads automatically.
5. Upload each tracking ZIP; the run is named after the file.
6. Annotate and Detect the run.
7. Triangulate.
8. Optionally Classify.
9. Inspect the Results panel, edit centroids if needed, and re-run Triangulate.

"Reprocess camera" clears and re-detects one camera without touching the others. "Clear outputs" and "Clear calibration" are scoped and job-locked. **"Clear dataset" in the Mocap tabs also deletes No-Mocap detections and reconstructions** (F4).

### 4.7 Configuration reference

| Setting | Default | Role |
|---|---|---|
| `--work-root` | `/work` | All persistent state (§1.5). |
| `--host` / `--port` | `0.0.0.0` / 8080 | Bind address. Use 127.0.0.1 unless the network is trusted. |
| `FIRETRACK_FFMPEG` | `ffmpeg` | FFmpeg binary; Docker sets `/usr/bin/ffmpeg`. |
| `FIRETRACK_REUSE_SAM_PREDICTOR=1` | off | Share one SAM3 predictor across videos. |
| `FIRETRACK_CLASSIFIER_CACHE` | `<work-root>/.cache/huggingface/hub` (server) | Classifier model cache; Docker sets `/opt/firetrack/classifier-cache`. |
| `HF_HOME`, `HF_TOKEN` | Hugging Face defaults | SAM3 weights cache and credentials. `HF_HOME` also drives the "weights cached" chip. |
| `HF_HUB_OFFLINE=1` | — | Shown as "offline" on the chip; enforcement is up to huggingface_hub. |

Calibration settings are covered in §2.6. Reconstruction's `max_reproj_px` (30) and the smoother parameters are only function arguments.

### 4.8 Tests: what they check and what happened when run

All tests use `unittest` with synthetic numbers, temporary folders, and mocks. None uses real video, a real `.BIN`, mocap files, SAM3, or a browser.

| Test file | Tests | What it checks | Windows (3.14.7) | Linux (3.14.4) |
|---|---:|---|---|---|
| test_calibration_solver.py | 11 | Older solver: pose + fractional lag recovery, noise/outliers, analytic vs numeric Jacobians, line/circle degeneracy, fixed lag, priors, bad settings, spline gaps/smoothing | 11 pass | 11 pass |
| test_joint_calibration_solver.py | 3 | 3-camera shared lag/pose/focal recovery (noise-free), fixed mean focal, duplicate names, tilted-sensor distortion | 3 pass | 3 pass |
| test_calibration_integration.py | 8 | Export via `calibrate_from_log` (log mocked), preservation of old file, explicit-K rules, missing files, no cross-run fallback, UTC frame, log ambiguity, prior conventions | 8 pass | 8 pass |
| test_flight_reference.py | 10 | ENU axes/altitude, frame validation, gap handling, reconstruction untouched, lag sign, staleness, ambiguous logs, snapshot preference, metrics, GPS→UTC (real pymavlink function) | 10 pass | 10 pass |
| test_automatic_flight_reference.py | 5 | Triangulate→reference ordering, stored vs legacy logs, failure sidecars, no comparison after failed triangulation | 5 pass | 5 pass |
| test_calibration_clear.py | 5 | Calibration-only deletion (repeatable), symlink refusal, annotator cache reset, busy-lock | 3 pass, **2 error** (symlink privilege) | 5 pass |
| test_camera_repair.py | 6 | Per-camera clears/redetect, preserved inputs, legacy/default scope, invalid names/symlinks, busy lock, required clicks | 3 pass, **3 error** (2 backslash paths in the test's comparison, 1 symlink) | 6 pass |
| test_tracking_output_clear.py | 4 | Run-scoped deletion, invalid runs/symlinks, flat-layout isolation, busy lock | 3 pass, **1 error** (symlink) | 4 pass |
| test_tracking_names.py | 5 | Legacy/new run discovery, public names, download names, legacy clear, source aliases | 4 pass, **1 fail** (`'tracking_uploads\\run1' != 'tracking_uploads/run1'`, a real product difference, F13) | 5 pass |
| test_tracking_classification.py | 4 | Prediction persistence, stale hiding, clear, CPU preload | 3 pass, **1 fail** (package not installed) | 3 pass, **1 fail** (same) |
| **Total** | **61** | | **53 pass** | **60 pass** |

Windows runs took ~55–70 s, Linux ~18 s; results were identical across two Windows runs. The one remaining Linux failure calls the real `find_spec("fire_moonshot_classifier")` without mocking it, so it passes only where the external package is installed, which is impossible on 3.13+ (F17).

**Gaps in test coverage:**
- no SAM3/detection test,
- no test of `triangulate_uploads` or the smoother,
- no joint-solver test with noise, smoothing, or two cameras,
- no fixed-lag *export* test (F5),
- no job-log overflow test,
- no upload/ZIP handler tests,
- no browser tests.

### 4.9 Where common changes belong

| Change | Primary place | Also touch |
|---|---|---|
| New CLI option or command | `cli.py` | The module function; README/docs |
| New dashboard stage | `webui.py` (`*_STAGES`, `_stage_fn`, `build_status`) | `dashboard.py` stage cards/buttons; output invalidation |
| New input layout or device | `sources.py`, `webui.nomocap_store_rel`/ZIP handlers | `dataset_527.py` for the mocap dataset |
| Detection prompting/selection | `detect.py` | Click schema in `clicks.py`; repair checks in `webui.py` |
| Calibration model or loss | `joint_calibration_solver.py` | `calibrate_logs.py` export; CLI options; tests |
| Log parsing / world frame | `flight_reference.py` | `calibration_priors.py` (shares the frame) |
| Reconstruction / outlier policy | `triangulate_527.py` shared helpers | Both loops (`triangulate_uploads.py`, `triangulate_run`/`triangulate_raw_run`, which duplicate each other) |
| Motion model / gap policy | `ekf.py` | Summary counts; UI wording |
| Result schema or viewer | Both writers + `write_csv` | `results.py`, `results_page.py`, `flight_reference.py`, classifier, tests |
| Clear/repair scope | `webui.py` target helpers | Preservation tests |
| Classifier | `classify.py`, `classification.py`, `prepare_classifier.py` | Manifest JSON; Dockerfile |

There is no central schema layer; artifact keys are repeated across writers and readers.

### 4.10 Missing prerequisites and what blocked verification

| Missing or blocked | Effect on this analysis |
|---|---|
| NVIDIA GPU, PyTorch, SAM3, gated weights | Detection was never executed. |
| Real videos, `.BIN` logs, mocap TSVs, phone sensor CSVs | No real-data accuracy; the log reader was tested only through mocks and the real pymavlink time function. |
| Docker, network access to the published image | No image build or run. |
| FFmpeg | Dataset formatting not run. |
| A browser/JavaScript runtime | Page behaviour inferred from source; only HTTP responses were checked. |
| Python 3.10–3.12 + PyTorch for the classifier | Classifier inference not run. |

---

## 5. Findings and Open Questions

Severity reflects risk to a new maintainer's results or data. Status uses the labels at the top.

### 5.1 Priority findings

| ID | Finding | Evidence | Status | Suggested direction |
|---|---|---|---|---|
| **F1** | **The default log smoothing can silently bias No-Mocap calibration.** In a synthetic 2-camera test, `smoothing_m=0.1` put cameras ~1 m off and focal ~10% high, while acceptance checks passed and held-out errors looked small; a flight 10 m higher reconstructed ~2 m off. Only the per-camera σ values in the JSON hinted at a problem (≈0.6 m, 48 px), and they were about half the real error. The held-out "track RMSE" uses the same smoothed spline, so it cannot detect this, and the dashboard always uses the default. | [calibration_solver.py:47](../firetrack/calibration_solver.py#L47), [joint_calibration_solver.py:456](../firetrack/joint_calibration_solver.py#L456), [webui.py:767-772](../firetrack/webui.py#L767-L772), [cli.py:241](../firetrack/cli.py#L241); §3.1 | **Reproduced** (synthetic) | Validate against an independent reference (measured camera positions, a flight at another distance); consider default 0; expose settings in the UI; report the spline's own residual. |
| **F2** | **On native Windows, one uploaded log is seen twice**, so dashboard calibration always fails with "Multiple uploaded flight logs…". The status log count is also doubled. | [webui.py:411](../firetrack/webui.py#L411), [459](../firetrack/webui.py#L459), [467](../firetrack/webui.py#L467), [480-481](../firetrack/webui.py#L480-L481) | **Reproduced** (`summarize_log_and_run` mocked) | Single glob + `suffix.lower()`, as [785-786](../firetrack/webui.py#L785-L786) already does. |
| **F3** | **Docker build from this checkout fails**: the SAM3 vocabulary file the Dockerfile copies is absent (deleted in `3a0a659`, ignored by `.gitignore:49`). | [Dockerfile:40-42](../Dockerfile#L40-L42), [.gitignore:49](../.gitignore#L49) | Absence and history **verified**; build not run | Restore from `66c31bc` (§4.4) or document where it comes from. |
| **F4** | **"Clear dataset" (Mocap tabs) deletes No-Mocap outputs too.** It removes the entire `detections/` and `triangulation/` trees without the job lock, although the confirmation only mentions "5-27 files and generated dataset outputs". | [webui.py:1516-1524](../firetrack/webui.py#L1516-L1524), [dashboard.py:778-783](../firetrack/dashboard.py#L778-L783) | **Reproduced** | Scope to dataset/mocap_raw targets; use the job lock. |
| **F5** | **Fixed-lag calibration cannot be saved**: a frozen lag has NaN uncertainty, and the strict JSON export rejects NaN. The docs advertise radius 0. | [joint_calibration_solver.py:373-374](../firetrack/joint_calibration_solver.py#L373-L374), [578-580](../firetrack/joint_calibration_solver.py#L578-L580), [calibrate_logs.py:353](../firetrack/calibrate_logs.py#L353), [361](../firetrack/calibrate_logs.py#L361), [docs:110](../docs/flight-log-calibration.md#L110) | **Reproduced** | Emit `null` for frozen parameters; add an export test. |
| **F6** | **A tracking clip without `metadata.json`/`startTime` silently uses the calibration flight's start time.** In the test this produced 0 triangulated frames and no warning; partial misalignment is also possible. | [triangulate_uploads.py:226](../firetrack/triangulate_uploads.py#L226), [241-259](../firetrack/triangulate_uploads.py#L241-L259), [calibrate_logs.py:338](../firetrack/calibrate_logs.py#L338) | **Reproduced** | Require `startTime` for tracking clips, or at least warn. |
| **F7** | **Tracking cameras are matched to calibration by folder name only.** A renamed folder is skipped (with a console note), and swapped names swap poses silently. | [triangulate_uploads.py:279-293](../firetrack/triangulate_uploads.py#L279-L293) | Source-confirmed | Check labels at upload/detect time and show mismatches in the UI. |
| **F8** | **Switching cameras while editing results discards unsaved edits, which stay drawn on screen until reload.** | [results_page.py:164](../firetrack/results_page.py#L164), [176-182](../firetrack/results_page.py#L176-L182), [212](../firetrack/results_page.py#L212), [227-235](../firetrack/results_page.py#L227-L235) | Source-confirmed (not browser-run) | Keep edits per camera, or prompt; save before switching. |
| **F9** | **Dashboard refresh (every 2.5 s) rebuilds the clip lists and unticks checkboxes**; an empty selection means "all clips". | [dashboard.py:586-591](../firetrack/dashboard.py#L586-L591), [629-630](../firetrack/dashboard.py#L629-L630), [838](../firetrack/dashboard.py#L838); [webui.py:247-251](../firetrack/webui.py#L247-L251) | Source-confirmed | Preserve selections across refreshes. |
| **F10** | **Job-log polling stalls after 5,000 lines** (the cursor equals the capped length). | [jobs.py:58-62](../firetrack/jobs.py#L58-L62), [107-118](../firetrack/jobs.py#L107-L118) | **Reproduced** | Use a monotonic sequence number. |
| **F11** | **`.BIN` logs inside a tracking ZIP are discarded**; only the calibration bucket keeps logs. | [webui.py:1334-1340](../firetrack/webui.py#L1334-L1340) | **Reproduced** | Store them under `tracking_logs/<run>/`, or tell the user. |
| **F12** | **A dataset ZIP with several runs is merged into the first run**, overwriting files. | [webui.py:159-175](../firetrack/webui.py#L159-L175), [187-199](../firetrack/webui.py#L187-L199) | **Reproduced** | Map per member path, or reject multi-run archives. |
| **F13** | **On native Windows, No-Mocap Results never shows reconstructions**: trajectory IDs use backslashes, and the page filters on `tracking_uploads/`. | [results.py:118-120](../firetrack/results.py#L118-L120), [results_page.py:474-475](../firetrack/results_page.py#L474-L475) | API output **Reproduced**; UI effect source-confirmed | Serialize with `as_posix()`. |
| **F14** | **Isolated detections and some gap-edge frames never contribute**, because interpolation needs both bracketing frames and epoch-scale timestamps are rarely exact. | [triangulate_527.py:565-576](../firetrack/triangulate_527.py#L565-L576) | **Reproduced** | Use exact-frame tolerance, as the calibration interpolator does ([joint_calibration_solver.py:69-72](../firetrack/joint_calibration_solver.py#L69-L72)). |
| **F15** | **Edited or re-detected inputs don't invalidate downstream results.** Detection reuse checks existence only; edits keep old trajectories. | [detect.py:158-162](../firetrack/detect.py#L158-L162), [231-243](../firetrack/detect.py#L231-L243), [results.py:164-200](../firetrack/results.py#L164-L200) | Source-confirmed | Fingerprint inputs; mark results stale. |
| **F16** | **The annotator's "active source" is shared by every browser tab**, and click requests carry only a row index. | [webui.py:852-872](../firetrack/webui.py#L852-L872), [1034-1041](../firetrack/webui.py#L1034-L1041), [1577-1598](../firetrack/webui.py#L1577-L1598) | Source-confirmed; corruption scenario inferred | Include source/run/label in requests. |
| **F17** | **Local classifier setup is fragile.** The wheel requires Python <3.13, but the prep script's comment says ≥3.10 and it only relaxes the check below 3.10. The prep script and server default to different caches. The preload test depends on the package. | [prepare_classifier.py:34-37](../scripts/prepare_classifier.py#L34-L37), [45-51](../scripts/prepare_classifier.py#L45-L51), [classification.py:47-50](../firetrack/classification.py#L47-L50), [test_tracking_classification.py:77-85](../tests/test_tracking_classification.py#L77-L85) | **Reproduced** (dry-run refusal); paths source-confirmed | Document the Python range; share one cache default; mock `find_spec`. |
| **F18** | **A malformed clicks file breaks `/api/status`**, and the dashboard silently stops updating. | [webui.py:633-637](../firetrack/webui.py#L633-L637), [clicks.py:213](../firetrack/clicks.py#L213), [dashboard.py:626](../firetrack/dashboard.py#L626) | **Reproduced** | Catch `JSONDecodeError`; show an error. |

### 5.2 Additional defects and fragilities

| ID | Finding | Evidence | Status |
|---|---|---|---|
| F19 | `remove_upload` rewrites the calibration as `{"cameras": …}`, dropping `world_frame`, `calibration_clock`, and `diagnostics` (the route is unused by the UI). | [webui.py:496-506](../firetrack/webui.py#L496-L506) | Reproduced |
| F20 | Reconstruction rescales `K` separately in x and y when the calibration resolution differs, even for a rotated resolution (fx 905 → 1609, fy → 509), without a warning. | [triangulate_uploads.py:194-203](../firetrack/triangulate_uploads.py#L194-L203) | Reproduced |
| F21 | In the flat legacy `default` tracking layout, detections are labelled with the camera name as run, so Results for `default` shows no detections. | [results.py:79-80](../firetrack/results.py#L79-L80) | Reproduced |
| F22 | Qualisys untracked rows keep zero rotation matrices although documented as NaN. No current consumer reads them. | [mocap.py:187-190](../firetrack/mocap.py#L187-L190), [214-218](../firetrack/mocap.py#L214-L218) | Reproduced |
| F23 | Job output capture is process-wide, so other threads' prints appear in the job log. | [jobs.py:79](../firetrack/jobs.py#L79) | Reproduced (live) |
| F24 | Detections store an **absolute** `video_path`. Moving or remounting the work root (e.g., Docker `/work` vs a host path) breaks result frames and metadata lookup. | [detect.py:186](../firetrack/detect.py#L186), [results.py:136-150](../firetrack/results.py#L136-L150) | Source-confirmed |
| F25 | Mocap-tab playback ignores per-camera sync offsets: summary keys are phones, lookups use labels. | [results_page.py:345-352](../firetrack/results_page.py#L345-L352), [triangulate_527.py:731](../firetrack/triangulate_527.py#L731) | Source-confirmed; effect inferred |
| F26 | No auth/TLS and a default bind of `0.0.0.0`. Mutation endpoints outside the job lock; upload/clear buttons stay enabled during jobs. | [cli.py:281](../firetrack/cli.py#L281), [dashboard.py:634](../firetrack/dashboard.py#L634) | Source-confirmed |
| F27 | 7 tests are not portable to native Windows (symlink privilege, slash-separated path strings). | §4.8 | Reproduced |
| F28 | `dashboard.py:462` contains `\.` inside a Python bytes literal, producing a `SyntaxWarning` (a future error) on Python 3.12+; the JavaScript still receives `\.`. | [dashboard.py:462](../firetrack/dashboard.py#L462) | Reproduced |
| F29 | The dashboard's `HELP` text is never shown: `renderHelp` targets a missing `#getting-started` element. | [dashboard.py:454-457](../firetrack/dashboard.py#L454-L457) | Source-confirmed |
| F30 | Nine server routes are not used by the bundled pages (listed in §2.2), including the near-duplicate `/api/nomocap-format-zip`. | text search | Source-confirmed |
| F31 | Text-only detection takes the first reported object in each frame, so identity can switch if several objects match "drone". | [detect.py:143-149](../firetrack/detect.py#L143-L149) | Source-confirmed; effect inferred |
| F32 | The last robust-rejection pass can change visibility without re-solving before cost and Jacobian are reported. | [joint_calibration_solver.py:387-440](../firetrack/joint_calibration_solver.py#L387-L440) | Inferred edge case |

### 5.3 Numerical and evaluation limitations

- **Mocap accuracy is mocap-assisted.** Each evaluated run re-searches its time offsets against its own mocap before scoring against that mocap ([triangulate_527.py:595-619](../firetrack/triangulate_527.py#L595-L619), [704-707](../firetrack/triangulate_527.py#L704-L707)). This is useful diagnostically but is not independent evidence of No-Mocap performance.
- **Calibration acceptance has no absolute accuracy test** (F1). It relies on pixel agreement, depth, bounds, rank, and ambiguity checks. Its covariance ignores the robust weighting and the spline bias.
- **Weak synchronization failure modes.** A mocap sync search with no valid candidate returns the nominal offset with infinite error and continues ([475-511](../firetrack/triangulate_527.py#L475-L511)). Upload reconstruction can write an all-NaN trajectory when timelines don't overlap.
- **Coverage.** The fewest-rows reference camera limits output rows. Subset search grows exponentially with cameras. Smoothing fills up to 30 frames (1 s at 30 fps) with extrapolation. Interpret smoothed counts together with `n_views`.
- **Timing model.** Constant FPS and fixed clock offsets are assumed. Dropped or variable-rate frames and clock drift are not modelled, and rolling shutter is ignored.
- **Coordinate assumptions.** Legacy Android crop/rotation mapping and dataset calibration after rotation need fixture-based confirmation with real data ([calibrate_logs.py:381-473](../firetrack/calibrate_logs.py#L381-L473), [format_527.py:111-114](../firetrack/format_527.py#L111-L114)).
- **3D view orientation.** The viewer treats world y as vertical, while ENU data has z up (§2.11).

### 5.4 Documentation versus implementation

| Documentation says | Implementation does |
|---|---|
| README: detection needs click annotations ([46](../README.md#L46)) | Text-only detection works when no approved clicks exist (§2.4). |
| README: six "main commands"; triangulation workflow ([57-104](../README.md#L57-L104)) | Nine commands exist; `triangulate` is mocap-only; No-Mocap tracking is dashboard-only. |
| README: two workflows, dataset and manual upload ([39-45](../README.md#L39-L45)) | The default tab is log-calibrated No-Mocap with tracking runs and classification. |
| docs: radius 0 fixes the lag ([110](../docs/flight-log-calibration.md#L110)) | The solve works, but the export fails (F5). |
| docs: held-out "all-observation and inlier-only errors and coverage" ([66-67](../docs/flight-log-calibration.md#L66-L67)) | The joint solver reports aggregate track/ray errors and depth fractions; per-camera inlier counts are used for acceptance but not exported. |
| docs: `--smoothing-m 0.1` described neutrally as an "RMS allowance" ([106](../docs/flight-log-calibration.md#L106)) | It can bias calibration substantially (F1). |
| CLI help: `--huber-px` is "Huber transition… for each 2D detection residual" ([cli.py:242](../firetrack/cli.py#L242)) | It is the pixel normalization; with `f_scale=1` that also makes it the per-component Huber transition, and it sets the pixel-vs-track weighting. |
| `prepare_classifier.py` comment: wheel declares `>=3.10` ([34-35](../scripts/prepare_classifier.py#L34-L35)) | The metadata says `<3.13,>=3.10`; install fails on 3.13+ (F17). |
| `classify.py`: "Start the app once while online" fetches the model ([52-53](../firetrack/classify.py#L52-L53)) | Server preload is `local_files_only=True`; only the prep script downloads. |
| `ekf.py`: "Extended Kalman Filter"; "Short NaN gaps are interpolated" ([1](../firetrack/ekf.py#L1), [92](../firetrack/ekf.py#L92)) | Linear filter; gaps are predicted, and up to 30 frames are extrapolated after the last detection. |
| `results.py`: "Read-only access" ([1](../firetrack/results.py#L1)) | Also rewrites detections (`apply_centroid_edits`). |
| `mocap.py`: loaders for `*_mocap.tsv` / `*_mocap_6D.tsv`; rotations NaN-masked ([3-11](../firetrack/mocap.py#L3-L11), [189-190](../firetrack/mocap.py#L189-L190)) | The pipeline looks for `<run>_data_6D.tsv` or `<run>_6D.tsv`; rotations are not masked (F22). |
| Dockerfile: vocab "packaged under vendor/sam3_assets" ([37](../Dockerfile#L37)) | The file was removed from the repository (F3). |
| dashboard.py docstring: "single self-contained document" ([4](../firetrack/dashboard.py#L4)) | Loads Google Fonts from the internet (fallback fonts exist). |
| pyproject: Python ≥3.9 | Only 3.14 was exercised here (light dependencies). The classifier needs 3.10–3.12, and SAM3/torch compatibility is unverified. |

### 5.5 Open questions for the project owner

1. What values of `--smoothing-m` and `--focal-sigma` have been validated on real ArduPilot logs, and against what independent reference? (F1)
2. Where should the SAM3 vocabulary come from for builds? Is restoring the `66c31bc` blob acceptable, given its origin and license? (F3)
3. Are the phones' clocks synchronized to each other before recording? What drift and dropped-frame tolerance is acceptable?
4. For the 5-27 dataset, does `calibration.json` describe the rotated (formatted) frames or the raw sensor frames?
5. Do SAM3 and OpenCV decode phone videos with identical frame counts and orientation (rotation metadata)?
6. Which Python/CUDA/SAM3 combination and image revision are known to work end to end?
7. What coverage and accuracy define success, and how biased is ArduPilot's POS estimate expected to be at these ranges?
8. What evidence supports trusting the ArduPilot/PX4 classifier on reconstructed (vision) trajectories?
9. Is the broad "Clear dataset" scope intentional (F4), and should tracking ZIPs keep their logs (F11)?

### 5.6 Significant corrections to the previous analysis

| Previous statement | Correction in this revision |
|---|---|
| "None of the 61 test methods ran"; Git "not found on PATH" | Tests **ran**: 53/61 on Windows and 60/61 on Linux. Every failure the previous report predicted as environment-specific was confirmed, and no numerical test failed. Git was available through Git Bash, so history and diffs were inspected. |
| "The repository contains no procedure that creates this vocabulary asset" | Still true as a procedure, but the exact file exists in Git history (`66c31bc`) and was deliberately removed in `3a0a659`; restore steps are in §4.4. |
| "Even a help command can fail when numerical/video dependencies are missing" | Refined: `--help` needs only NumPy and OpenCV (verified); PyTorch, SAM3, SciPy, and CasADi are loaded lazily. |
| Calibration "reports 3D log disagreement but does not reject against a maximum 3D RMSE threshold" | Kept, and strengthened with a reproduced ~1 m / 10% bias from default smoothing that every diagnostic missed (F1), plus the reason: validation uses the same smoothed spline. |
| Windows log globbing: "a single usable log **can** appear twice" | Reproduced: with one log, native-Windows dashboard calibration **always** fails (F2). |
| Interpolation loses "a valid frame immediately preceding a missing one" at "an exact integer frame" | Corrected: epoch-scale timestamps are rarely exact (200/3,000 in the check), so the rule is "both bracketing frames must be detected". Isolated detections never contribute, and gap edges are lost depending on rounding (F14, reproduced). |
| The Windows results-listing and cursor-stall claims were mechanism demos | Now reproduced through the live API and the real `JobRunner` (F13, F10). |
| Fixed-lag export failure shown with a stand-alone `json.dumps(NaN)` | Reproduced through the real `calibrate_from_log` (F5). |
| End-to-end example was illustrative only | Now backed by an executed synthetic run (§3.1) and a verified geometry example (§3.3). |
| Not previously reported | F11 tracking-ZIP logs dropped; F12 multi-run dataset ZIPs merged; F17 classifier Python range and cache mismatch; F18 malformed clicks break status; F20 anisotropic K rescale; F21 flat-layout labels; F24 absolute video paths; F25 Mocap playback offsets; F8 extended (discarded edits stay drawn). |
| "`python` resolves to an inaccessible WindowsApps alias" | Not re-verified and removed; this analysis used `python3.14`. |

---

## 6. Recommended Reading Order

1. **README.md, pyproject.toml, Dockerfile, and §1.4–1.5 of this report.** What the tool promises, how it is packaged, and what lives on disk.
2. **cli.py, then `WebConfig`, `_stage_fn`, and `_handle_run` in webui.py.** Which operations exist and how a button becomes a function call.
3. **sources.py and dataset_527.py.** How folders become `VideoSpec` labels and output paths, including legacy names.
4. **clicks.py and detect.py.** One click through prompts, SAM3 sessions, and centroids on disk.
5. **triangulate_uploads.py, then triangulate_527.py lines 514-582.** The simplest complete path from calibrated cameras to 3D points: conventions, interpolation, DLT, subset choice.
6. **ekf.py and `write_csv`.** What "smoothed" means and why raw coverage matters.
7. **docs/flight-log-calibration.md, then flight_reference.py lines 51-118.** Trustworthy time and coordinates first.
8. **calibrate_logs.py `calibrate_from_log`, `SplinePath`, then joint_calibration_solver.py** alongside `test_joint_calibration_solver.py` and `test_calibration_integration.py`. The core algorithm, with runnable examples; §3.1 shows what to watch for.
9. **results.py and results_page.py.** How results are served, drawn, and edited.
10. **jobs.py and the rest of webui.py (uploads, clears, repairs), then dashboard.py.** Operational behavior and the findings in §5.
11. **mocap.py, format_527.py, triangulate_527.py (mocap parts), calibrate_mocap.py.** Only if you need the 5-27 mocap workflow.
12. **classify.py, classification.py, scripts/prepare_classifier.py, the remaining tests; calibration_solver.py `fit_pose_time` last** (legacy).

---

## 7. Inspection Coverage

### 7.1 Files read in full

All 49 non-Git files were read completely, including every line of embedded HTML/CSS/JavaScript:

| Group | Files |
|---|---|
| Prior analysis (2), read first | `analysis/fireRepo_high_level.md`, `analysis/fireRepo_detailed_analysis.md` |
| Root configuration and docs (7) | `.dockerignore`, `.gitignore`, `Dockerfile`, `pyproject.toml`, `requirements.txt`, `README.md`, `docs/flight-log-calibration.md` |
| Package (25) | `firetrack/__init__.py`, `__main__.py`, `cli.py`, `webui.py`, `dashboard.py`, `results_page.py`, `clicks.py`, `jobs.py`, `sources.py`, `dataset_527.py`, `format_527.py`, `detect.py`, `flight_reference.py`, `calibrate_logs.py`, `joint_calibration_solver.py`, `calibration_solver.py`, `calibration_priors.py`, `mocap.py`, `calibrate_mocap.py`, `triangulate_527.py`, `triangulate_uploads.py`, `ekf.py`, `results.py`, `classify.py`, `classification.py` |
| Script (1) | `scripts/prepare_classifier.py` |
| Tests (10) | all files in `tests/` |
| Vendor (4) | `vendor/__init__.py`, `vendor/sam3_assets/__init__.py`, `vendor/moonshot_classifier_assets/__init__.py`, `fire-moonshot-classifier-model.json` |

That is 47 application-related files (39 Python) plus the 2 reports.

### 7.2 Exclusions

- **`.git/` internals**: version-control data, not application code. They were queried through `git log`/`git show`/`git cat-file` for history, and no objects were modified.
- **Generated files, installed dependencies, binaries, build artifacts**: none exist in the checkout. The ignore rules name possible ones (caches, environments, videos, NPZ, weights, the vocab `.gz`), but none were present.
- **External packages**: pymavlink, OpenCV, SciPy, CasADi, SAM3, PyTorch, and the classifier are dependencies, not part of this repository. Only three narrow checks touched them: pymavlink's `gps_time_to_epoch` source, the classifier wheel's file list, metadata, and inference-path import statements, and the pinned downloads' checksums. The classifier's model weights and training were not reviewed.

### 7.3 Checks performed

| Check | Environment | Result |
|---|---|---|
| Full unittest suite (twice on Windows, once on Linux) | Python 3.14.7 Windows; 3.14.4 WSL Ubuntu 26.04 | 53/61 and 60/61 (§4.8) |
| `python -m firetrack --help` and CLI import footprint | Windows venv without torch/sam3 | Exit 0; only NumPy + OpenCV imported |
| Synthetic end-to-end run incl. live server, `/api/run`, job runner | Windows venv, 127.0.0.1 | §3.1 |
| Calibration sensitivity (7 variants) and spline shrinkage | Windows venv | §3.1, F1 |
| Targeted reproductions: F2, F4, F5, F6, F10–F14, F17–F23, F28 | Windows venv | All reproduced as described |
| Two-camera geometry and outlier-subset example | Windows venv | Exact (§3.3) |
| Route-usage text search of the three pages | grep | 9 routes unused |
| Git history of `vendor/sam3_assets` | Git Bash | Blob in `66c31bc` (1,356,917 B, SHA-256 `924691ac…6804a`), removed in `3a0a659` |
| Classifier pins | network | Wheel and model SHA-256 match; wheel requires Python <3.13; pip refuses on 3.14 |
| Repository integrity | SHA-256 of all 49 files before and after | Identical. One stray empty file (`=0.30,`) that a mis-quoted WSL command created in the repo root during this session was found by this check and deleted; nothing else was touched. |

All environments, scripts, and outputs live in the session scratchpad, not in the repository. Only these two reports were changed.

### 7.4 Remaining gaps

- SAM3 detection quality, GPU memory behaviour, and frame/orientation agreement with OpenCV.
- Real ArduPilot logs (parsing, clock stability, POS accuracy) and real phone videos.
- Recommended calibration settings for real data (F1 was shown on synthetic data only).
- Browser behaviour (F8, F9, F25) and the Docker image build/run.
- FFmpeg formatting and the mocap dataset pipeline, which needs 5-27 data.
- Classifier inference and its accuracy.
- torch/sam3 installation on Windows or Python 3.14.

### 7.5 Revision and environment

Analysis date 2026-09-29. Repository `main` at `769561474c793ae7a4a7813bc520a9241e6224a6`; application source identical to `1bed945d11f5d54c46969634be9f7e02c6518df6`. Environment: Windows 11 (Python 3.14.7 via `python3.14`; Git via Git Bash; no Docker/FFmpeg/Node) and WSL Ubuntu 26.04 (Python 3.14.4).

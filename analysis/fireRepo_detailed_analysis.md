# fireRepo: detailed technical analysis

FireTrack reconstructs a drone's 3D trajectory from videos recorded by stationary cameras. It combines machine-learning detection, camera calibration, time alignment, geometric reconstruction, and smoothing. A browser dashboard orchestrates these stages, displays results, and offers optional flight-log comparison and trajectory classification. The Python package is named **firetrack**; the actual repository folder is **fireRepo**.

Companion report: [fireRepo_high_level.md](fireRepo_high_level.md).

This analysis describes the checkout inspected on **2026-09-28**, whose local main-branch reference is **1bed945d11f5d54c46969634be9f7e02c6518df6**. Paths in citations are relative to fireRepo/. Line numbers include blank lines and refer to this checkout.

**Evidence and limits.** All 47 non-Git files were read in full for this analysis. “Source-confirmed” below means the code establishes the behavior; it does not mean a complete pipeline run was observed. The following standard-library-only checks were run with bytecode writing disabled:
- All 39 Python files passed in-memory compilation using Python 3.14.7.
- `pyproject.toml` parsed.
- The classifier manifest passed its validation function.
- A job-log check reproduced the cursor stall.
- Strict JSON rejected NaN, which is the mechanism behind the fixed-lag export failure.

The 61 repository test methods, GPU inference, numerical solvers, browser interactions, the Docker image, and real-data workflows were **not executed**, because their dependencies and input assets are absent locally. SHA-256 hashes of all 47 repository files were identical before and after this work, so no repository files were changed.

## 1. Repository Structure and File Breakdown

### 1.1 Annotated directory tree

This is the complete application-file tree, including hidden configuration files. Git internals are collapsed.

~~~text
fireRepo/
├── .git/                              Git metadata; excluded from application review
├── .dockerignore                      Docker context exclusions
├── .gitignore                         Generated/local file exclusions
├── Dockerfile                         CUDA/PyTorch runtime image and asset preparation
├── pyproject.toml                     Package, dependencies, entry point, pytest settings
├── requirements.txt                   Docker's runtime dependency list
├── README.md                          User introduction and older workflow commands
├── docs/
│   └── flight-log-calibration.md       New calibration model, timing, and usage notes
├── firetrack/
│   ├── __init__.py                     Package version 0.1.0
│   ├── __main__.py                     python -m firetrack entry point
│   ├── cli.py                         Argument parsing and stage dispatch
│   ├── webui.py                       HTTP API, upload processing, and orchestration
│   ├── dashboard.py                   Main dashboard HTML/CSS/JavaScript
│   ├── clicks.py                      Click storage, frame serving, annotation HTML
│   ├── jobs.py                        One background job, progress, and captured logs
│   ├── sources.py                     Source discovery and legacy-name compatibility
│   ├── dataset_527.py                 Fixed dataset runs, phones, and camera paths
│   ├── format_527.py                  Dataset video rotation/transcoding
│   ├── detect.py                      SAM3 sessions → per-frame mask centroids
│   ├── mocap.py                       Qualisys TSV positions, rotations, and timing
│   ├── calibrate_mocap.py             Export mocap-derived upload calibration
│   ├── calibrate_logs.py              ArduPilot log reading and calibration orchestration
│   ├── calibration_priors.py          Optional measured camera constraints
│   ├── calibration_solver.py          Earlier single-camera calibration solver
│   ├── joint_calibration_solver.py    Shared-lag, multi-camera calibration solver
│   ├── triangulate_527.py             Shared geometry and mocap/dataset reconstruction
│   ├── triangulate_uploads.py         Reconstruction using upload calibration
│   ├── ekf.py                         Linear Kalman filter and backward smoother
│   ├── flight_reference.py            Independent log-reference comparison sidecars
│   ├── results.py                     Result readers, centroid edits, and downloads
│   ├── results_page.py                Synchronized 2D and interactive 3D viewer
│   ├── classify.py                    Pinned classifier manifest/model loading
│   └── classification.py             Per-run classifier state and prediction files
├── scripts/
│   └── prepare_classifier.py          Verified wheel install and model prefetch
├── tests/
│   ├── test_automatic_flight_reference.py  Automatic reference selection and failure behavior
│   ├── test_calibration_clear.py           Scoped calibration deletion and locking
│   ├── test_calibration_integration.py     Calibration orchestration and export
│   ├── test_calibration_solver.py          Single-camera numerical solver
│   ├── test_camera_repair.py               Selected-camera clear and redetect
│   ├── test_flight_reference.py            Coordinates, timing, metrics, and staleness
│   ├── test_joint_calibration_solver.py    Joint poses, focal lengths, and lag
│   ├── test_tracking_classification.py     Mocked classification and persistence
│   ├── test_tracking_names.py              Old/new tracking-name compatibility
│   └── test_tracking_output_clear.py       Run-scoped output deletion
└── vendor/
    ├── __init__.py                     Runtime asset namespace
    ├── sam3_assets/
    │   └── __init__.py                 Asset namespace; expected vocabulary gzip absent
    └── moonshot_classifier_assets/
        ├── __init__.py                 Classifier manifest namespace
        └── fire-moonshot-classifier-model.json  Model repository/revision/checksum
~~~

The files under vendor/ are small first-party packaging/asset metadata, so they were read rather than automatically excluded as third-party dependencies. There is no checked-in video, flight log, mocap recording, checkpoint, trajectory, compiled frontend, or full dependency source tree.

### 1.2 Entry points and component boundaries

- **Command line:** pyproject.toml:22–23 installs the firetrack command pointing to cli.main; __main__.py:1–4 invokes the same function for python -m firetrack. cli.py:288–291 builds the parser, parses arguments, and calls the selected handler.
- **Browser:** cli._cmd_webui calls webui.serve_webui. That function constructs configuration, job state, annotation state, and classification services, then serves a nested HTTP handler (webui.py:875 onward). dashboard.py and results_page.py provide embedded browser code; there is no separate JavaScript build.
- **Core computation:** detect.py produces 2D observations; calibrate_logs.py or mocap helpers establish camera geometry; triangulate_527.py contains reusable geometry; triangulate_uploads.py handles newer uploads; ekf.py smooths trajectories.
- **Supporting state:** clicks.py stores annotations; sources.py normalizes source identities; jobs.py holds transient execution state; results.py reads and edits disk artifacts; classification.py and flight_reference.py add optional derived results.
- **Tests:** the ten files exercise numerical calibration and state-management behavior. Their exact coverage and limitations appear in section 4.

Several imports connect these boundaries tightly. In particular, cli.py:9–15 imports most orchestration modules immediately. Therefore even a help command can fail when numerical/video dependencies are missing; the CLI is not dependency-free.

### 1.3 Configuration, documentation, and asset files

| File | Main contents and connections |
|---|---|
| [README.md](fireRepo/README.md) | Describes static-camera tracking, published Docker commands, staged dataset processing, output paths, and gated SAM3 access; its workflow list omits newer calibration/tracking features. |
| [docs/flight-log-calibration.md](fireRepo/docs/flight-log-calibration.md) | Explains flight-log timing, camera conventions, the joint solver, held-out validation, priors, and command options; compare claims with the implementation in section 5. |
| [pyproject.toml](fireRepo/pyproject.toml) | Setuptools build, distribution docker-fire-tracking 0.1.0, Python >=3.9, eight runtime dependencies, firetrack command, vendor asset inclusion, and pytest testpaths/pythonpath. |
| [requirements.txt](fireRepo/requirements.txt) | Same eight runtime dependencies used by Docker; no pytest or external classifier wheel. |
| [Dockerfile](fireRepo/Dockerfile) | Starts from PyTorch 2.6/CUDA 12.4, installs system tools including FFmpeg, installs the package, runs classifier preparation, copies SAM vocabulary, exposes port 8080, and defaults to firetrack webui. |
| [.dockerignore](fireRepo/.dockerignore) | Keeps Git metadata, environments, generated results, model binaries, recordings, and archives out of builds; the vocabulary gzip is not excluded here. |
| [.gitignore](fireRepo/.gitignore) | Excludes environments, local state, outputs, recordings, checkpoints, and specifically vendor/sam3_assets/*.gz, explaining why the required Docker asset can be absent. |
| [scripts/prepare_classifier.py](fireRepo/scripts/prepare_classifier.py) | Downloads a specific release wheel, checks SHA256, installs it without dependency resolution, then retrieves a checksum-verified model; Docker runs this at build time. |
| [vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json](fireRepo/vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json) | Contains model repo_id, filename, immutable revision, and SHA256; classify.load_model_manifest consumes it. These are public asset identifiers, not credentials. |
| vendor package markers | The three one-line __init__.py files make resource namespaces importable; they contain no algorithms. |

### 1.4 The most important data contracts

**SAM3** (Meta's “Segment Anything Model 3”) is the external neural network this project uses to outline, or *segment*, a prompted object in every frame of a video. A **mask** is that per-pixel outline. An **.npz** file is NumPy's zip archive of named arrays. A **centroid** is the average x/y pixel coordinate of an object's mask. **Intrinsics** describe how camera-space directions map to image pixels: focal lengths fx/fy, principal point cx/cy, and lens distortion. **Extrinsics** describe position and orientation relative to a shared world frame. **Triangulation** recovers a 3D point from observations in multiple calibrated cameras.

| Artifact | Essential data | Producer → consumer |
|---|---|---|
| Camera video and sidecars | Video frames, camera parameters, start time and sampling information | Upload/formatting → detection and calibration |
| Click JSON | label, click_x, click_y, click_frame, approval state | clicks.py → detect.load_clicks |
| centroids.npz | centroids shaped (N,2), frame indices, dimensions, fps, video path, label, prompt, anchors | detect.save_detection_outputs → calibration, triangulation, results |
| Detection summary.json | Frame count, detected count/rate, dimensions, paths, prompt/click information | detect.py → dashboard/result listings |
| Optional masks.npz | Frame-number keys holding binary masks | detect.py → saved diagnostic data; triangulation uses centroids |
| uploads_calibration.json | Camera intrinsics/extrinsics plus timing and calibration diagnostics | calibrate_logs.py, calibrate_mocap.py, or manual upload → triangulate_uploads.py |
| trajectory.npz | Time samples, raw/smoothed positions, view counts, reprojection errors, reference/ground-truth fields | Reconstruction → viewer, classifier, reference comparison |
| trajectory.csv | Human-readable per-frame reconstruction rows | Reconstruction → download/analysis |
| Trajectory summary.json | Counts, camera/time metadata, calibration snapshot, quality metrics | Reconstruction → readiness, viewer, reference comparison |
| flight_reference.json / .npz | Reference provenance, metrics, fingerprint, sampled log positions | flight_reference.py → results.py |
| flight_comparison.csv | Reconstruction and flight-log reference values/errors | flight_reference.py → download |
| classification/.../prediction.json | External prediction plus trajectory/model identity | ClassificationService → dashboard/results |

N is the number of frames or trajectory samples. Missing numerical observations generally use NaN (“not a number”) internally. Browser JSON converts missing array entries to null; this matters because strict JSON cannot encode NaN. Detection schema: detect.py:165–215. Result conversion: results.py:127–133. Classification persistence: classification.py:130–150.

## 2. In-Depth Code Analysis

### 2.1 Overall execution and data flow

~~~mermaid
flowchart TD
    A[Camera videos and metadata] --> B[Optional dataset formatting]
    A --> C[SAM3 detection with text or clicks]
    B --> C
    C --> D[2D centroids by camera and frame]
    L[Calibration flight log] --> E[Joint camera calibration]
    D --> E
    M[Motion capture or manual camera poses] --> F[Camera calibration]
    E --> F
    D --> G[Align camera observation times]
    F --> G
    G --> H[Multiview triangulation and outlier rejection]
    H --> I[Kalman filtering and backward smoothing]
    I --> J[Trajectory NPZ / CSV / JSON]
    J --> K[Browser results and manual corrections]
    K --> D
    J --> N[Optional CPU trajectory classification]
    J --> O[Optional comparison with flight-log reference]
~~~

The calibration flight and the later tracking flight are distinct inputs in the No-Mocap workflow. A flight log supplies known 3D positions while the solver learns the cameras. Later tracking reconstruction uses those saved camera parameters and video detections. Reference comparison is a separate optional operation; it does not adjust the reconstruction to make the comparison look better.

### 2.2 CLI and detection

**CLI handlers.** cli.py:31–135 are thin wrappers around formatter, annotation, detection, triangulation, log summary, calibration, and server functions. They pass parsed settings and print summaries. _cmd_run_all (138–173) creates formatted, detections, and triangulation paths, then runs those three stages unless their skip flags are supplied. It does not create click annotations, perform No-Mocap log calibration, or classify trajectories. _add_only supplies repeatable camera labels; --run instead selects dataset runs for triangulation.

**Discovery and output records.** detect.VideoSpec identifies a label, relative directory, and actual video path. DetectionOutputPaths groups the three artifact paths. DetectionResult reports dimensions, counts, anchors, and a derived detection_rate. discover_specs adapts dataset discovery; select_specs rejects unknown requested labels; output_paths preserves source-relative hierarchy (detect.py:20–74).

**Prompt selection.** With a clicks file, run_detection_on_specs skips videos without an approved usable click. With no clicks file, it uses text prompts at approximately 35%, 60%, and 85% of each video. The README's statement that click annotations are required is thus too restrictive. Explicit approved=false rows are ignored; missing approval is accepted by load_clicks (detect.py:77–114,343–400).

**SAM3 interaction.** build_predictor imports the external SAM3 builder lazily and passes the packaged vocabulary path if present. run_sam3_on_video probes dimensions with OpenCV, optionally reuses completed outputs, initializes a SAM session, adds prompts, propagates masks, and saves centroids. Text-only mode takes the first reported object mask; click mode requires object ID 9999. That distinction matters when several drone-like objects appear (detect.py:117–155,218–340).

The important detection block, explained in execution order:

| Lines in detect.py | What happens and why |
|---|---|
| 231–243 | Reuses existing output files unless overwrite is requested; this is an existence check, not a check that the prompt/video/calibration inputs are unchanged. |
| 245–255 | Reads frame count and dimensions, validates click frame bounds, or chooses text anchors. |
| 261–264 | Opens a model session against the actual video path. |
| 266–279 | For click mode, adds a text prompt and performs an initial bidirectional propagation. |
| 280–287 | Adds the user's positive point, normalized by image width/height, with object ID 9999. |
| 289–295 | Without a click, adds text prompts at the selected anchor frames. |
| 297–298 | Allocates an (N,2) array filled with NaN and optionally a mask dictionary. |
| 300–315 | Propagates both directions, ignores invalid frame indices or empty/missing masks, then assigns mean x and mean y of mask pixels. |
| 316–319 | Optionally retains masks and publishes progress; a callback is also invoked every eight stream results. |
| 320–325 | Closes the model session and releases cached CUDA memory in finally blocks. |
| 327–338 | Writes NPZ arrays and a JSON summary after successful processing. |

The mask centroid is not a detector bounding-box center, a rigid-body origin, or a known physical point on the drone. Its location can move as the silhouette changes; that is one source of measurement error.

**Memory and errors.** By default each video's predictor is constructed and released separately. FIRETRACK_REUSE_SAM_PREDICTOR=1 reuses one predictor across videos. GPU cleanup errors are suppressed. A RuntimeError containing “out of memory” is replaced with a more useful diagnostic; other runtime failures propagate. No discovered videos or no processed videos is an error. Saving all masks can consume substantial host memory because they are held until output writing (detect.py:125–134,297–317,353–399).

### 2.3 Flight-log calibration: from files to camera parameters

**File-facing orchestration.** calibrate_logs.calibrate_from_log (246–368) reads the drone log, constructs a smooth reference path, loads every camera folder and its detections, creates JointCamera inputs, calls fit_joint_calibration, and publishes reusable calibration JSON. It requires at least two cameras and will not save a successful subset when another camera fails. Output is written to a temporary sibling and replaced only after serialization succeeds.

The supporting records and loaders matter because camera geometry is only meaningful when units and timestamps agree:

- load_drone_track reads timed positions into DroneTrack and establishes the geographic frame (calibrate_logs.py:23–28,63–70).
- Camera-folder discovery requires immediate subdirectories containing video.mp4 and associated camera.json and metadata.json; missing sidecars are reported together (126–153).
- Explicit camera.json uses K, dist, and resolution. These must describe **decoded video pixels**; resolution mismatch is rejected rather than silently resized or rotated (73–82,457–462).
- Legacy Android camera dictionaries are also accepted. _aspect_fill_K assumes a centered uniform crop; _sensor_to_decoded_K infers rotations from orientation/dimensions; _rotate_K transforms image axes; _image_space_K incorporates optional formatting rotation (371–473). This is compatibility logic with additional assumptions, not a substitute for knowing the decoded-image calibration.
- _with_detections validates centroid shape, positive FPS, dimensions, and optional finite, strictly increasing integer frame_indices. _find_centroids searches only the relevant run-specific layouts, preventing accidental fallback to a different flight (156–195).
- summarize_log_and_run reports UTC overlap and uses video duration when metadata lacks an end time (198–239). Dashboard log selection requires exactly one usable overlapping log.

**Flight-log source.** flight_reference.read_flight_track (51–89) opens an ArduPilot DataFlash binary with pymavlink. It uses primary-receiver GPS messages with a valid 3D fix to map boot time to UTC, taking the median UTC-minus-boot offset. It rejects a 5th-to-95th-percentile clock spread above 0.5 seconds and backwards POS timestamps. The trajectory comes from fused **POS** latitude, longitude, and absolute altitude, not raw GPS position samples. Duplicate position timestamps are removed; the reader is closed even on failure.

**World coordinates.** to_calibration_frame computes local horizontal east/north distances around the first valid latitude/longitude while retaining **absolute altitude above mean sea level (AMSL)** as its third coordinate (flight_reference.py:92–105). Thus a z coordinate of 150 means 150 m AMSL, not 150 m above the camera or launch point. The function computes ellipsoid-surface Earth-centered coordinates, subtracts the surface origin, projects onto east/north axes, and appends altitude unchanged. This is a local horizontal frame with absolute height, not a full altitude-aware Earth-centered conversion. enu_frame validates frame metadata and rejects inconsistent camera origins (25–48).

**Clock convention.** The joint solver defines:

~~~text
delta = camera_time - log_time
log_query_time = camera_time - delta
camera_frame_time = startTime_microseconds / 1,000,000 + frame_index / fps
~~~

A positive lag of 0.237 s means camera time 100.237 corresponds to log time 100.000. The exported calibration_clock.camera_minus_log_s records this lag. It is explicitly marked as **not applied to later tracking camera timestamps**; per-camera source.time_offset_s is zero for this calibration path. Later videos keep their own synchronized timestamps. Optional log comparison applies the saved lag to log queries only. The hidden backward-compatible time_offset_s calibration argument has the opposite sign and is negated (calibrate_logs.py:280–290,320–324,351–355; flight_reference.py:121–144,178–179).

This design assumes cameras are already synchronized with one another. The current joint fit solves one common camera-versus-log lag, not separate camera offsets or clock drift.

#### Continuous reference trajectory and optional priors

SplinePath in calibration_solver.py:31–68 turns discrete logged positions into a continuous function that also supplies velocity. It requires finite, ordered samples, splits at gaps above one second, and fits separate cubic splines to segments with at least four samples. Per-axis smoothing budget is n × smoothing_m². The default 0.1 m is a residual allowance for spline fitting, not a claim that the logged flight path is accurate to 0.1 m. Querying outside supported intervals or across gaps raises an error. A calibration observation is retained only if its entire searched lag interval stays within one supported segment.

Priors are optional measured constraints that supplement image evidence. Priors stores a camera position and uncertainty, directional observations and uncertainty, and source names (calibration_solver.py:71–77). calibration_priors.load_priors (42–84) requires explicit conventions_verified=true before accepting configured sensors:

| Prior | Input and checks | Effect |
|---|---|---|
| GPS camera position | gps.csv timestamps, latitude/longitude/altitude/accuracy; explicit AMSL altitude convention | Median camera position in the same geographic frame, weighted by uncertainty. |
| Gravity direction | imu.csv accelerometer values, verified device-to-camera rotation, specific-force convention, roughly stationary gravity magnitude | Constrains camera orientation relative to world up. |
| Magnetic direction | imu.csv magnetometer values and explicitly supplied world magnetic direction | Adds an orientation constraint without assuming magnetic north equals true north. |

_samples restricts sensor rows to the clip time window; _unit and _sigma reject invalid vectors or uncertainties. device_to_camera must be a proper rotation. Defaults are 2 m GPS uncertainty, 1.7 degrees gravity uncertainty, and 20 degrees magnetic uncertainty, with reported GPS accuracy able to increase horizontal uncertainty (calibration_priors.py:14–39,50–84). These checks verify declared conventions and numerical validity, not actual sensor accuracy.

#### Joint calibration algorithm

The active engine is joint_calibration_solver.fit_joint_calibration (476–614). calibration_solver.fit_pose_time remains a separately tested older solver; it is not the engine called by current calibrate_from_log.

**Observation preparation.** JointCamera groups camera name, epochs, pixels, K, distortion, and priors. JointObservations groups synchronized pixels, visibility, and training/validation masks. prepare_joint_observations chooses the camera grid with the most finite observations, interpolates other tracks without bridging gaps larger than 1.5 median frame periods, requires two visible cameras, enforces spline support, and samples at most 600 times by default. It divides time into ten blocks, holding out blocks 4 and 9 for validation, with minimum global and per-camera sample counts (joint_calibration_solver.py:26–127).

**Initialization.** _initial_poses searches candidate shared lags. At each lag it pairs logged 3D positions with observed pixels, uses OpenCV PnP (“perspective-n-point,” camera-pose estimation from known 3D/2D pairs) inside RANSAC, and scores plausible camera poses. RANSAC repeatedly tests subsets to reduce the effect of bad observations. At least 12 inliers and mostly positive depth are required. Up to two separated seeds are refined (130–177,511–524).

**Unknowns.** Each camera contributes three rotation parameters, three position parameters, and one log focal ratio. A final shared lag gives 7C+1 parameters for C cameras. The fitted focal is f0 × exp(a), where f0 is the mean supplied fx/fy. Output fx=fy **within each camera**; different cameras can have different fitted focal lengths. Principal point, skew, and distortion remain fixed. Rotation increments use Rodrigues' exponential map so the result remains a valid 3D rotation (180–187,232–245,294–310).

**The central ray calculation, line by line.** _GeometryModel triangulates a point from the candidate camera arrangement before comparing it with the log:

| Lines in joint_calibration_solver.py | Meaning |
|---|---|
| 236–245 | Define symbolic camera poses and positive focal lengths from the parameter vector. |
| 249–250 | Convert each pixel to a unit viewing ray, remove lens distortion, and rotate the direction into world coordinates. |
| 251 | Form P = I − d dᵀ, which measures displacement perpendicular to ray d. |
| 252–254 | Accumulate A = ΣP and b = ΣP × camera_position over visible cameras; solve A X = b. |
| 255–261 | Reproject the candidate 3D point into cameras and retain pixel errors and depths. |
| 262–273 | Evaluate the same geometry for all sample times, masking invisible views. |
| 274–276 | Ask CasADi to differentiate the geometry with respect to camera parameters. |

Solving A X = b minimizes summed squared distance perpendicular to the viewing rays. Nearly parallel rays can make this poorly conditioned. This calibration-time ray solver is distinct from the DLT solver used for final tracking reconstruction.

**Objective.** _residual_and_jacobian (313–360) combines normalized 3D distance to the log spline, pixel reprojection errors, a penalty for depth below 0.1, optional focal constraints, and sensor priors. Reprojection means projecting a reconstructed 3D point back into an image and measuring the miss in pixels. A Jacobian is the matrix of derivatives telling the optimizer how each residual changes with each unknown. The log-time derivative uses spline velocity, allowing fractional-frame lag refinement.

_fit_seed (363–449) calls SciPy bounded least squares with a global Huber loss: small residual components incur quadratic cost; large ones incur approximately linear cost. This robustification applies to **all residual blocks**, including priors. Local rotation components are bounded to ±0.7 radians around the seed; focal is bounded between one-quarter and four times f0; lag is bounded by the requested search interval. Up to four passes remove the worst inconsistent camera observation at a time when at least three views are available. With exactly two cameras, neither can be removed while preserving triangulatable support.

**Acceptance and uncertainty.** The solver rejects stationary/effectively collinear motion, similar-cost substantially different fits, inadequate per-camera reprojection/depth consistency in training or held-out data, too many negative-depth points, lag/focal boundary solutions, and rank-deficient sensitivity. The per-camera consistency requirement is at least 60%. Positive depth must reach at least 95% separately in the training and held-out splits (joint_calibration_solver.py:561–562). It reports 3D log disagreement but does not reject against a maximum 3D RMSE threshold (497–574).

Local covariance is approximated from the raw residual Jacobian and raw squared cost. It is conditional uncertainty near the solution, not a bound on real-world accuracy, and it differs from covariance based on the robust optimizer's effective weighting (575–594). The fixed-lag serialization problem is explained in section 5.

#### Export, defaults, and the older solver

calibrate_from_log exports world_frame, calibration_clock, cameras, and report. A camera includes video label, K, distortion, resolution, R, t, position, pose_convention="w2c", clip start, and provenance. R is world-to-camera and t = −R × position. Thus q_camera = R × X_world + t. Publishing the whole camera set atomically prevents a partial calibration from replacing a usable one (calibrate_logs.py:309–362).

| CLI option | Default | Implementation effect |
|---|---:|---|
| --sample-count | 600 | Maximum synchronized calibration sample count. |
| --clock-lag-s | 0.22 | Center/initial shared camera-minus-log lag, seconds. |
| --offset-search-radius-s | 2.0 | Lag bound around that center; zero requests fixed timing but currently breaks strict export. |
| --offset-search-step-s | 0.25 | Maximum spacing of PnP initialization lags; final refinement is continuous. |
| --ransac-reproj-px | 8 | PnP threshold and later consistency threshold. |
| --smoothing-m | 0.1 | Per-axis spline residual allowance. |
| --huber-px | 3 | Passed as pixel_sigma_px to the joint residual normalization; not solely the older vector-Huber threshold. |
| --focal-sigma | infinity | Free focal by default; zero fixes each camera to mean supplied fx/fy; values in (0,1) add relative-focal priors. |

Sources: cli.py:230–244; calibrate_logs.py:246–300. Track residual scale, rejection-pass settings, and iteration controls also exist in the numerical function but are not all exposed by CLI.

The older fit_pose_time in calibration_solver.py:184–301 estimates one camera's pose and lag with fixed intrinsics. observation_residual, huber_blocks, prior_residual, chart_residual, and _refine implement analytic derivatives, vector-based pixel Huber loss, quadratic sensor priors, repeated local rotation updates, outlier rejection, and ambiguity/observability checks (80–181). It returns explicit all-observation and inlier-only validation metrics. SplinePath, Priors, skew, and left_jacobian are still shared with the joint solver; the presence of two engines is intentional historical structure, but their diagnostics and loss functions must not be conflated.

### 2.4 Dataset preparation, motion capture, and reusable calibration

**Dataset identity.** dataset_527.py:8–18 maps three specific Android models to pixel9, galaxyS8, and galaxyS21, each with one counterclockwise quarter-turn for normalization. Camera527 holds label, run, phone, source UUID, and paths. discover_cameras expects raw videos at exactly root/run/group/camera/video.mp4, reads camera.json.device.model, and rejects unknown models or duplicate run/phone labels. discover_normalized instead expects root/run/label/video.mp4 (21–105). These are concrete dataset conventions, not general video import rules.

**Formatting.** format_527.normalize_camera rotates and re-encodes video with FFmpeg, copies available sidecars unchanged, probes old/new dimensions, and returns a manifest row. normalize_dataset validates optional labels, processes cameras, merges entries into an existing manifest, and writes manifest.json (99–163). SIDECARS includes calibration.json, camera.json, gps.csv, imu.csv, metadata.json, and rf_data.jsonl.

build_ffmpeg_cmd selects only the first video stream, H.264/libx264, CRF 18, yuv420p, and fast-start MP4. It does not copy audio or request a scale filter. run_ffmpeg_normalize tries modern passthrough timing, then older vsync flags, then default synchronization; failed partial destinations are removed. Frame-count changes produce a warning, not rejection (format_527.py:14–96,117–118). Existing outputs skip re-encoding unless overwrite is supplied, but sidecars are still copied. Removed inputs can leave retained manifest entries.

**Qualisys reading.** mocap.py defines MocapHeader, marker/rigid-body trajectory records, and 3D/6D run containers. _parse_header_lines reads required header fields; _read_numeric_block converts empty cells to NaN and pads uneven rows. load_mocap_3d loads marker coordinates; load_mocap_6d loads body position, Euler angles, residual, and rotation matrix (27–56,78–227).

Positions are converted from millimeters to meters. Rotation matrices are interpreted as column-major. Relative time is generated from actual numeric row count and sample frequency; header row-count consistency is not checked. Wall-clock parsing expects fractional seconds and defaults to a fixed UTC−4 timezone rather than a daylight-saving-aware region (mocap.py:59–75,132–168,183–227). Missing 6D positions/Euler angles are masked, but rotations are not masked as promised; section 5 describes that discrepancy.

**Dataset calibration policy.** triangulate_527.py:17–37 hardcodes fallback run names, phone configurations, mocap camera-body names, which reference runs supply calibration, and an exclusion for one phone/run. The tracked drone body must be named drone. Intrinsics, SessionData, Correspondences, FitResult, Observation, PointResult, and CamCalib carry progressively more processed camera/point information (40–108).

make_session and make_raw_session join detection NPZs to metadata and lens calibration. scaled_intrinsics rescales focal lengths/principal point by resolution ratios; frame_mocap_indices converts video frame times to nearest mocap sample numbers. build_correspondences retains supported finite observations and samples them evenly (triangulate_527.py:126–234).

fit_extrinsic requires at least eight 3D/2D pairs and uses iterative OpenCV PnP with RANSAC, followed by inlier refinement. find_calibration_time_offset searches −5 to +5 seconds at 0.1 s spacing, then ±0.1 s at 0.01 s spacing, scoring median reprojection error. camera_center_world calculates −Rᵀt; finite_body_center supplies diagnostic mocap camera locations (250–340).

calibrate/calibrate_raw build the configured camera set; save_calibration/load_calibration serialize it by phone/config identity. build_camera/build_raw_camera reuse those poses and separately call sync_time_offset for the current run, which searches a ±1.5 s interval around nominal offset in 0.02 s steps with pose held fixed (343–511,595–646). Because this uses **the evaluated run's drone mocap**, reported dataset accuracy includes mocap-assisted time alignment. It is not an independent demonstration of No-Mocap operation.

**Mocap export bridge.** calibrate_mocap.export_mocap_calibration (56–142) uses a chosen formatted run and its drone mocap to estimate each camera pose, then emits the generic upload calibration schema. _manifest_by_label retrieves source UUIDs; _default_upload_label maps camera1-style names to cam1; _parse_label_map accepts repeated source=target overrides (22–53). Export includes K/distortion/resolution, world-to-camera R/t, the fitted time offset, and provenance. It validates the whole camera list before writing.

### 2.5 Generic upload geometry and final reconstruction

**Calibration schema.** triangulate_uploads.validate_calibration (48–100) accepts a nonempty cameras list. Every entry needs a unique video label, 3×3 K, a rotation representation, and either translation t or camera position. Rotation can be a matrix R, Euler angles, quaternion [x,y,z,w], or Rodrigues vector rvec. Optional fields include distortion, resolution, start_epoch_s, and pose_convention.

rotation_matrix chooses R first, then Euler, quaternion, and rvec. euler_to_matrix uses degrees by default; roll, pitch, and yaw correspond to X/Y/Z, with default ZYX multiplication. world_to_camera gives position precedence over t. With position, default orientation is camera-to-world and is transposed; a w2c declaration avoids that transpose. With direct t, rotation is already treated as world-to-camera. This is important when manually entering camera poses (103–169).

Validation mainly checks structure. It is not a complete physical-validity check: the whole-document path does not enforce all rotation, finite-value, distortion-length, positive-resolution, and path-name constraints. rotation_issue checks orthonormality/determinant but is used by single-camera upsert, not automatically by the reconstruction loader (triangulate_uploads.py:172–178; webui.py:600–610).

**Camera loading and time.** UploadCamera contains K/distortion, R/t/P, centroids, FPS, and start time. _scaled_K adapts supplied calibration resolution; _load_camera reads the detection archive and constructs P = K[R|t]. _with_upload_metadata overrides the calibration clip start with the tracking clip's metadata.startTime in seconds, adding source.time_offset_s when present. Missing/unreadable metadata falls back to supplied start_epoch_s or zero (triangulate_uploads.py:181–259).

triangulate_uploads (262–365) matches calibration labels to detected cameras, reports skipped cameras, and requires at least two loaded cameras. Extra detections without calibration are ignored. It chooses the camera with the fewest centroid rows as the reference timeline; this is not necessarily the shortest duration when frame rates differ. Each reference epoch obtains interpolated observations from all cameras. There is no image-based synchronization in this path and no requirement that the resulting cameras actually have overlapping valid observations.

The geometry convention is:

~~~text
camera_position = -transpose(R) * t
camera_point = R * world_point + t
homogeneous_pixel = K * camera_point
pixel = homogeneous_pixel.xy / homogeneous_pixel.z
P = K * [R | t]
~~~

_distortion handling in final reconstruction uses OpenCV undistortPoints with P=K, keeping undistorted coordinates in pixel units. This makes them compatible with the projection matrix used by DLT (triangulate_527.py:579–582).

#### DLT and outlier selection, line by line

For a pixel (u,v) observed by camera projection P, the unknown homogeneous point Xh must satisfy:

~~~text
(u * P_row3 - P_row1) * Xh = 0
(v * P_row3 - P_row2) * Xh = 0
~~~

triangulate_point_dlt (triangulate_527.py:514–524) stacks two such equations for every camera. It uses SVD, a matrix factorization, to find the least-error homogeneous solution and divides XYZ by W to obtain ordinary coordinates. There is no explicit near-zero-W or minimum-baseline guard.

select_best_triangulation (539–562) performs these steps:

| Lines | Behavior and implication |
|---|---|
| 542–543 | Fewer than two observations cannot produce a point. |
| 545–548 | Try every camera subset of size two or more. |
| 549–550 | Reject points behind any selected camera. |
| 551–557 | Calculate selected/all-view reprojection errors and reject a selected-view error above the threshold. |
| 558–562 | Rank by all-view median error, then selected-view mean error, then number of views; return the best candidate or no point. |

More views are a tie-breaker, not an unconditional preference. For C cameras there are 2^C−C−1 subsets, acceptable for the intended two/three-camera recordings but expensive for large camera arrays. A two-camera fit can win despite disagreement with another camera.

interpolate_centroid converts the requested epoch into a fractional frame and linearly interpolates adjacent centroids (565–576). It currently requires both neighbors even at an exact integer frame, which unnecessarily discards a valid frame immediately preceding a missing one.

#### Output orchestration and errors

triangulate_run and triangulate_raw_run load cameras, create frame epochs, reconstruct each point, smooth, sample nearest mocap ground truth, score, and write artifacts. Raw mode nests results under mocap_raw/. run_triangulation/run_raw_triangulation choose available or configured runs, load or create calibration caches, and collect per-run summaries (triangulate_527.py:664–819,839–951). Existing calibration caches are reused without input fingerprints or automatic completion of missing configurations.

Generic uploads write the same main NPZ keys, with gt_drone filled with NaN because there is no mocap ground truth. Dataset output additionally has mocap_times. The arrays are:

~~~text
times_s, epoch_times_s                  (N,)
trajectory_raw, trajectory_smooth      (N,3)
gt_drone                               (N,3)
n_views                                (N,)
reproj_errors_px                       (N,)
used_cameras                           (N,) strings joined with "+"
~~~

Dataset times_s is relative to mocap start; upload times_s is relative to the earliest loaded camera start. Do not assume a shared relative-time origin across modes. Upload summary includes effective camera starts and a calibration snapshot so later comparisons can use the geometry that actually produced the reconstruction (triangulate_uploads.py:302–365).

write_csv writes frame/time, view count, reprojection error, used-camera names, raw xyz, smooth xyz, and ground-truth xyz. Nonfinite values are blank; positions/time are rounded to four decimal places (triangulate_527.py:822–836). NPZ preserves more precision.

Dataset wrappers catch per-run exceptions and record error summaries, while calibration failures before the run loop abort the command. The CLI still returns zero after a wrapper returns, even if some run summaries contain failures (triangulate_527.py:891–905; cli.py:74–85). Check summaries rather than relying solely on process exit status.

### 2.6 Smoothing: what ekf.py actually does

Despite its name, ekf.py implements a **linear** Kalman filter followed by an RTS backward smoother. A Kalman filter combines a motion prediction with noisy observations; the backward pass uses future observations to improve earlier estimates. RTS refers to the Rauch–Tung–Striebel smoothing equations. This is offline trajectory processing, not a live drone controller.

The state is [x,y,z,vx,vy,vz]. _build_transition advances position by velocity × dt. _build_process_noise uses integrated white-acceleration covariance with per-axis terms q×dt³/3, q×dt²/2, and q×dt. _H selects the position coordinates. _measurement_noise uses a heuristic standard deviation:

~~~text
sigma = base_sigma * (2 / max(n_views,2)) * (1 + 0.01 * reprojection_error_px)
measurement_covariance = sigma² * identity(3)
~~~

This rewards more views and lower reprojection error but does not model baseline geometry or the full uncertainty of intersecting rays (ekf.py:15–68).

ekf_smooth_trajectory (71–205) takes raw (N,3) points, times, optional view counts, and errors; returns only smoothed positions. Default process scale is 2, base measurement sigma 0.15 m, maximum prediction-only gap 30 frames.

| Lines in ekf.py | Explanation |
|---|---|
| 94–107 | Allocate filtered/predicted states, covariance matrices, and transitions for every time. |
| 113–126 | Initialize at the first detected position with zero velocity. |
| 142–154 | Compute dt, substitute 1/30 s for nonpositive dt, and predict state/covariance. |
| 156–164 | Continue motion prediction through at most 30 missing observations, then leave missing state and reset. |
| 167–179 | Correct the prediction using observed position and Kalman gain. |
| 181–198 | Run backward through valid neighboring states, using the next smoothed state to refine the current one. |
| 200–205 | Return position components; retain NaN where state is unavailable. |

For clarity, the update is residual = z−Hxp, S = HPpHᵀ+R, gain = PpHᵀS⁻¹, corrected state = xp+gain×residual. The backward gain is Pf Fnextᵀ Pp,next⁻¹. Forward inverse errors propagate; a singular backward inverse is skipped.

Smoothing can fill or extrapolate up to 30 frames, including after the last detection. Consequently a finite smoothed point may have n_views=0 and no raw reprojection error. Smoothed-point count is not detection coverage. Initial gaps and longer unsupported regions remain missing. Inputs are assumed to have consistent shapes, meter-scale coordinates, and whole-row missing values; comprehensive input validation is absent.

### 2.7 Web server, upload handling, jobs, and annotations

**Workspace configuration.** WebConfig is a frozen record holding work_root; properties derive every persistent location (webui.py:76–142). There is no database.

~~~text
work/
├── dataset_uploads/                        Raw mocap recordings
├── formatted/                              Rotated/transcoded dataset clips
├── uploads/<camera>/                       Calibration flight videos and sidecars
├── tracking_uploads/<run>/<camera>/         Named tracking flight inputs
├── calibration_logs/                       Calibration/reference BIN files
├── tracking_logs/<run>/                    Older separately uploaded reference logs
├── detections/<source-or-run>/<camera>/    Centroid archives and summaries
├── triangulation/<source-or-run>/          Trajectories and comparison sidecars
├── classification/tracking_uploads/<run>/prediction.json
├── uploads_calibration.json                Current reusable calibration
├── clicks.json / mocap_raw_clicks.json      Dataset annotations
├── uploads_clicks.json                      Calibration annotations
├── tracking_uploads_<run>_clicks.json        Tracking annotations
└── webui_state.json                         Selected tracking run
~~~

The actual_uploads directory and historical source name actual are recognized for compatibility. sources.tracking_result_path changes public naming to tracking_uploads without moving files. An older flat upload layout is exposed as run default; current named directories take precedence over legacy duplicates (sources.py:20–36; webui.py:254–318). _read_state tolerates invalid/missing selection JSON; _write_state writes directly.

**Source adapters.** sources.py:39–110 adapts formatted dataset, raw mocap, and generic uploaded videos into shared VideoSpec or ClickVideo records. Generic discovery searches one camera directory level for video.mp4; it does not itself validate video content or sidecars. This allows the detector and annotator to work with multiple input organizations without implementing separate neural pipelines.

**Upload mapping.** safe_stem/safe_relpath sanitize names. dataset_store_rel and infer_dataset_run map allowed dataset files into the duplicate-run layout. nomocap_store_rel keeps only the immediate camera-parent name and an allowed filename, so an archive should contain one logical flight with unique camera folder names (webui.py:145–213). normalize_nomocap_videos merely probes and indexes native videos into manifest.json; unlike dataset formatting, it does not rotate, resize, or re-encode (216–240).

Upload handlers read raw HTTP bodies, not multipart forms. ZIPs are staged under work_root, filtered, and extracted with containment checks for camera/dataset destinations, then temporary archives are removed. Extraction is not transactional; earlier files can remain after an error. Ordinary uploads and ZIP extraction overwrite existing paths directly. Single-log upload clears existing calibration logs; calibration ZIP import instead retains/overwrites individual named BINs (webui.py:1161–1376). The separate reference-log upload uses a temporary file and checks upload completeness (1390–1423).

**Browser/server contract.** The main routes are:

| Route family | Responsibility |
|---|---|
| GET /, /clicks, /results | Embedded dashboard, annotator, and result documents. |
| GET /api/status, /api/log | Source readiness, environment hints, job progress, and log polling. |
| GET /api/results, /api/centroids, /api/trajectory | Result inventory and cleaned numerical payloads. |
| GET /api/result-frame, /api/upload-frame, /frame | JPEG extraction from the corresponding source video. |
| GET /api/trajectory/download, /api/calibration, /api/rows | Artifact downloads, calibration JSON, active annotation rows. |
| POST /api/run | Validate and start a named pipeline stage. |
| POST /api/upload, /api/calibration-log, /api/dataset-upload-zip | Video, BIN, and dataset upload. |
| POST /api/nomocap-format-zip, /api/nomocap-upload-zip, /api/nomocap-format | No-Mocap import/indexing; ZIP routes largely duplicate logic. |
| POST /api/tracking-run/select, /api/tracking-run/clear-outputs | Persist a selection or clear selected generated outputs. |
| POST /api/tracking-run/camera/{clear-outputs,detect}, /api/calibration/camera/{clear-outputs,detect} | Scoped camera repair. |
| POST /api/tracking-run/reference-log, /api/tracking-run/reference | Add a reference log or compute comparison. |
| POST /api/calibration, /api/calibration/camera, /api/calibration/clear | Replace/merge calibration or clear calibration inputs and outputs. |
| POST /api/upload/remove, /api/dataset/clear, /api/centroids/edit | Direct persistent mutations. |
| POST /api/clicks/select, /api/click, /api/skip | Select and edit annotation state. |

Routes and handlers: webui.py:906–1598. Jobs return 202 when accepted and 409 when already busy; actual processing errors appear later in job status/logs. Result handlers convert many invalid/missing requests to 400/404, but there is no uniform exception boundary around JSON parsing. A successful job submission is not proof of successful reconstruction.

**Stage dispatch.** _stage_fn (640–694) maps four backend sources to operations: dataset supports format/detect/triangulate/run-all; mocap_raw supports detect/triangulate; upload supports detect/calibrate/triangulate; tracking supports detect/triangulate/classify. The corresponding wrapper functions discover the right cameras and click files and pass configuration to the computation modules (697–833). Tracking triangulation automatically attempts an optional reference comparison, then clears stale classification after successful reconstruction.

**Background jobs.** JobState holds name, status, error, lines, and progress. _LineWriter splits print output into lines. JobRunner.start atomically refuses a second job, resets state, starts a daemon thread, redirects stdout, and marks completion or captures the final traceback lines on failure. status and log_since return lock-protected snapshots (jobs.py:18–118). State/history is not persisted; process exit can interrupt a job; there is no resume or cancellation API. Stdout redirection is process-wide, so it is not strictly isolated to the worker thread. Only mutations dispatched through this runner share its exclusion.

**Annotation service.** build_manifest_from_videos joins saved clicks by label to current clips and selects a default viewing frame at 40% of the clip. FrameCache locks a single OpenCV capture, seeks and JPEG-encodes frames, and releases it when switching videos. ClicksService creates/saves a manifest as soon as the annotator is opened; apply_click stores native-pixel coordinates, frame, and approval; toggle_skip flips approval (clicks.py:15–128). Browser clicks are converted from displayed dimensions to original image dimensions in the embedded JavaScript (140).

_effective_clicks avoids treating an empty newly created manifest as usable initialization: it returns None until there is at least one usable approved click. At that point detector rules skip cameras missing clicks (webui.py:321–332; detect.py:355–366). A repair/redetect action is stricter: it requires an approved click for the selected camera **before deleting anything** (webui.py:1475–1489).

**Clearing and repair.** tracking_output_targets and calibration target helpers validate membership/symlink boundaries before removing generated files. Per-camera tracking repair removes that camera's detections plus the whole run's reconstruction/classification. Calibration camera repair removes its detections and invalidates saved calibration. Scoped clear actions preserve their documented unrelated inputs and use the job runner (webui.py:520–597,1448–1514). The older dataset-clear route has broader and inconsistent behavior, discussed in section 5.

**Dashboard behavior.** dashboard.py contains one bytes constant with CSS, markup, and JavaScript. Its three tabs are No-Mocap, Mocap Raw, and Mocap; No-Mocap combines calibration and named tracking panels. Stage cards and controls derive from status API fields; status refresh runs every 2.5 seconds and job logs poll about every second. Uploads use XMLHttpRequest for progress; calibration can be automatically downloaded after a user-started successful solve; embedded annotator/results URLs are updated as selections change (dashboard.py:238–387,397–420,625–658,717–838). Google Fonts is an external browser request despite embedded page assets; fallback fonts exist.

The server has no application authentication or TLS and defaults to binding all interfaces through the CLI. Upload, delete, and edit endpoints make that a concrete deployment property. The README's Docker example maps the port only to host loopback; a similarly local bind is appropriate for the commands below. GPU/cache indicators are filesystem heuristics, not actual inference tests (webui.py:613–630).

### 2.8 Results, manual correction, and playback

results.py joins artifacts to browser-friendly payloads. _safe_subdir rejects result paths escaping their root and resolves historical tracking names. list_detections skips unreadable/missing-summary artifacts; list_trajectories finds trajectory archives. load_centroids returns dimensions, FPS, optional metadata start time, and null-cleaned centroids; video_path_for caches source paths (23–161).

apply_centroid_edits changes or clears selected frame coordinates, rewrites the centroid archive, and updates detection counts in summary.json. It validates frame range and finite coordinates but not image bounds; it does not automatically retriangulate or invalidate downstream results (164–200). trajectory_file resolves supported CSV/NPZ/comparison downloads and refuses a stale comparison. load_trajectory returns raw/smooth/reference arrays, camera starts, quality metrics, and at most about 3000 plotted samples; downloads retain full data (203–266). The module's “read-only” introductory description is stale because it now contains edits.

results_page.py renders all selected camera frames with centroid overlays and draws a rotatable 3D-looking view using ordinary 2D canvas projection, not a 3D graphics library. setupView/project/poly/axes/drawScene implement centering, yaw/pitch, projection, lines, and markers (248–301). The shared playback logic uses camera starts plus FPS, chooses overlapping viewing time, loads all images before updating the scene, and locates the nearest plotted trajectory time by binary search (338–425). This is image-by-image playback, not direct HTML video streaming.

The editor accumulates changed frame positions, posts them to /api/centroids/edit, and tells the user to rerun triangulation (213–225). runVersion and request counters reject stale asynchronous loads during run changes (149–155,380–394,457–465). The shared edit map has a separate cross-camera bug described below.

### 2.9 Optional flight-log comparison and classification

**Reference comparison.** flight_reference.sample_track converts log positions into the saved frame and interpolates only within supported intervals, without bridging gaps above one second. comparison_metrics reports compared counts, coverage, 3D RMSE, median error, horizontal RMSE, and vertical RMSE, excluding rows missing either side (108–118,147–158).

attach_reference hashes the completed trajectory, validates finite increasing epochs, prefers the calibration snapshot in its summary, computes log query times using saved lag, and requires exactly one supplied log with usable overlap. It writes separate reference NPZ/JSON and comparison CSV files, leaving trajectory.npz unchanged. If the comparison overlaps the calibration interval, it warns that this is calibration consistency rather than independent validation. A second hash check detects reconstruction changes during work; load_reference later validates hash, shape, and epoch identity (161–254).

webui._tracking_reference tries stored calibration logs first, uses older per-run logs only when none are stored, removes stale sidecars, and turns comparison failure into an error sidecar without failing a valid reconstruction (777–804). “No reference” therefore does not imply “no trajectory.”

**Model preparation.** classify.load_model_manifest validates nonempty model fields, a full revision SHA, and a SHA256 digest. _download_model retrieves the pinned Hugging Face file and checks its actual bytes. build_classifier_predictor constructs the external TrajectoryPredictor on CPU by default with batch size 128 (classify.py:15–90).

scripts/prepare_classifier.py downloads a pinned external wheel, verifies its checksum, installs with --no-deps if the package is absent, then tries cached-model retrieval before online retrieval (14–56). An already installed package is not checked for the pinned version. The Docker build invokes this script; ordinary pip install -e . does not. The model manifest is inspected metadata; the downloaded wheel, network architecture, training data, and learned weights were not available for review.

**Run service.** ClassificationService owns a reusable predictor and per-run prediction files. preload deliberately calls build_classifier_predictor with local_files_only=True and catches failure so the rest of the UI remains usable. It does not automatically fetch missing weights merely because the browser server starts (classification.py:41–69).

status hashes the trajectory, checks at least ten finite samples with distinct timestamps, verifies a successful triangulation summary with raw points, and only accepts saved predictions whose trajectory and model hashes match. _inspect_trajectory caches shape/readiness checks by path and digest. run invokes predict_trajectory(path, measurement_type="vision"), checks that the trajectory did not change, then atomically publishes prediction.json (20–38,92–150). It expects the external result to carry model_sha256; FireTrack itself adds trajectory_sha256.

The dashboard labels this as classifying ArduPilot versus PX4 (dashboard.py:419). The mock tests use PX4 outputs, but this checkout alone does not establish the external model's accuracy, windowing rules, confidence calibration, or supported operating conditions.

## 3. End-to-End Example

This is an **illustrative No-Mocap workflow**, traced through the source. No real videos, flight logs, model inference, calibration fit, or resulting trajectory were executed during this review.

Suppose two stationary cameras record a calibration flight and a later tracking flight. They remain in the same places, keep the same lens/video settings, and already share synchronized timestamps.

### 3.1 Inputs

A calibration ZIP can contain:

~~~text
calibration_flight/
├── cam1/
│   ├── video.mp4
│   ├── camera.json
│   └── metadata.json
├── cam2/
│   ├── video.mp4
│   ├── camera.json
│   └── metadata.json
└── calibration.BIN
~~~

For a hypothetical 1920×1080 decoded video, explicit camera.json could contain:

~~~json
{
  "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
  "dist": [0, 0, 0, 0, 0],
  "resolution": [1920, 1080]
}
~~~

These numbers are explanatory, not a calibration to copy into a real camera. metadata.json needs a genuine UTC microsecond startTime, such as the illustrative value 1760000000000000. A .BIN must be an actual compatible ArduPilot log containing valid timing and POS messages. Calibration motion must provide enough nondegenerate spatial/time variation; an arbitrary straight-line flight may not identify camera pose and lag reliably.

### 3.2 Follow one flight through the application

| Step | Code involved | Expected persistent result |
|---|---|---|
| Upload calibration ZIP | dashboard.uploadNoMocapZip → webui ZIP handler → normalize_nomocap_videos | uploads/cam1 and cam2, calibration_logs/*.BIN, indexed manifest. |
| Click the drone in each camera | _ClicksRegistry.select → ClicksService.apply_click | uploads_clicks.json with native x/y and frame numbers. |
| Detect calibration flight | _handle_run → _upload_detect_fn → run_detection_on_specs → run_sam3_on_video | detections/uploads/cam1 and cam2 centroid archives and summaries. |
| Calibrate | _upload_calibrate_fn → calibrate_from_log → fit_joint_calibration | uploads_calibration.json with both camera poses, focal lengths, common lag, provenance, and diagnostics. |
| Upload later tracking ZIP | Same import family with tracking bucket/run | tracking_uploads/demo_flight/cam1 and cam2; selected run persisted. |
| Annotate and detect tracking flight | _tracking_detect_fn plus detector | tracking_uploads_demo_flight_clicks.json and detections/tracking_uploads/demo_flight/<camera>/. |
| Reconstruct | _tracking_triangulate_fn → triangulate_uploads → interpolate_centroid → undistort_pixel → select_best_triangulation → ekf_smooth_trajectory | triangulation/tracking_uploads/demo_flight/trajectory.npz, trajectory.csv, summary.json. |
| View/compare/classify | results.py/results_page.py; optional attach_reference and ClassificationService.run | Browser visualization, optional comparison files, optional classification/.../prediction.json. |

Routing references: webui.py:721–819,1123–1159,1306–1360. Geometry references: triangulate_uploads.py:262–365 and triangulate_527.py:514–582.

If calibration estimated camera_minus_log_s=0.237, a later camera sample at epoch T+60.000 remains at T+60.000 in reconstruction. Optional comparison queries the log at T+59.763. It does not shift one camera relative to the other. If the only stored BIN covers the calibration flight, reference comparison for the later flight can fail while reconstruction succeeds.

At one time sample, imagine two equal-intrinsic cameras whose centers are one meter apart in x, with the drone five meters forward of the first camera. In a local frame with identity rotations, the point [0,0,5] projects through the example K to [960,540] in camera one and [760,540] in camera two. Ideal undistortion changes nothing, DLT yields [0,0,5], and both selected views have zero reprojection error. This arithmetic explains the geometry only; it is not an observed output, and a geographic calibration would express the same geometry using its saved world origin and altitude convention.

The final CSV would contain frame/time, camera count, reprojection error, and raw/smoothed xyz. Missing raw samples are blank there and NaN in NPZ. Smoothing may fill short missing intervals. Classification is available only after a usable trajectory and cached CPU model exist; its numerical prediction cannot be inferred from this example.

### 3.3 Repository-supplied synthetic example

tests/test_calibration_integration.py:23–80 constructs two fake camera folders, empty video marker files, 240 synthetic centroids at 20 Hz, explicit 1080×1920 intrinsics with 905 px focal length, and a curved known flight near 150 m AMSL with 0.237 s lag. It replaces the flight-log loader, runs the calibration/export boundary, and checks later tracking timestamps are preserved. This is a useful executable template once dependencies are installed, but it does **not** decode video or parse a real BIN, and it was not run here.

## 4. How to Run and Modify the Repository

### 4.1 Prerequisites and environment

The declared runtime dependencies are CasADi >=3.6, NumPy >=1.26, opencv-python-headless >=4.9, pymavlink >=2.4, SciPy >=1.10, PyTorch >=2.6, SAM3 ==0.1.2, and huggingface-hub >=0.30,<1.0. Setuptools >=69 and wheel build the package (pyproject.toml:1–20). Only SAM3 is exactly pinned among those libraries; there is no lockfile.

The metadata says Python >=3.9, but that is not proof that every permitted interpreter/dependency combination is installable. The Dockerfile supplies a specific CUDA/PyTorch base. Local syntax compilation on Python 3.14.7 does not establish runtime support on 3.14.

Additional prerequisites depend on the operation:

- FFmpeg must be on PATH or selected using FIRETRACK_FFMPEG for dataset normalization.
- Detection needs SAM3 weights and access to the gated facebook/sam3 model through an authorized Hugging Face cache or credential. No credential is included in these reports.
- The documented GPU container path needs working NVIDIA GPU/container support; GPU inference was not tested here.
- Classification needs the separate fire-moonshot-classifier wheel and cached pinned model; these are not installed by the normal runtime dependency list.
- Actual videos, metadata, intrinsics, and either poses, a calibration log, or suitable mocap recordings must be supplied.

### 4.2 Local development commands

Run from the repository directory. These commands follow the package layout; they were not run as installs during the inspection.

~~~powershell
cd C:\Users\gmato\Desktop\Research\fireRepo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m firetrack --help
.\.venv\Scripts\python.exe -m firetrack webui --work-root ..\firetrack-work --host 127.0.0.1 --port 8080
~~~

Use a working compatible Python executable in place of python if the Windows alias is unavailable. The editable install makes code changes visible without rebuilding the package. It installs runtime packages; it does not fetch all gated model assets in advance.

For optional classification, after installation:

~~~powershell
.\.venv\Scripts\python.exe scripts\prepare_classifier.py
~~~

This command can download/install the pinned wheel and model. Restart the web process afterward if its classifier preload previously failed. The error message suggesting that simply starting the app online will populate the model cache does not match the current local_files_only=True preload path (classify.py:50–54; classification.py:62–64).

### 4.3 Docker

The README advertises ashreeku/firetrack:latest; remote availability and contents were not verified. This Bash form closely follows README.md:18–28:

~~~bash
mkdir -p firetrack-work
docker run --rm -it --gpus all \
  -p 127.0.0.1:8080:8080 \
  -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf \
  -v "$PWD/firetrack-work:/work" \
  ashreeku/firetrack:latest webui --work-root /work --host 0.0.0.0 --port 8080
~~~

Building this checkout would normally use:

~~~bash
docker build -t firetrack-local .
~~~

**Current build blocker:** Dockerfile:40–42 copies vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz, but that file is absent. If earlier build steps succeed, that copy cannot succeed from this checkout as supplied. The repository contains no procedure that creates this vocabulary asset. Obtain the correct asset or resolve its packaging before treating a local Docker build as reproducible. The build also downloads the external classifier wheel/model; it is not an offline build.

### 4.4 Supported command-line workflows

Dataset staging:

~~~bash
firetrack format --data-root /data/raw --out-root /work/formatted
firetrack clicks serve --data-root /work/formatted --clicks-json /work/clicks.json --host 127.0.0.1 --port 8080
firetrack clicks status --clicks-json /work/clicks.json
firetrack detect --data-root /work/formatted --out-root /work/detections --clicks-json /work/clicks.json
firetrack triangulate --raw-root /data/raw --formatted-root /work/formatted --detections-root /work/detections --out-root /work/triangulation
~~~

Stop the annotation server after saving clicks if reusing its terminal/port. Omitting --clicks-json on detect selects text-only initialization. --only accepts repeatable camera labels; --run accepts repeatable dataset run names. --overwrite on detect recalculates existing outputs. triangulate --calibrate-only rebuilds and saves calibration; --calibration-json chooses the calibration cache path (cli.py:183–223).

Log calibration for existing uploaded camera folders and completed detections:

~~~bash
firetrack log-summary --log /work/calibration_logs/calibration.BIN --run-root /work/uploads
firetrack calibrate-from-log --log /work/calibration_logs/calibration.BIN --run-root /work/uploads --detections-root /work/detections --out-json /work/uploads_calibration.json
~~~

Mocap-to-upload calibration export:

~~~bash
firetrack calibrate-from-mocap --raw-root /data/raw --formatted-root /work/formatted --detections-root /work/detections --run ardu_run1 --out-json /work/uploads_calibration.json --label-map camera1=cam1
~~~

run-all combines only the older format/detect/triangulate sequence, with skip and overwrite flags. The CLI triangulate command calls the mocap dataset pipeline even when --calibration-json is supplied; its cache schema differs from uploads_calibration.json. There is no dedicated CLI subcommand for generic upload triangulation, flight-reference attachment, or classification. Those paths are exposed through the dashboard and importable Python functions (cli.py:138–173,214–283).

### 4.5 Environment settings and configuration changes

| Setting | Role |
|---|---|
| --work-root | Persistent input/output state; default /work is container-oriented. |
| --host / --port | HTTP bind address and port; default 0.0.0.0:8080. |
| FIRETRACK_FFMPEG | FFmpeg executable; Docker sets /usr/bin/ffmpeg. |
| FIRETRACK_REUSE_SAM_PREDICTOR=1 | Reuse one SAM predictor across videos rather than recreate it. |
| HF_HOME | Hugging Face cache location; Docker defaults under /work, README overrides through a mount. |
| HF_HUB_OFFLINE | Displayed by status; underlying Hugging Face behavior belongs to the external dependency. |
| FIRETRACK_CLASSIFIER_CACHE | Classifier model cache; Docker sets /opt/firetrack/classifier-cache. |

Do not move cameras, change focal/zoom/crop/resolution assumptions, or reuse labels for another arrangement while expecting old calibration to remain valid. There is no general automatic calibration-freshness detector. For algorithm settings, see section 2.3's CLI table and the numerical-function defaults.

### 4.6 Tests and actual verification

All tests use unittest style and temporary files/mocks; pytest configuration is present but pytest is not a runtime dependency. With dependencies installed:

~~~bash
python -m unittest discover -s tests -v
~~~

Alternatively install pytest into the development environment and run python -m pytest. The suite imports numerical/video modules even for many filesystem tests.

| Test file | Methods | Intended checks |
|---|---:|---|
| test_calibration_solver.py | 11 | Pose/fractional lag, missing/outlier observations, analytic derivatives, line/circle degeneracy, fixed timing, priors, invalid settings, spline gaps/smoothing. |
| test_joint_calibration_solver.py | 3 | Three-camera shared lag/pose/focal recovery, fixed mean focal, duplicate names and unsupported distortion. |
| test_calibration_integration.py | 8 | File/export integration, preserving prior calibration on failure, explicit intrinsics, missing files, run isolation, UTC/frame conversion, log ambiguity, sensor conventions. |
| test_flight_reference.py | 10 | Geographic axes/altitude, origins, gaps, unchanged reconstruction, lag sign, freshness hashes, ambiguous logs, snapshots, missing-data metrics, mocked GPS time conversion. |
| test_automatic_flight_reference.py | 5 | Reconstruction/reference sequencing, stored/legacy logs, optional-reference failures, stale-sidecar cleanup. |
| test_calibration_clear.py | 5 | Calibration-only deletion, repeatability, symlinks, annotation cache invalidation, busy runner. |
| test_camera_repair.py | 6 | Selected-camera clearing/redetection, preserved inputs, legacy/default layouts, invalid paths, busy state, required clicks. |
| test_tracking_output_clear.py | 4 | Selected-run deletion, invalid paths/symlinks, flat default-run isolation, busy endpoint. |
| test_tracking_names.py | 5 | Current/legacy discovery, path/download names, unchanged payloads, scoped clears, old source alias. |
| test_tracking_classification.py | 4 | Mocked prediction persistence, changed-trajectory invalidation, clear behavior, cached CPU predictor construction. |

Total: **61 test methods**, counted from Python syntax trees, not a test-run result.

Important gaps: no full real video/BIN/mocap/model fixtures; no SAM inference test; no browser-driven test; no full final-triangulation/smoother regression suite; no joint fixed-lag **export** test; no ring-buffer overflow regression. Large outlier robustness is tested mainly on the older single-camera solver, not the current joint engine. The classifier-preload test mocks construction but leaves package find_spec unmocked, so it also needs the external classifier installed (test_tracking_classification.py:77–85; classification.py:59). Some camera-repair expected path strings are POSIX-specific and can fail on Windows.

Executed inspection checks were deliberately limited:

| Check | Result |
|---|---|
| Compile all 39 .py files in memory with bytecode disabled | Passed; dashboard.py:462 emitted an invalid-escape SyntaxWarning. |
| Parse pyproject.toml with tomllib | Passed. |
| Call classify.load_model_manifest | Passed without fetching weights. |
| Count test methods from AST | 61 found. |
| Append 5000 job-log lines, read cursor, append another line, poll again | Reproduced missing new line: returned lines=[], next=5000. |
| json.dumps(NaN, allow_nan=False), the serialization mode used at calibrate_logs.py:361 | Raises ValueError (“Out of range float values are not JSON compliant”). |
| str() of a relative Windows path, as compared in test_camera_repair.py:46 | Produces backslash separators, so the test's slash-separated expected strings cannot match on native Windows. |
| Probe dependencies with find_spec | NumPy, SciPy, OpenCV, CasADi, pymavlink, PyTorch, SAM3, Hugging Face Hub, classifier, and pytest all absent. |
| SHA-256 of all 47 repository files before and after the work | Identical; no __pycache__ directories were created. |

Git, Docker, and Node.js were not found on PATH; the `python` command resolved to Python 3.14.7. No package installation, external-model download, server launch, Docker build, or full test run was attempted. These omissions prevent runtime compatibility and numerical-accuracy claims.

### 4.7 Where to make common changes

| Desired change | Primary files and connected changes |
|---|---|
| Add a CLI option/action | cli.py parser and handler; update documentation and the owning function's parameter handling. |
| Add a dashboard stage | webui.py stage sets, _stage_fn, readiness/status and handler; dashboard.py controls/progress; define output invalidation. |
| Support another input layout/device | sources.py and upload mapping; dataset_527.py for legacy model/rotation rules; update naming/calibration assumptions. |
| Change detection prompting/object selection | detect.py; keep click schema in clicks.py and repair requirements consistent. |
| Change calibration variables or loss | joint_calibration_solver.py model, residual/Jacobian, bounds, validation, covariance; calibrate_logs.py export and CLI settings. |
| Add a measured prior or log format | calibration_priors.py/Priors or flight_reference.py log/coordinate parsing; preserve units and clock conventions. |
| Change reconstruction/outlier strategy | Shared triangulate_527.py helpers plus upload/dataset loops; check duplicate raw-mode implementation. |
| Change motion model or gap policy | ekf.py; make effects on finite-output counts and uncertainty explicit. |
| Change result schema/view | Both reconstruction writers, shared write_csv, results.py/results_page.py, classification/reference consumers, and tests. |
| Fix clear/repair behavior | webui.py validated target helpers and preserving-unrelated-files tests. |
| Replace classifier | classify.py manifest/API wrapper, preparation script, classification.py persistence/readiness; audit external result schema. |

Use the existing synthetic geometry tests to check mathematical changes, and add regression tests at the failing behavior boundary for the findings below. Input/output schema changes often cross several modules because there is no central versioned schema layer.

## 5. Findings and Open Questions

The priority here is helping a new maintainer avoid misleading results or lost work. Unless marked “reproduced,” these findings are established by source/control-flow inspection and have not been exercised in a browser or full numerical run.

### 5.1 Concrete defects and operational problems

| Finding | Evidence and consequence | Suggested direction |
|---|---|---|
| **Clear dataset deletes other workflows' outputs.** Source-confirmed. | dashboard.py:778–780 asks to clear uploaded 5-27 files and generated dataset outputs; webui.py:1516–1524 removes the entire detections and triangulation roots, including calibration and named tracking results. It bypasses JobRunner. | Use source-scoped validated deletion targets and the shared job lock; add preservation tests covering all sources. |
| **Unsaved manual edits can be applied to the wrong camera.** Source-confirmed. | results_page.py:138 keeps one frame-keyed edit map; 227–234 switches camera with keepEdits=true; 213–216 posts every pending edit to the latest det.dir. Edit cam1 frame 5, then cam2 frame 7, then Save can write both to cam2. Same-frame edits can overwrite each other in the map. | Store pending edits per camera or require saving/discarding before switching. |
| **Fixed-lag joint calibration cannot export its diagnostics.** Source-confirmed. | Radius zero freezes lag (joint_calibration_solver.py:373–374). Uncertainty starts NaN and is populated only for free parameters (578–580); shared_clock_delta_std_s_local includes the fixed lag's NaN (604). calibrate_logs.py:353,361 copies it into strict JSON with allow_nan=False, which rejects NaN. Existing calibration is not replaced because serialization precedes replacement. | Represent fixed-parameter uncertainty as null, as the old solver does, and add an integration test that writes fixed-lag calibration. |
| **Docker build requires a missing file.** File absence verified. | Dockerfile:40–42 copies vendor/sam3_assets/bpe_simple_vocab_16e6.txt.gz; only the package marker exists. .gitignore:49 excludes these gzip files. | Supply/document asset preparation or change packaging so the build resolves it explicitly. |
| **Dashboard polling resets camera selections.** Source-confirmed. | dashboard.py:586–591,625–630 replaces camera list HTML with newly unchecked inputs every 2.5 seconds (838). Empty selection means all cameras (webui.py:247–251), so lost selection can change processing scope. | Preserve selected labels across refreshes; test the browser behavior. |
| **Job log polling stalls after 5000 lines.** Reproduced. | jobs.py:58–62 caps list length, while log_since uses length as cursor (107–118). After cursor=5000, new lines replace old ones but polling still returns an empty slice with next=5000. | Use a monotonically increasing sequence number and retained-buffer offset. |
| **Exact-frame detections can be discarded.** Source-confirmed. | triangulate_527.py:565–576 requires both adjacent detections even when interpolation fraction is exactly zero. A valid frame followed by a missing frame is lost at its own timestamp. | Handle exact samples before requiring two valid neighbors, as the joint-calibration interpolator already does. |
| **A tracking clip without timing metadata silently inherits the calibration clip's start time.** Mechanism source-confirmed; consequence inferred. | calibrate_logs.py:338 stores each calibration clip's start_epoch_s in the exported camera entry. _with_upload_metadata (triangulate_uploads.py:241–259) replaces it only when the tracking clip's metadata.json exists and contains startTime; otherwise the calibration-flight start is used (226). That camera is then placed at the wrong absolute time: it either never overlaps the others or is misaligned if they also lack metadata. No warning is printed. | Require startTime for tracking clips, or at least warn when falling back, and never reuse a calibration-flight start for a different recording. |
| **Tracking cameras are matched to calibration by folder name only.** Source-confirmed. | triangulate_uploads.py:279–293 loads a camera only when its calibration `video` label equals a detected tracking folder name; anything else is skipped with a console note. Renaming cam1 to camA in the tracking ZIP drops that camera, and swapping two folder names silently swaps their poses. | Validate that tracking labels equal calibration labels before detection, and surface mismatches in the dashboard rather than only the log. |
| **Annotator source is shared across clients/tabs.** State design confirmed; corruption scenario inferred. | _ClicksRegistry.active changes on any clicks/select request (webui.py:852–872,1110–1113); frame/click requests identify row index but not source/run (1034–1041,1577–1598). Another tab can redirect the active collection. | Include source/run/session identity in annotation requests and service lookup. |
| **Input edits do not consistently invalidate downstream artifacts.** Source-confirmed. | Detection reuse is based on file existence (detect.py:158–162,231–243). Centroid edits overwrite detections but retain trajectories/predictions (results.py:164–200; webui.py:1537–1551). Upload/calibration/remove handlers bypass the job lock, unlike camera repair. | Define provenance fingerprints and dependency invalidation, or consistently require explicit recomputation. Prediction hashes only detect changed trajectories, not unchanged trajectories with changed upstream inputs. |
| **Mocap missing rotations are not masked.** Source-confirmed documentation discrepancy. | mocap.py:187–190 promises missing rigid-body pose masking, but 214–218 masks position/Euler only; rotation is built from original values. Untracked zero rotations can survive. | Apply the untracked mask to rotation matrices; add a loader fixture. |
| **Native Windows test portability problem.** Source-confirmed condition. | tests/test_camera_repair.py:44–49,55–58 compares str(relative Path), which uses Windows backslashes, with slash-based expected strings. | Compare Path objects or normalize with as_posix(). |
| **Classifier preparation is a separate prerequisite.** Source-confirmed. | Normal package dependencies omit the classifier wheel; preload requires it and uses cached assets only (classification.py:59–64). The preload unit test does not mock the presence check (test_tracking_classification.py:77–85). | Document/install optional classifier preparation explicitly or isolate the unit test from package discovery. |

Several other concrete fragilities deserve attention during related changes: _safe_click_status does not catch malformed JSON (webui.py:633–637; clicks.py:213–216); general request JSON has no central error boundary (webui.py:902–904); remove_upload rewrites calibration as only a cameras list, discarding top-level frame/clock/report fields (503–506). New calibrations duplicate some essential frame/lag values in per-camera source, so metadata loss does not necessarily break every current comparison, but manually supplied top-level-only metadata can be lost.

A text search of the three embedded pages (dashboard.py, results_page.py, and clicks.py's HTML) shows that **nine server routes are never called by the bundled browser code**: /api/upload, /api/calibration-log, /api/nomocap-format-zip, /api/nomocap-format, /api/tracking-run/reference-log, /api/tracking-run/reference, /api/calibration/camera, /api/upload/remove, and /api/upload-frame (routes at webui.py:942–1101). They may serve scripts or older clients, but they are untested, and remove_upload's metadata loss above is reachable only through one of them. Also, dashboard.renderHelp writes into a #getting-started element that does not exist, so its HELP text is never shown (dashboard.py:421–457).

### 5.2 Numerical and evaluation limitations

- **Mocap-conditioned evaluation:** target-run mocap is used to optimize time alignment before reporting errors against that same mocap (triangulate_527.py:475–511,595–646,704–707). This is useful diagnostic evaluation but should be identified as assisted, rather than independent performance.
- **Calibration acceptance does not bound 3D error:** joint validation reports track RMSE but accepts/rejects primarily through ray reprojection, depth, boundaries, and rank (joint_calibration_solver.py:461–472,552–574). Low reprojection alone does not guarantee accurate world coordinates or accurate onboard POS reference.
- **Robust objective versus diagnostics:** joint fitting applies componentwise Huber to all residuals, but seed ranking and local covariance use unrobustified squared residuals/Jacobian (411,436–448,537,575–577). The numerical effect needs tests; uncertainty is not a calibrated absolute error guarantee.
- **Potential final-rejection-pass inconsistency:** _fit_seed can change visibility on the fourth and final pass, then evaluate that new model without fitting it again (387–448). This is an inferred edge case to reproduce with adversarial observations.
- **Cache and camera assumptions:** dataset cache reuse does not fingerprint inputs/configuration; target camera matrices can come from cached calibration rather than target-session intrinsics (triangulate_527.py:595–619,875–884). Camera movement, lens changes, missing configurations, or resolution differences can invalidate reuse.
- **Weak synchronization/overlap failure behavior:** when dataset synchronization finds no valid candidate it returns nominal offset/infinite error/zero samples and continues (475–511). Generic upload reconstruction requires two loaded cameras but can still write all-missing raw output if their valid timelines never overlap (triangulate_uploads.py:280–329).
- **Coverage and uncertainty limitations:** choosing the fewest-frame reference does not maximize usable overlap; DLT lacks explicit degeneracy guards; subset enumeration grows exponentially; smoothing extrapolates short gaps and uses heuristic noise. Interpret errors alongside raw detection/triangulation coverage, not smoothed count alone.
- **Coordinate conventions remain a data-dependent question:** dataset formatting rotates video but copies calibration unchanged, followed by dimension-based scaling (format_527.py:100–114; triangulate_527.py:126–136). Whether supplied calibration already matches upright images needs actual dataset evidence. Legacy Android crop/rotation and distortion-axis conventions also warrant fixture-based checks (calibrate_logs.py:371–473).
- **3D view orientation:** the results canvas treats world **y** as screen-vertical (results_page.py:264–271). Flight-log calibrations use east/north/up, so altitude initially appears as depth until the user orbits the view. This is a display convention, not a data error; it is inferred from the projection code and not observed in a browser.
- **Input validation and execution assumptions:** some geometry/filters check only the first coordinate for missingness; schema validation is not full physical validation; constant FPS and fixed clock offsets do not account for dropped/variable-rate frames or drift. No supplied recordings establish whether these assumptions hold for a new capture system.

### 5.3 Documentation versus implementation

| Written expectation | What this checkout implements |
|---|---|
| README.md:39–47 presents dataset/manual-upload workflows and says detection needs click annotations. | There is also log-calibrated No-Mocap tracking and optional classification; text-only detection is supported (webui.py:640–694; detect.py:343–400). |
| README.md:57–66 lists six “main commands.” | cli.py defines nine; log-summary, calibrate-from-log, and calibrate-from-mocap are absent from the README (cli.py:225–260). The README's triangulate stage is also mocap-only; no CLI command performs No-Mocap triangulation (cli.py:214–223). |
| docs/flight-log-calibration.md:110 advertises zero search radius to fix lag. | Numerical freezing exists, but strict exported JSON fails on NaN lag uncertainty. |
| docs/flight-log-calibration.md:66–67 promises held-out all/inlier errors and coverage. | Current joint _metrics returns aggregate errors/counts/depth fraction; per-camera acceptance counts are not exported. The older solver has more explicit inlier metrics. |
| cli.py:242 calls --huber-px a 2D detection-residual Huber transition. | Current wrapper passes it as pixel_sigma_px into a global componentwise Huber objective (calibrate_logs.py:297–299; joint_calibration_solver.py:411). |
| ekf.py introductory wording describes an Extended Kalman Filter and filling gaps. | The model is linear and includes up to 30 frames of prediction/extrapolation, including trailing gaps. |
| results.py:1 describes read-only output access. | apply_centroid_edits mutates stored detections and summaries. |
| classify.py:52–54 suggests starting the app online can fetch a missing model. | Service preload requests local files only; the preparation script handles fetching. |
| Dockerfile:37 says the SAM3 vocabulary is packaged. | The required gzip is absent in this snapshot. |
| Dashboard page description suggests a self-contained document. | HTML/JS/CSS are embedded, but Google Fonts CSS/resources are requested externally (dashboard.py:13–15; results_page.py:10–11). |

### 5.4 Questions for the project owner

1. Where are the authoritative sample datasets, camera/metadata schema examples, and missing SAM3 vocabulary asset obtained?
2. Which Python/CUDA/SAM3 combinations and container image revision have actually passed the intended production or research workflow?
3. Are all No-Mocap cameras synchronized to a common clock before capture, and what drift/dropped-frame tolerance is acceptable?
4. Which calibration reuse conditions are guaranteed: fixed camera mounts, fixed zoom/focus, same recording resolution/crop, and stable labels?
5. What world-coordinate accuracy and coverage constitute success, and how biased is the onboard POS reference expected to be?
6. Should joint priors remain quadratic, should log-distance acceptance be explicit, and how should uncertainty/held-out diagnostics be reported?
7. What external classifier version/model metrics justify interpreting the ArduPilot/PX4 output for these recordings?
8. Is the broad dataset-clear scope intentional? The UI wording and newer scoped clear APIs suggest it should be narrower.

These questions are not prerequisites to understanding the source, but they affect trustworthy deployment, evaluation, and extension.

## 6. Recommended Reading Order

1. **README.md, pyproject.toml, Dockerfile.** Learn the intended user workflow, entry point, declared dependencies, and packaged runtime; keep the missing-asset and stale-documentation findings in mind.
2. **cli.py and WebConfig/_stage_fn in webui.py.** See which operations actually exist and how input/output roots and stages connect before reading HTTP details.
3. **sources.py, dataset_527.py, and the data-contract table in this report.** Learn labels, namespaces, raw/formatted/generic layouts, and legacy naming.
4. **clicks.py and detect.py.** Follow one user click through prompt selection, segmentation, centroid extraction, and saved arrays.
5. **triangulate_uploads.py, then shared geometry at triangulate_527.py:514–582.** Understand the simpler calibrated-input reconstruction before tackling dataset-specific calibration.
6. **ekf.py and output serialization.** Understand what smoothed points mean, where gaps are filled, and why raw coverage must be retained.
7. **docs/flight-log-calibration.md and calibrate_logs.py:246–368.** Learn the actual automatic-calibration file boundary and clock/world-frame conventions.
8. **flight_reference.py:51–118, calibration_solver.SplinePath, then joint_calibration_solver.py.** Establish trustworthy units and time first, then study synchronized observations, ray intersection, objective derivatives, optimization, and validation alongside synthetic tests.
9. **mocap.py, format_527.py, dataset calibration/orchestration, calibrate_mocap.py.** Understand the older dataset's assumptions and how mocap-derived calibration is reused/exported.
10. **jobs.py, the remaining webui.py handlers, dashboard.py, results.py/results_page.py.** Trace one button request through asynchronous work, disk state, and browser presentation; inspect clear/edit behavior before relying on it.
11. **classify.py, classification.py, scripts/prepare_classifier.py, and all tests.** Learn optional-model boundaries, freshness rules, existing regression expectations, and where new tests belong. Read the older fit_pose_time solver last unless maintaining that API.

## 7. Inspection Coverage

### 7.1 Files read completely

Every file below was read in full during this analysis, including the embedded HTML/CSS/JavaScript in dashboard.py, results_page.py, and clicks.py. The annotated tree in section 1 is not merely a directory listing: every non-Git file in it was inspected. Explicit coverage is:

| Group | Files read in full |
|---|---|
| Root configuration/documentation (6) | .dockerignore; .gitignore; Dockerfile; pyproject.toml; README.md; requirements.txt |
| Documentation (1) | docs/flight-log-calibration.md |
| Package entry points and shell (9) | firetrack/__init__.py; __main__.py; cli.py; webui.py; dashboard.py; clicks.py; jobs.py; sources.py; detect.py |
| Calibration and reference (6) | firetrack/calibrate_logs.py; calibration_priors.py; calibration_solver.py; joint_calibration_solver.py; flight_reference.py; calibrate_mocap.py |
| Dataset, geometry, and smoothing (6) | firetrack/dataset_527.py; format_527.py; mocap.py; triangulate_527.py; triangulate_uploads.py; ekf.py |
| Results and classification (4) | firetrack/results.py; results_page.py; classify.py; classification.py |
| Preparation script (1) | scripts/prepare_classifier.py |
| Tests (10) | tests/test_automatic_flight_reference.py; test_calibration_clear.py; test_calibration_integration.py; test_calibration_solver.py; test_camera_repair.py; test_flight_reference.py; test_joint_calibration_solver.py; test_tracking_classification.py; test_tracking_names.py; test_tracking_output_clear.py |
| Asset packaging/metadata (4) | vendor/__init__.py; vendor/sam3_assets/__init__.py; vendor/moonshot_classifier_assets/__init__.py; vendor/moonshot_classifier_assets/fire-moonshot-classifier-model.json |

Total: **47 files**, including **25 firetrack modules**, **10 test modules**, and **39 Python files overall**. The root Python file count includes the script and three vendor package markers. In table cells, bare filenames after a directory-qualified first item share that directory.

### 7.2 Exclusions and remaining gaps

- **Git internals:** excluded object database, hooks, logs, index, and other version-control internals because they are not application implementation. Only .git/HEAD, .git/refs/heads/main, .git/packed-refs, .git/logs/HEAD, and .git/config (for the remote name) were read, for snapshot identification. Commit history/diffs were not analyzed because Git is not installed and the history is stored in a binary pack file.
- **Generated files, dependencies, binary assets, and build artifacts:** none were present outside .git in the enumerated 47-file checkout. Ignore rules name possible caches, environments, videos, checkpoints, archives, build/dist directories, and numerical outputs; those rules do not prove such files exist. There were no such source-adjacent artifacts to read or skip.
- **External components:** installed-library implementation, SAM3 internals/vocabulary/weights, external classifier wheel/model, pretrained-model training data, FFmpeg implementation, and published Docker layers were unavailable and not inspected. The reports analyze this repository's use of their interfaces.
- **Real input data and credentials:** no videos, BIN logs, mocap TSVs, camera sensor CSVs, or secrets were supplied in the repository. Example numbers and layouts in section 3 are illustrative. No credentials or private user environment values are reproduced.
- **Earlier versions of these reports:** this document revises the version already present in the parent directory. Every implementation claim kept from it was re-checked against the source files listed above. Its execution checks were rerun, one overstated threshold was corrected, and further findings were added. The parent directory also holds three working-notes files (calibration_notes.tmp.md, tracking_notes.tmp.md, web_notes.tmp.md). They are not part of the repository and were not read or used as evidence.
- **Execution gaps:** tests and numerical/browser/GPU workflows remain unrun for the reasons in section 4.6. Successful syntax/manifest checks do not establish dependency compatibility, solver convergence, model accuracy, or real-data behavior.

Both reports are outside fireRepo/. SHA-256 hashes of all 47 repository files, recorded before and after this work, are identical. The verification script ran from a separate scratch directory with bytecode writing disabled, so it left no files in the repository.

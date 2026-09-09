"""Web dashboard that drives the whole FireTrack pipeline.

Two input modes:

- ``dataset``: uploaded 5-27 recordings + mocap -> full pipeline
  (format -> clicks -> detect -> triangulate) with per-video selection.
- ``upload``: no-mocap camera-clip zips -> calibration-flight detection ->
  camera calibration -> tracking-flight detection and triangulation.

Built on the stdlib http.server, matching the existing ``clicks serve`` pattern
(no extra dependencies).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .clicks import HTML_PAGE as CLICKS_HTML, ClicksService, FrameCache, click_status
from .calibrate_logs import calibrate_from_log, summarize_log_and_run
from .classification import ClassificationService
from .dashboard import DASHBOARD_HTML
from .detect import VideoSpec, load_clicks, output_paths, run_detection_on_specs
from . import results as results_api
from .results_page import RESULTS_HTML
from .format_527 import normalize_dataset, probe
from .jobs import JobRunner
from .flight_reference import attach_reference
from .sources import (
    TRACKING_UPLOAD_SUBDIR,
    LEGACY_TRACKING_UPLOAD_SUBDIR,
    MOCAP_RAW_SUBDIR,
    UPLOAD_SUBDIR,
    dataset_click_videos,
    dataset_specs,
    mocap_raw_click_videos,
    mocap_raw_specs,
    upload_click_videos,
    upload_specs,
    tracking_source,
    tracking_result_path,
)
from .triangulate_527 import run_raw_triangulation, run_triangulation
from .triangulate_uploads import (
    rotation_issue,
    rotation_matrix,
    triangulate_uploads,
    validate_calibration,
)

UPLOAD_CHUNK = 1 << 20
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
DATASET_STAGES = {"format", "detect", "triangulate", "run-all"}
MOCAP_RAW_STAGES = {"detect", "triangulate"}
UPLOAD_STAGES = {"detect", "calibrate", "triangulate"}
TRACKING_STAGES = {"detect", "triangulate", "classify"}
DATASET_FILE_NAMES = {
    "video.mp4",
    "metadata.json",
    "camera.json",
    "calibration.json",
    "gps.csv",
    "imu.csv",
    "rf_data.jsonl",
}
NOMOCAP_FILE_NAMES = {"video.mp4", "metadata.json", "camera.json", "gps.csv", "imu.csv", "rf_data.jsonl"}


@dataclass(frozen=True)
class WebConfig:
    work_root: Path

    @property
    def formatted_root(self) -> Path:
        return self.work_root / "formatted"

    @property
    def detections_root(self) -> Path:
        return self.work_root / "detections"

    @property
    def triangulation_root(self) -> Path:
        return self.work_root / "triangulation"

    @property
    def clicks_json(self) -> Path:
        return self.work_root / "clicks.json"

    @property
    def mocap_raw_clicks_json(self) -> Path:
        return self.work_root / "mocap_raw_clicks.json"

    @property
    def dataset_uploads_root(self) -> Path:
        return self.work_root / "dataset_uploads"

    @property
    def uploads_root(self) -> Path:
        return self.work_root / "uploads"

    @property
    def tracking_uploads_root(self) -> Path:
        return self.work_root / TRACKING_UPLOAD_SUBDIR

    @property
    def state_json(self) -> Path:
        return self.work_root / "webui_state.json"

    @property
    def uploads_clicks_json(self) -> Path:
        return self.work_root / "uploads_clicks.json"

    @property
    def tracking_uploads_clicks_json(self) -> Path:
        return self.work_root / "tracking_uploads_clicks.json"

    def tracking_run_root(self, run: str) -> Path:
        return self.tracking_uploads_root / safe_stem(run)

    def tracking_run_clicks_json(self, run: str) -> Path:
        path = self.work_root / f"tracking_uploads_{safe_stem(run)}_clicks.json"
        legacy = self.work_root / f"{LEGACY_TRACKING_UPLOAD_SUBDIR}_{safe_stem(run)}_clicks.json"
        return legacy if not path.exists() and legacy.exists() else path

    @property
    def uploads_calibration_json(self) -> Path:
        return self.work_root / "uploads_calibration.json"

    @property
    def calibration_logs_root(self) -> Path:
        return self.work_root / "calibration_logs"

    @property
    def classification_root(self) -> Path:
        return self.work_root / "classification"


def safe_stem(filename: str) -> str:
    stem = Path(filename).name.rsplit(".", 1)[0]
    cleaned = SAFE_NAME.sub("_", stem).strip("._-")
    return cleaned or "upload"


def safe_relpath(path: str) -> Path:
    parts = [SAFE_NAME.sub("_", p).strip("._-") for p in Path(path).parts]
    clean = [p for p in parts if p and p not in (".", "..")]
    if not clean:
        raise ValueError("empty path")
    return Path(*clean)


def dataset_store_rel(upload_path: str, run_hint: str | None = None) -> Path:
    """Map uploaded 5-27 files into the raw layout expected by format/triangulate.

    The existing 5-27 code expects ``<root>/<run>/<run>/<camera>/video.mp4`` and
    ``<root>/<run>/<run>/<run>_data_6D.tsv``. Browser directory uploads often
    provide ``<run>/<camera>/video.mp4``; insert the duplicate run segment when
    needed while preserving already-compatible uploads.
    """
    rel = safe_relpath(upload_path)
    parts = rel.parts
    run = run_hint or parts[0]
    if len(parts) >= 2 and parts[1] == run:
        return rel
    if run_hint is not None:
        tail = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
        return Path(run) / run / tail
    return Path(run) / run / Path(*parts[1:])


def is_dataset_file(path: str) -> bool:
    name = Path(path).name
    return (
        name in DATASET_FILE_NAMES
        or name.endswith("_data_6D.tsv")
        or name.endswith("_data.tsv")
    )


def infer_dataset_run(paths: list[str]) -> str | None:
    """Infer a 5-27 run name from mocap TSV names inside an upload."""
    for path in paths:
        name = Path(path).name
        suffix = "_data_6D.tsv"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    for path in paths:
        name = Path(path).name
        suffix = "_data.tsv"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def nomocap_store_rel(upload_path: str) -> Path | None:
    """Map a no-mocap archive member to ``<camera>/<file>``.

    The expected archive shape is flexible: ``cam1/video.mp4``,
    ``run/cam1/video.mp4``, and deeper variants all map to ``cam1/video.mp4``.
    """
    rel = safe_relpath(upload_path)
    if rel.name not in NOMOCAP_FILE_NAMES:
        return None
    if len(rel.parts) < 2:
        return None
    return Path(safe_stem(rel.parts[-2])) / rel.name


def normalize_nomocap_videos(root: Path) -> list[dict]:
    """Index no-mocap videos without rewriting pixels.

    The no-mocap geometry path uses camera intrinsics/extrinsics in the native
    video frame. Re-encoding or rotating here makes that bookkeeping fragile, so
    this function only records what was uploaded.
    """
    manifest_path = root / "manifest.json"
    manifest = []
    for video in sorted(root.glob("*/video.mp4")):
        label = video.parent.name
        n, w, h = probe(video)
        manifest.append({
            "label": label,
            "video_path": str(video),
            "rotation_steps_ccw": 0,
            "source_width": w,
            "source_height": h,
            "width": w,
            "height": h,
            "n_frames": n,
        })
    if manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _has_detection(cfg: WebConfig, spec: VideoSpec) -> bool:
    return output_paths(spec, cfg.detections_root).summary_path.exists()


def _filter(specs: list[VideoSpec], only: list[str] | None) -> list[VideoSpec]:
    if not only:
        return specs
    wanted = set(only)
    return [spec for spec in specs if spec.label in wanted]


def _tracking_runs(cfg: WebConfig) -> list[str]:
    runs = set()
    for root in (cfg.tracking_uploads_root, cfg.work_root / LEGACY_TRACKING_UPLOAD_SUBDIR):
        if not root.exists():
            continue
        for child in (p for p in root.iterdir() if p.is_dir()):
            if not (child / "video.mp4").exists() and any(child.glob("*/video.mp4")):
                runs.add(child.name)
        if any(root.glob("*/video.mp4")):
            runs.add("default")
    return sorted(runs)


def _read_state(cfg: WebConfig) -> dict:
    if not cfg.state_json.exists():
        return {}
    try:
        data = json.loads(cfg.state_json.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(cfg: WebConfig, state: dict) -> None:
    cfg.state_json.write_text(json.dumps(state, indent=2, sort_keys=True))


def _selected_tracking_run(cfg: WebConfig) -> str | None:
    runs = _tracking_runs(cfg)
    state = _read_state(cfg)
    selected = state.get("tracking_run")
    if isinstance(selected, str) and selected in runs:
        return selected
    return runs[0] if runs else None


def _set_selected_tracking_run(cfg: WebConfig, run: str) -> str:
    safe = safe_stem(run)
    state = _read_state(cfg)
    state["tracking_run"] = safe
    _write_state(cfg, state)
    _tracking_root(cfg, safe).mkdir(parents=True, exist_ok=True)
    return safe


def _tracking_root(cfg: WebConfig, run: str | None) -> Path:
    for root in (cfg.tracking_uploads_root, cfg.work_root / LEGACY_TRACKING_UPLOAD_SUBDIR):
        candidate = root if run == "default" else root / safe_stem(run or "tracking_run")
        if any(candidate.glob("*/video.mp4")):
            return candidate
    if run == "default":
        return cfg.tracking_uploads_root
    return cfg.tracking_run_root(run or "tracking_run")


def _tracking_prefix(cfg: WebConfig, run: str | None) -> Path:
    return _tracking_root(cfg, run).relative_to(cfg.work_root)


def _clicks_json_for(cfg: WebConfig, source: str, tracking_run: str | None = None) -> Path:
    if source == "tracking":
        return cfg.tracking_run_clicks_json(tracking_run or _selected_tracking_run(cfg) or "tracking_run")
    if source == "mocap_raw":
        return cfg.mocap_raw_clicks_json
    return cfg.uploads_clicks_json if source == "upload" else cfg.clicks_json


def _effective_clicks(path: Path) -> Path | None:
    """Return the clicks file only if it holds >=1 real approved click.

    Merely opening the annotator creates an empty manifest, so gating on file
    existence would make detect skip every un-annotated video. Gate on content.
    """
    if not path.exists():
        return None
    try:
        return path if load_clicks(path) else None
    except (ValueError, OSError):
        return None


def build_status(
    cfg: WebConfig,
    runner: JobRunner,
    classifier: ClassificationService | None = None,
) -> dict:
    tracking_runs = _tracking_runs(cfg)
    selected_tracking_run = _selected_tracking_run(cfg)
    tracking_root = _tracking_root(cfg, selected_tracking_run)
    tracking_prefix = _tracking_prefix(cfg, selected_tracking_run)
    dataset = [
        {"label": s.label, "has_detection": _has_detection(cfg, s)}
        for s in dataset_specs(cfg.formatted_root)
    ]
    mocap_raw = [
        {"label": s.label, "has_detection": _has_detection(cfg, s)}
        for s in mocap_raw_specs(cfg.dataset_uploads_root)
    ]
    uploads = [
        {"label": s.label, "has_detection": _has_detection(cfg, s)}
        for s in upload_specs(cfg.uploads_root, relative_prefix=UPLOAD_SUBDIR)
    ]
    tracking_uploads = [
        {"label": s.label, "has_detection": _has_detection(cfg, s)}
        for s in upload_specs(tracking_root, relative_prefix=tracking_prefix)
    ]
    return {
        "dataset": dataset,
        "mocap_raw": mocap_raw,
        "uploads": uploads,
        "tracking_uploads": tracking_uploads,
        "tracking_runs": tracking_runs,
        "tracking_run_summaries": [_tracking_run_summary(cfg, run) for run in tracking_runs],
        "calibration_summary": _calibration_summary(cfg, uploads),
        "selected_tracking_run": selected_tracking_run,
        "outputs": {
            "dataset_uploads": cfg.dataset_uploads_root.exists(),
            "formatted": cfg.formatted_root.exists(),
            "detections": cfg.detections_root.exists(),
            "triangulation": (cfg.triangulation_root / "summary.json").exists(),
            "mocap_raw_triangulation": (cfg.triangulation_root / MOCAP_RAW_SUBDIR / "summary.json").exists(),
            "uploads_triangulation": (cfg.triangulation_root / "uploads" / "summary.json").exists(),
            "tracking_uploads_triangulation": (
                selected_tracking_run is not None
                and (cfg.triangulation_root / tracking_prefix / "summary.json").exists()
            ),
        },
        "calibration_log": _calibration_log_status(cfg),
        "clicks": _safe_click_status(cfg.clicks_json),
        "mocap_raw_clicks": _safe_click_status(cfg.mocap_raw_clicks_json),
        "uploads_clicks": _safe_click_status(cfg.uploads_clicks_json),
        "tracking_uploads_clicks": _safe_click_status(_clicks_json_for(cfg, "tracking", selected_tracking_run)),
        "calibration": _calibration_status(cfg.uploads_calibration_json),
        "classification": (
            classifier.status(str(tracking_prefix))
            if classifier is not None and selected_tracking_run is not None
            else {
                "ready": False,
                "trajectory_ready": False,
                "reason": "Select a tracking run first.",
                "result": None,
            }
        ),
        "env": _env_info(),
        "job": runner.status(),
    }


def _count_done(items: list[dict]) -> int:
    return sum(1 for item in items if item.get("has_detection"))


def _calibration_summary(cfg: WebConfig, uploads: list[dict]) -> dict:
    clicks = _safe_click_status(cfg.uploads_clicks_json)
    calibration = _calibration_status(cfg.uploads_calibration_json)
    logs = []
    if cfg.calibration_logs_root.exists():
        logs = sorted(cfg.calibration_logs_root.glob("*.BIN")) + sorted(cfg.calibration_logs_root.glob("*.bin"))
    return {
        "clips": len(uploads),
        "indexed": (cfg.uploads_root / "manifest.json").exists(),
        "detections": _count_done(uploads),
        "annotations": clicks,
        "logs": len(logs),
        "calibrated": calibration.get("present", False),
        "cameras": calibration.get("n_cameras", 0),
    }


def _tracking_run_summary(cfg: WebConfig, run: str) -> dict:
    root = _tracking_root(cfg, run)
    prefix = _tracking_prefix(cfg, run)
    uploads = [
        {"label": s.label, "has_detection": _has_detection(cfg, s)}
        for s in upload_specs(root, relative_prefix=prefix)
    ]
    return {
        "run": run,
        "clips": len(uploads),
        "indexed": (root / "manifest.json").exists(),
        "detections": _count_done(uploads),
        "annotations": _safe_click_status(_clicks_json_for(cfg, "tracking", run)),
        "triangulated": (cfg.triangulation_root / prefix / "summary.json").exists(),
    }


def _calibration_status(path: Path) -> dict:
    """Report whether a valid upload calibration is present, for the UI."""
    if not path.exists():
        return {"present": False, "n_cameras": 0, "cameras": []}
    try:
        cams = validate_calibration(json.loads(path.read_text()))
        return {"present": True, "n_cameras": len(cams), "cameras": [c["video"] for c in cams]}
    except (ValueError, OSError, json.JSONDecodeError):
        return {"present": False, "n_cameras": 0, "cameras": [], "error": "invalid"}


def _calibration_log_status(cfg: WebConfig) -> dict:
    log = _calibration_log_path(cfg)
    return {"present": log is not None, "path": str(log) if log is not None else None}


def _calibration_log_path(cfg: WebConfig) -> Path | None:
    if not cfg.calibration_logs_root.exists():
        return None
    logs = sorted(cfg.calibration_logs_root.glob("*.BIN")) + sorted(cfg.calibration_logs_root.glob("*.bin"))
    return logs[0] if logs else None


def _best_calibration_log_path(cfg: WebConfig) -> Path | None:
    """Select a unique overlapping log; never guess between tracked drones."""
    if not cfg.calibration_logs_root.exists():
        return None
    logs = sorted(cfg.calibration_logs_root.glob("*.BIN")) + sorted(cfg.calibration_logs_root.glob("*.bin"))
    if not logs:
        return None
    candidates = []
    for log in logs:
        try:
            summary = summarize_log_and_run(log, cfg.uploads_root)
        except Exception as exc:
            print(f"Skipping calibration log {log.name}: {exc}")
            continue
        overlap = sum(float(cam.get("overlap_s", 0.0)) for cam in summary.get("cameras", []))
        if overlap > 0:
            candidates.append(log)
    if len(candidates) > 1:
        raise ValueError("Multiple uploaded flight logs overlap calibration. Use calibrate-from-log --log to explicitly select the tracked drone's log.")
    return candidates[0] if candidates else None


def load_calibration_store(path: Path) -> dict:
    """Read the calibration document, or an empty one."""
    if not path.exists():
        return {"cameras": []}
    try:
        doc = json.loads(path.read_text())
        return doc if isinstance(doc.get("cameras"), list) else {"cameras": []}
    except (json.JSONDecodeError, OSError):
        return {"cameras": []}


def remove_upload(cfg: WebConfig, label: str) -> None:
    """Delete an uploaded clip, its detection outputs, and its calibration entry."""
    safe = safe_stem(label)
    for root in (cfg.uploads_root, cfg.detections_root / "uploads"):
        target = (root / safe).resolve()
        if root.resolve() in target.parents and target.exists():
            shutil.rmtree(target, ignore_errors=True)
    store = load_calibration_store(cfg.uploads_calibration_json)
    kept = [c for c in store["cameras"] if c.get("video") != safe]
    if len(kept) != len(store["cameras"]):
            cfg.uploads_calibration_json.write_text(json.dumps({"cameras": kept}, indent=2))


def remove_tracking_upload(cfg: WebConfig, label: str, tracking_run: str | None = None) -> None:
    safe = safe_stem(label)
    run = tracking_run or _selected_tracking_run(cfg) or "tracking_run"
    upload_root = _tracking_root(cfg, run)
    detections_root = cfg.detections_root / _tracking_prefix(cfg, run)
    for root in (upload_root, detections_root):
        target = (root / safe).resolve()
        if root.resolve() in target.parents and target.exists():
            shutil.rmtree(target, ignore_errors=True)


def tracking_output_targets(cfg: WebConfig, run: str, camera: str | None = None) -> list[Path]:
    """Resolve only generated outputs belonging to an existing tracking run."""
    if not isinstance(run, str) or not run or safe_stem(run) != run:
        raise ValueError("invalid tracking run")
    if run not in _tracking_runs(cfg):
        raise ValueError("tracking run not found")
    labels = [p.parent.name for p in _tracking_root(cfg, run).glob("*/video.mp4")]
    if camera is not None and (not isinstance(camera, str) or camera not in labels):
        raise ValueError("camera not found in the selected tracking run")
    work = cfg.work_root.resolve()
    detections = work / "detections" / _tracking_prefix(cfg, run)
    triangulation = work / "triangulation" / _tracking_prefix(cfg, run)
    classification = work / "classification" / _tracking_prefix(cfg, run)
    if run == "default":
        # The old flat layout can share its parent with named runs.
        targets = [detections / label for label in ([camera] if camera is not None else labels)]
        targets += [triangulation / name for name in ("trajectory.npz", "trajectory.csv", "summary.json", "flight_reference.npz", "flight_reference.json", "flight_comparison.csv")]
        targets += [classification / "prediction.json"]
    else:
        targets = [detections / camera if camera is not None else detections, triangulation, classification]
    if any(target.resolve() != target for target in targets):
        raise ValueError("refusing to clear outputs through a symbolic link")
    return targets


def clear_tracking_outputs(cfg: WebConfig, run: str, camera: str | None = None) -> None:
    targets = tracking_output_targets(cfg, run, camera)
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    results_api.video_path_for.cache_clear()
    print(f"Cleared {'detections for ' + camera if camera is not None else 'all detections'}, reconstruction, and classification for tracking run: {run}")
    print("Videos, annotations, and camera calibration were kept.")


def calibration_clear_targets(cfg: WebConfig) -> list[Path]:
    work = cfg.work_root.resolve()
    targets = [work / name for name in (
        "uploads", "calibration_logs", "detections/uploads", "triangulation/uploads",
        "uploads_clicks.json", "uploads_calibration.json",
    )]
    if any(target.resolve() != target for target in targets):
        raise ValueError("refusing to clear calibration through a symbolic link")
    return targets


def calibration_camera_output_targets(cfg: WebConfig, camera: str) -> list[Path]:
    if not isinstance(camera, str) or camera not in [s.label for s in upload_specs(cfg.uploads_root)]:
        raise ValueError("camera not found in the calibration flight")
    work = cfg.work_root.resolve()
    targets = [work / "detections/uploads" / camera, work / "triangulation/uploads", work / "uploads_calibration.json"]
    if any(target.resolve() != target for target in targets):
        raise ValueError("refusing to clear camera outputs through a symbolic link")
    return targets


def clear_calibration_camera_outputs(cfg: WebConfig, camera: str) -> None:
    for target in calibration_camera_output_targets(cfg, camera):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    results_api.video_path_for.cache_clear()
    print(f"Cleared calibration detections for {camera}, calibration reconstruction, and the active calibration JSON.")
    print("Other cameras' detections, all inputs and annotations, flight logs, and tracking runs were kept. Run Calibrate after repairing detections.")


def clear_calibration(cfg: WebConfig) -> None:
    for target in calibration_clear_targets(cfg):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    results_api.video_path_for.cache_clear()
    print("Cleared calibration clips, metadata, flight logs, annotations, detections, reconstruction, and camera calibration JSON.")
    print("Tracking runs and saved results, mocap data, and files outside the work folder were kept.")


def upsert_camera(path: Path, camera: dict) -> int:
    """Validate one camera and merge it into the calibration store by ``video``."""
    validate_calibration({"cameras": [camera]})
    issue = rotation_issue(rotation_matrix(camera))
    if issue:
        raise ValueError(issue)
    store = load_calibration_store(path)
    others = [c for c in store["cameras"] if c.get("video") != camera["video"]]
    store["cameras"] = [*others, camera]
    path.write_text(json.dumps(store, indent=2))
    return len(store["cameras"])


def _env_info() -> dict:
    """Cheap environment readout for the dashboard HUD (no torch import)."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    weights = Path(hf_home) / "hub" / "models--facebook--sam3"
    return {
        "gpu": _gpu_visible(),
        "weights_cached": weights.exists(),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }


def _gpu_visible() -> bool:
    """Detect an attached NVIDIA GPU on native Linux and WSL2 (no torch import)."""
    device_nodes = ("/proc/driver/nvidia/gpus", "/dev/nvidia0", "/dev/dxg")
    if any(Path(p).exists() for p in device_nodes):
        return True
    libdirs = ("/usr/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib")
    return any(any(Path(d).glob("libcuda.so*")) for d in libdirs if Path(d).exists())


def _safe_click_status(path: Path) -> dict | None:
    try:
        return click_status(path)
    except FileNotFoundError:
        return None


def _stage_fn(
    cfg: WebConfig,
    stage: str,
    source: str,
    only: list[str] | None,
    progress=None,
    tracking_run: str | None = None,
    classifier: ClassificationService | None = None,
):
    """Return a zero-arg callable that runs the requested stage."""
    if source == "tracking":
        run = tracking_run or _selected_tracking_run(cfg)
        if run is None:
            raise ValueError("Upload or select a tracking flight first.")
        if stage == "triangulate":
            def triangulate() -> None:
                _tracking_triangulate_fn(cfg, run)()
                if classifier is not None:
                    classifier.clear(str(_tracking_prefix(cfg, run)))

            return triangulate
        if stage == "classify":
            if classifier is None:
                raise ValueError("Trajectory classifier is unavailable.")
            return lambda: classifier.run(str(_tracking_prefix(cfg, run)), progress)
        return _tracking_detect_fn(cfg, run, only, progress)
    if source == "upload":
        if stage == "calibrate":
            return _upload_calibrate_fn(cfg)
        if stage == "triangulate":
            return _upload_triangulate_fn(cfg)
        return _upload_detect_fn(cfg, only, progress)
    if source == "mocap_raw":
        if stage == "detect":
            return _mocap_raw_detect_fn(cfg, only, progress)
        if stage == "triangulate":
            return lambda: run_raw_triangulation(
                raw_root=cfg.dataset_uploads_root,
                detections_root=cfg.detections_root,
                out_root=cfg.triangulation_root,
            )
    if stage == "format":
        return lambda: normalize_dataset(cfg.dataset_uploads_root, cfg.formatted_root, only=only)
    if stage == "detect":
        return _dataset_detect_fn(cfg, only, progress)
    if stage == "triangulate":
        return lambda: run_triangulation(
            raw_root=cfg.dataset_uploads_root,
            formatted_root=cfg.formatted_root,
            detections_root=cfg.detections_root,
            out_root=cfg.triangulation_root,
        )
    if stage == "run-all":
        return _run_all_fn(cfg, only, progress)
    raise ValueError(f"unknown stage: {stage}")


def _dataset_detect_fn(cfg: WebConfig, only: list[str] | None, progress=None):
    def run() -> None:
        run_detection_on_specs(
            _filter(dataset_specs(cfg.formatted_root), only),
            out_root=cfg.detections_root,
            clicks_json=_effective_clicks(cfg.clicks_json),
            on_progress=progress,
        )

    return run


def _mocap_raw_detect_fn(cfg: WebConfig, only: list[str] | None, progress=None):
    def run() -> None:
        run_detection_on_specs(
            _filter(mocap_raw_specs(cfg.dataset_uploads_root), only),
            out_root=cfg.detections_root,
            clicks_json=_effective_clicks(cfg.mocap_raw_clicks_json),
            on_progress=progress,
        )

    return run


def _upload_detect_fn(cfg: WebConfig, only: list[str] | None, progress=None):
    def run() -> None:
        specs = _filter(upload_specs(cfg.uploads_root, relative_prefix=UPLOAD_SUBDIR), only)
        print(f"Calibration detection: {len(specs)} clip(s)")
        run_detection_on_specs(
            specs,
            out_root=cfg.detections_root,
            clicks_json=_effective_clicks(cfg.uploads_clicks_json),
            on_progress=progress,
        )

    return run


def _tracking_detect_fn(cfg: WebConfig, tracking_run: str, only: list[str] | None, progress=None):
    def run() -> None:
        specs = _filter(upload_specs(_tracking_root(cfg, tracking_run), relative_prefix=_tracking_prefix(cfg, tracking_run)), only)
        print(f"Tracking detection ({tracking_run}): {len(specs)} clip(s)")
        run_detection_on_specs(
            specs,
            out_root=cfg.detections_root,
            clicks_json=_effective_clicks(_clicks_json_for(cfg, "tracking", tracking_run)),
            on_progress=progress,
        )

    return run


def _upload_triangulate_fn(cfg: WebConfig):
    def run() -> None:
        triangulate_uploads(
            detections_root=cfg.detections_root,
            calibration_json=cfg.uploads_calibration_json,
            out_root=cfg.triangulation_root,
            uploads_root=cfg.uploads_root,
        )

    return run


def _upload_calibrate_fn(cfg: WebConfig):
    def run() -> None:
        log_path = _best_calibration_log_path(cfg)
        if log_path is None:
            raise RuntimeError("No usable calibration .BIN log overlaps the camera timestamps. Check calibration upload and UTC metadata.")
        print(f"Using calibration flight log: {log_path}")
        calibrate_from_log(
            log_path=log_path,
            run_root=cfg.uploads_root,
            detections_root=cfg.detections_root,
            out_json=cfg.uploads_calibration_json,
        )

    return run


def _tracking_reference(cfg: WebConfig, tracking_run: str) -> None:
    target = cfg.triangulation_root / _tracking_prefix(cfg, tracking_run)
    sidecars = [target / name for name in (
        "flight_reference.json", "flight_reference.npz", "flight_comparison.csv",
    )]
    try:
        for path in sidecars:
            path.unlink(missing_ok=True)
        logs = sorted(p for p in cfg.calibration_logs_root.glob("*")
                      if p.is_file() and p.suffix.lower() == ".bin")
        # Keep older runs usable when their logs were uploaded separately.
        if not logs:
            root = cfg.work_root / "tracking_logs" / tracking_run
            logs = sorted(p for p in root.glob("*")
                          if p.is_file() and p.suffix.lower() == ".bin")
        if not logs:
            raise ValueError("No stored flight logs. Include .BIN logs in the calibration ZIP for reference comparison.")
        attach_reference(target, logs, cfg.uploads_calibration_json)
    except Exception as exc:
        # Comparison is optional: its failure must not invalidate triangulation.
        message = f"Flight-log reference unavailable: {exc}"
        print(message)
        try:
            for path in sidecars:
                path.unlink(missing_ok=True)
            sidecars[0].write_text(json.dumps({"error": message}))
        except OSError as cleanup_error:
            print(f"Could not update reference status: {cleanup_error}")


def _tracking_triangulate_fn(cfg: WebConfig, tracking_run: str):
    def run() -> None:
        triangulate_uploads(
            detections_root=cfg.detections_root,
            calibration_json=cfg.uploads_calibration_json,
            out_root=cfg.triangulation_root,
            uploads_root=_tracking_root(cfg, tracking_run),
            detections_subdir=str(_tracking_prefix(cfg, tracking_run)),
            output_subdir=str(_tracking_prefix(cfg, tracking_run)),
        )
        _tracking_reference(cfg, tracking_run)

    return run


def _run_all_fn(cfg: WebConfig, only: list[str] | None, progress=None):
    def run() -> None:
        normalize_dataset(cfg.dataset_uploads_root, cfg.formatted_root, only=only)
        _dataset_detect_fn(cfg, only, progress)()
        run_triangulation(
            raw_root=cfg.dataset_uploads_root,
            formatted_root=cfg.formatted_root,
            detections_root=cfg.detections_root,
            out_root=cfg.triangulation_root,
        )

    return run


class _ClicksRegistry:
    """Lazily builds and caches a ClicksService per source for the annotator."""

    def __init__(self, cfg: WebConfig) -> None:
        self._cfg = cfg
        self._services: dict[str, ClicksService] = {}
        self.active: ClicksService | None = None

    def discard_calibration(self) -> None:
        service = self._services.pop("upload", None)
        if service is not None:
            with service._lock:
                service.rows.clear()
                if self.active is service:
                    self.active = None

    def select(self, source: str, tracking_run: str | None = None) -> None:
        source = tracking_source(source)
        if source == "tracking":
            run = tracking_run or _selected_tracking_run(self._cfg)
            if run is None:
                videos = []
            else:
                videos = upload_click_videos(
                    _tracking_root(self._cfg, run),
                    run_name=_tracking_prefix(self._cfg, run),
                )
        elif source == "upload":
            videos = upload_click_videos(self._cfg.uploads_root, run_name=UPLOAD_SUBDIR)
        elif source == "mocap_raw":
            videos = mocap_raw_click_videos(self._cfg.dataset_uploads_root)
        else:
            videos = dataset_click_videos(self._cfg.formatted_root)
        # Rebuild every time so newly uploaded/formatted videos appear.
        service = ClicksService(videos, _clicks_json_for(self._cfg, source, tracking_run))
        self._services[source] = service
        self.active = service


def serve_webui(*, work_root: Path, host: str, port: int) -> None:
    cfg = WebConfig(work_root.resolve())
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    cfg.dataset_uploads_root.mkdir(parents=True, exist_ok=True)
    cfg.uploads_root.mkdir(parents=True, exist_ok=True)
    cfg.tracking_uploads_root.mkdir(parents=True, exist_ok=True)
    cfg.calibration_logs_root.mkdir(parents=True, exist_ok=True)
    runner = JobRunner()
    classifier = ClassificationService(cfg.triangulation_root, cfg.classification_root)
    classifier.preload()
    clicks = _ClicksRegistry(cfg)
    result_frames = FrameCache()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, payload: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _json(self, status: int, obj: object) -> None:
            self._send(status, "application/json", json.dumps(obj).encode())

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode()) if length else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/":
                self._send(200, "text/html; charset=utf-8", DASHBOARD_HTML)
                return
            if route == "/clicks":
                self._send(200, "text/html; charset=utf-8", CLICKS_HTML)
                return
            if route == "/results":
                self._send(200, "text/html; charset=utf-8", RESULTS_HTML)
                return
            if route == "/api/status":
                self._json(200, build_status(cfg, runner, classifier))
                return
            if route == "/api/results":
                self._json(200, {
                    "detections": results_api.list_detections(cfg.detections_root),
                    "trajectories": results_api.list_trajectories(cfg.triangulation_root),
                })
                return
            if route == "/api/centroids":
                self._serve_results(parse_qs(parsed.query), self._load_centroids)
                return
            if route == "/api/trajectory":
                self._serve_results(parse_qs(parsed.query), self._load_trajectory)
                return
            if route == "/api/trajectory/download":
                self._serve_trajectory_download(parse_qs(parsed.query))
                return
            if route == "/api/result-frame":
                self._serve_result_frame(parse_qs(parsed.query))
                return
            if route == "/api/calibration":
                self._json(200, load_calibration_store(cfg.uploads_calibration_json))
                return
            if route == "/api/upload-frame":
                self._serve_upload_frame(parse_qs(parsed.query))
                return
            if route == "/api/log":
                since = int(parse_qs(parsed.query).get("since", ["0"])[0])
                self._json(200, runner.log_since(since))
                return
            if route == "/api/rows":
                self._json(200, [] if clicks.active is None else json.loads(clicks.active.rows_bytes()))
                return
            if route == "/frame":
                self._serve_frame(parse_qs(parsed.query))
                return
            self._send(404, "text/plain", b"not found")

        def _load_centroids(self, rel: str) -> dict:
            return results_api.load_centroids(cfg.detections_root, rel)

        def _load_trajectory(self, rel: str) -> dict:
            data = results_api.load_trajectory(cfg.triangulation_root, rel)
            normalized = tracking_result_path(rel)
            if normalized == TRACKING_UPLOAD_SUBDIR or normalized.startswith(TRACKING_UPLOAD_SUBDIR + "/"):
                data["classification"] = classifier.result(normalized)
            return data

        def _serve_results(self, query: dict, loader) -> None:
            rel = query.get("dir", [""])[0]
            if not rel:
                self._json(400, {"error": "missing ?dir="})
                return
            try:
                self._json(200, loader(rel))
            except ValueError:
                self._send(400, "text/plain", b"bad dir")
            except (FileNotFoundError, OSError, KeyError):
                self._send(404, "text/plain", b"result not found")

        def _serve_upload_frame(self, query: dict) -> None:
            label = safe_stem(query.get("label", [""])[0])
            bucket = tracking_source(query.get("bucket", ["calibration"])[0])
            tracking_run = query.get("run", [""])[0] or _selected_tracking_run(cfg)
            try:
                frame_idx = int(query.get("frame", ["0"])[0])
            except ValueError:
                self._send(400, "text/plain", b"bad frame")
                return
            root = _tracking_root(cfg, tracking_run) if bucket == "tracking" else cfg.uploads_root
            video = root / label / "video.mp4"
            jpg = result_frames.read(str(video), frame_idx) if video.exists() else None
            if jpg is None:
                self._send(404, "text/plain", b"frame not available")
                return
            self._send(200, "image/jpeg", jpg)

        def _serve_trajectory_download(self, query: dict) -> None:
            rel = query.get("dir", [""])[0]
            fmt = query.get("fmt", ["csv"])[0]
            if not rel:
                self._send(400, "text/plain", b"missing dir")
                return
            try:
                path = results_api.trajectory_file(cfg.triangulation_root, rel, fmt)
            except ValueError:
                self._send(400, "text/plain", b"bad fmt")
                return
            except (FileNotFoundError, OSError):
                self._send(404, "text/plain", b"trajectory not found")
                return
            data = path.read_bytes()
            fname = results_api.trajectory_download_name(rel, path.name)
            ctype = "text/csv" if fmt in ("csv", "comparison") else "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_result_frame(self, query: dict) -> None:
            rel = query.get("dir", [""])[0]
            try:
                frame_idx = int(query.get("frame", ["0"])[0])
                video_path = results_api.video_path_for(str(cfg.detections_root), rel)
            except (ValueError, FileNotFoundError, OSError, KeyError):
                self._send(400, "text/plain", b"bad frame request")
                return
            jpg = result_frames.read(video_path, frame_idx)
            if jpg is None:
                self._send(404, "text/plain", b"frame read failed")
                return
            self._send(200, "image/jpeg", jpg)

        def _serve_frame(self, query: dict) -> None:
            if clicks.active is None:
                self._send(404, "text/plain", b"no active source")
                return
            try:
                index = int(query["index"][0])
                frame_idx = int(query["frame"][0])
                jpg = clicks.active.frame_jpeg(index, frame_idx)
            except (KeyError, ValueError, IndexError):
                self._send(400, "text/plain", b"bad frame request")
                return
            if jpg is None:
                self._send(404, "text/plain", b"frame read failed")
                return
            self._send(200, "image/jpeg", jpg)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/api/run":
                self._handle_run(self._read_json())
                return
            if route == "/api/upload":
                self._handle_upload(parse_qs(parsed.query))
                return
            if route == "/api/calibration-log":
                self._handle_calibration_log(parse_qs(parsed.query))
                return
            if route == "/api/dataset-upload-zip":
                self._handle_dataset_upload_zip(parse_qs(parsed.query))
                return
            if route == "/api/nomocap-format-zip":
                self._handle_nomocap_format_zip(parse_qs(parsed.query))
                return
            if route == "/api/nomocap-upload-zip":
                self._handle_nomocap_upload_zip(parse_qs(parsed.query))
                return
            if route == "/api/nomocap-format":
                self._handle_nomocap_format(parse_qs(parsed.query))
                return
            if route == "/api/tracking-run/select":
                self._handle_tracking_run_select(self._read_json())
                return
            if route == "/api/tracking-run/clear-outputs":
                self._handle_tracking_outputs_clear(self._read_json())
                return
            if route in ("/api/tracking-run/camera/clear-outputs", "/api/tracking-run/camera/detect"):
                self._handle_camera_action(self._read_json(), route.rsplit("/", 1)[-1], "tracking")
                return
            if route in ("/api/calibration/camera/clear-outputs", "/api/calibration/camera/detect"):
                self._handle_camera_action(self._read_json(), route.rsplit("/", 1)[-1], "upload")
                return
            if route == "/api/tracking-run/reference-log":
                self._handle_reference_log(parse_qs(parsed.query))
                return
            if route == "/api/tracking-run/reference":
                self._handle_reference(self._read_json())
                return
            if route == "/api/calibration":
                self._handle_calibration()
                return
            if route == "/api/calibration/clear":
                self._handle_calibration_clear()
                return
            if route == "/api/calibration/camera":
                self._handle_calibration_camera(self._read_json())
                return
            if route == "/api/upload/remove":
                self._handle_upload_remove(self._read_json())
                return
            if route == "/api/dataset/clear":
                self._handle_dataset_clear()
                return
            if route == "/api/centroids/edit":
                self._handle_centroid_edit(self._read_json())
                return
            if route == "/api/clicks/select":
                body = self._read_json()
                clicks.select(body.get("source", "dataset"), body.get("tracking_run"))
                self._json(200, {"ok": True})
                return
            if route == "/api/click":
                self._handle_click(self._read_json())
                return
            if route == "/api/skip":
                self._handle_skip(self._read_json())
                return
            self._send(404, "text/plain", b"not found")

        def _handle_run(self, body: dict) -> None:
            stage = body.get("stage", "")
            source = tracking_source(body.get("source", "dataset"))
            only = body.get("only") or None
            tracking_run = safe_stem(body["tracking_run"]) if source == "tracking" and body.get("tracking_run") else _selected_tracking_run(cfg)
            valid = (
                TRACKING_STAGES if source == "tracking"
                else MOCAP_RAW_STAGES if source == "mocap_raw"
                else UPLOAD_STAGES if source == "upload"
                else DATASET_STAGES
            )
            if stage not in valid:
                self._json(400, {"error": f"invalid stage {stage!r} for {source}"})
                return
            if source in ("upload", "tracking") and stage == "triangulate" and not cfg.uploads_calibration_json.exists():
                self._json(400, {"error": "Upload a camera calibration first."})
                return
            if source == "tracking" and stage == "classify":
                if tracking_run is None:
                    self._json(400, {"error": "Select a tracking run first."})
                    return
                classification = classifier.status(str(_tracking_prefix(cfg, tracking_run)))
                if not classification["ready"]:
                    self._json(400, {"error": classification["reason"]})
                    return
            try:
                fn = _stage_fn(
                    cfg, stage, source, only, runner.set_progress, tracking_run, classifier,
                )
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            label = f"{source}:{tracking_run}:{stage}" if source == "tracking" and tracking_run else f"{source}:{stage}"
            if not runner.start(label, fn):
                self._json(409, {"error": "a job is already running"})
                return
            self._json(202, {"started": label})

        def _handle_upload(self, query: dict) -> None:
            name = query.get("name", [""])[0]
            bucket = tracking_source(query.get("bucket", ["calibration"])[0])
            if not name:
                self._json(400, {"error": "missing ?name="})
                return
            stem = safe_stem(name)
            tracking_run = query.get("run", [""])[0] or _selected_tracking_run(cfg) or "tracking_run"
            if bucket == "tracking":
                tracking_run = _set_selected_tracking_run(cfg, tracking_run)
            dest_root = _tracking_root(cfg, tracking_run) if bucket == "tracking" else cfg.uploads_root
            dest_dir = dest_root / stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "video.mp4"
            remaining = int(self.headers.get("Content-Length", "0"))
            if remaining <= 0:
                self._json(400, {"error": "empty upload"})
                return
            with dest.open("wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            self._json(200, {"label": stem, "run": tracking_run if bucket == "tracking" else None})

        def _handle_calibration_log(self, query: dict) -> None:
            name = query.get("name", ["calibration.BIN"])[0]
            if not name.lower().endswith(".bin"):
                self._json(400, {"error": "upload a .BIN file"})
                return
            remaining = int(self.headers.get("Content-Length", "0"))
            if remaining <= 0:
                self._json(400, {"error": "empty upload"})
                return
            cfg.calibration_logs_root.mkdir(parents=True, exist_ok=True)
            for old in list(cfg.calibration_logs_root.glob("*.BIN")) + list(cfg.calibration_logs_root.glob("*.bin")):
                old.unlink(missing_ok=True)
            dest = cfg.calibration_logs_root / (safe_stem(name) + ".BIN")
            with dest.open("wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            self._json(200, {"path": str(dest)})

        def _handle_dataset_upload_zip(self, query: dict) -> None:
            name = query.get("name", ["dataset.zip"])[0]
            if not name.lower().endswith(".zip"):
                self._json(400, {"error": "upload a .zip file"})
                return
            remaining = int(self.headers.get("Content-Length", "0"))
            if remaining <= 0:
                self._json(400, {"error": "empty upload"})
                return
            tmp = cfg.work_root / f".dataset_upload_{safe_stem(name)}.zip"
            with tmp.open("wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            extracted = 0
            try:
                with zipfile.ZipFile(tmp) as zf:
                    names = [info.filename for info in zf.infolist()]
                    run_hint = infer_dataset_run(names)
                    for info in zf.infolist():
                        if info.is_dir() or not is_dataset_file(info.filename):
                            continue
                        rel = dataset_store_rel(info.filename, run_hint=run_hint)
                        dest = (cfg.dataset_uploads_root / rel).resolve()
                        if cfg.dataset_uploads_root.resolve() not in dest.parents:
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, dest.open("wb") as out:
                            shutil.copyfileobj(src, out, length=UPLOAD_CHUNK)
                        extracted += 1
            except zipfile.BadZipFile:
                self._json(400, {"error": "invalid zip file"})
                return
            finally:
                tmp.unlink(missing_ok=True)
            self._json(200, {"extracted": extracted})

        def _handle_nomocap_format_zip(self, query: dict) -> None:
            name = query.get("name", ["run.zip"])[0]
            bucket = tracking_source(query.get("bucket", ["calibration"])[0])
            if not name.lower().endswith(".zip"):
                self._json(400, {"error": "upload a .zip file"})
                return
            remaining = int(self.headers.get("Content-Length", "0"))
            if remaining <= 0:
                self._json(400, {"error": "empty upload"})
                return
            tracking_run = query.get("run", [""])[0] or safe_stem(name)
            if bucket == "tracking":
                tracking_run = _set_selected_tracking_run(cfg, tracking_run)
            dest_root = _tracking_root(cfg, tracking_run) if bucket == "tracking" else cfg.uploads_root
            tmp = cfg.work_root / f".nomocap_{bucket}_{safe_stem(name)}.zip"
            with tmp.open("wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            extracted = 0
            try:
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        rel = nomocap_store_rel(info.filename)
                        if rel is None:
                            if bucket == "calibration" and Path(info.filename).name.lower().endswith(".bin"):
                                cfg.calibration_logs_root.mkdir(parents=True, exist_ok=True)
                                dest = (cfg.calibration_logs_root / (safe_stem(Path(info.filename).name) + ".BIN")).resolve()
                            else:
                                continue
                        else:
                            dest = (dest_root / rel).resolve()
                            if dest_root.resolve() not in dest.parents:
                                continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, dest.open("wb") as out:
                            shutil.copyfileobj(src, out, length=UPLOAD_CHUNK)
                        extracted += 1
            except zipfile.BadZipFile:
                self._json(400, {"error": "invalid zip file"})
                return
            finally:
                tmp.unlink(missing_ok=True)
            manifest = normalize_nomocap_videos(dest_root)
            self._json(200, {
                "extracted": extracted,
                "bucket": bucket,
                "run": tracking_run if bucket == "tracking" else None,
                "indexed": len(manifest),
            })

        def _handle_nomocap_upload_zip(self, query: dict) -> None:
            name = query.get("name", ["run.zip"])[0]
            bucket = tracking_source(query.get("bucket", ["calibration"])[0])
            if not name.lower().endswith(".zip"):
                self._json(400, {"error": "upload a .zip file"})
                return
            remaining = int(self.headers.get("Content-Length", "0"))
            if remaining <= 0:
                self._json(400, {"error": "empty upload"})
                return
            tracking_run = query.get("run", [""])[0] or safe_stem(name)
            if bucket == "tracking":
                tracking_run = _set_selected_tracking_run(cfg, tracking_run)
            dest_root = _tracking_root(cfg, tracking_run) if bucket == "tracking" else cfg.uploads_root
            tmp = cfg.work_root / f".nomocap_upload_{bucket}_{safe_stem(name)}.zip"
            with tmp.open("wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            extracted = 0
            try:
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        rel = nomocap_store_rel(info.filename)
                        if rel is None:
                            if bucket == "calibration" and Path(info.filename).name.lower().endswith(".bin"):
                                cfg.calibration_logs_root.mkdir(parents=True, exist_ok=True)
                                dest = (cfg.calibration_logs_root / (safe_stem(Path(info.filename).name) + ".BIN")).resolve()
                            else:
                                continue
                        else:
                            dest = (dest_root / rel).resolve()
                            if dest_root.resolve() not in dest.parents:
                                continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, dest.open("wb") as out:
                            shutil.copyfileobj(src, out, length=UPLOAD_CHUNK)
                        extracted += 1
            except zipfile.BadZipFile:
                self._json(400, {"error": "invalid zip file"})
                return
            finally:
                tmp.unlink(missing_ok=True)
            manifest = normalize_nomocap_videos(dest_root)
            self._json(200, {
                "extracted": extracted,
                "bucket": bucket,
                "run": tracking_run if bucket == "tracking" else None,
                "indexed": len(manifest),
            })

        def _handle_nomocap_format(self, query: dict) -> None:
            bucket = tracking_source(query.get("bucket", ["calibration"])[0])
            tracking_run = query.get("run", [""])[0] or _selected_tracking_run(cfg) or "tracking_run"
            if bucket == "tracking":
                tracking_run = _set_selected_tracking_run(cfg, tracking_run)
            root = _tracking_root(cfg, tracking_run) if bucket == "tracking" else cfg.uploads_root
            if not any(root.glob("*/video.mp4")):
                self._json(400, {"error": "upload a zip before indexing"})
                return
            manifest = normalize_nomocap_videos(root)
            self._json(200, {
                "bucket": bucket,
                "run": tracking_run if bucket == "tracking" else None,
                "indexed": len(manifest),
            })

        def _handle_tracking_run_select(self, body: dict) -> None:
            run = body.get("run", "") if isinstance(body, dict) else ""
            if not run:
                self._json(400, {"error": "missing 'run'"})
                return
            safe = safe_stem(run)
            if safe not in _tracking_runs(cfg):
                self._json(404, {"error": f"tracking flight {safe!r} not found"})
                return
            _set_selected_tracking_run(cfg, safe)
            self._json(200, {"selected_tracking_run": safe})

        def _handle_reference_log(self, query: dict) -> None:
            run = query.get("run", [""])[0]
            name = query.get("name", [""])[0]
            if run not in _tracking_runs(cfg) or safe_stem(run) != run or not name.lower().endswith(".bin"):
                self._json(400, {"error": "Select a tracking run and an ArduPilot .BIN log."})
                return
            if runner.is_running:
                self._json(409, {"error": "a job is already running"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._json(400, {"error": "empty flight log"})
                return
            root = cfg.work_root / "tracking_logs" / run
            root.mkdir(parents=True, exist_ok=True)
            filename = safe_stem(name) + ".BIN"
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as f:
                    temp = Path(f.name)
                    while length:
                        chunk = self.rfile.read(min(length, UPLOAD_CHUNK))
                        if not chunk:
                            raise ValueError("incomplete flight-log upload")
                        f.write(chunk)
                        length -= len(chunk)
                temp.replace(root / filename)
            except (OSError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
                return
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            self._json(200, {"name": filename})

        def _handle_reference(self, body: dict) -> None:
            run = body.get("run")
            names = body.get("logs")
            if not isinstance(run, str) or run not in _tracking_runs(cfg) or safe_stem(run) != run:
                self._json(400, {"error": "Select an existing tracking run."})
                return
            if not isinstance(names, list) or not names or any(not isinstance(n, str) or n != safe_stem(n) + ".BIN" for n in names):
                self._json(400, {"error": "Upload the flight logs first."})
                return
            logs = [cfg.work_root / "tracking_logs" / run / name for name in set(names)]
            if any(not p.is_file() for p in logs):
                self._json(400, {"error": "A selected flight log is missing."})
                return
            target = cfg.triangulation_root / _tracking_prefix(cfg, run)
            if not (target / "trajectory.npz").is_file():
                self._json(400, {"error": "Triangulate this run before adding a flight-log reference."})
                return
            label = f"tracking:{run}:reference"
            if not runner.start(label, lambda: attach_reference(target, logs, cfg.uploads_calibration_json)):
                self._json(409, {"error": "a job is already running"})
                return
            self._json(202, {"started": label})

        def _handle_tracking_outputs_clear(self, body: dict) -> None:
            run = body.get("run") if isinstance(body, dict) else None
            try:
                tracking_output_targets(cfg, run)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            label = f"tracking:{run}:clear-outputs"
            if not runner.start(label, lambda: clear_tracking_outputs(cfg, run)):
                self._json(409, {"error": "a job is already running"})
                return
            self._json(202, {"started": label})

        def _handle_camera_action(self, body: dict, action: str, source: str) -> None:
            run = body.get("run") if isinstance(body, dict) else None
            camera = body.get("camera") if isinstance(body, dict) else None
            try:
                if camera is None or action not in ("clear-outputs", "detect"):
                    raise ValueError("Select a camera and a valid camera action.")
                if source == "tracking":
                    tracking_output_targets(cfg, run, camera)
                else:
                    calibration_camera_output_targets(cfg, camera)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return

            def job() -> None:
                if action == "detect":
                    annotations = _clicks_json_for(cfg, source, run)
                    if not annotations.is_file() or camera not in load_clicks(annotations):
                        raise ValueError(f"Annotate {camera} with an approved drone click before detecting it.")
                # A retry must also remove partial outputs and any reconstruction
                # made using the other cameras while this camera was being repaired.
                if source == "tracking":
                    clear_tracking_outputs(cfg, run, camera)
                    if action == "detect":
                        _tracking_detect_fn(cfg, run, [camera], runner.set_progress)()
                else:
                    clear_calibration_camera_outputs(cfg, camera)
                    if action == "detect":
                        _upload_detect_fn(cfg, [camera], runner.set_progress)()

            suffix = "detect" if action == "detect" else "clear-camera-outputs"
            label = f"tracking:{run}:{camera}:{suffix}" if source == "tracking" else f"upload:{camera}:{suffix}"
            if not runner.start(label, job):
                self._json(409, {"error": "a job is already running"})
                return
            self._json(202, {"started": label})

        def _handle_calibration_clear(self) -> None:
            try:
                calibration_clear_targets(cfg)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return

            def run() -> None:
                nonlocal result_frames
                clicks.discard_calibration()
                clear_calibration(cfg)
                result_frames = FrameCache()

            if not runner.start("upload:clear-calibration", run):
                self._json(409, {"error": "a job is already running"})
                return
            self._json(202, {"started": "upload:clear-calibration"})

        def _handle_dataset_clear(self) -> None:
            for root in (cfg.dataset_uploads_root, cfg.formatted_root, cfg.detections_root, cfg.triangulation_root):
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
            for path in (cfg.clicks_json, cfg.mocap_raw_clicks_json):
                if path.exists():
                    path.unlink()
            cfg.dataset_uploads_root.mkdir(parents=True, exist_ok=True)
            self._json(200, {"ok": True})

        def _handle_calibration(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            try:
                cams = validate_calibration(json.loads(raw.decode()))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            cfg.uploads_calibration_json.write_text(raw.decode())
            self._json(200, {"n_cameras": len(cams)})

        def _handle_centroid_edit(self, body: dict) -> None:
            rel = body.get("dir", "") if isinstance(body, dict) else ""
            edits = body.get("edits") if isinstance(body, dict) else None
            if not rel or not isinstance(edits, list):
                self._json(400, {"error": "need 'dir' and an 'edits' list"})
                return
            try:
                result = results_api.apply_centroid_edits(cfg.detections_root, rel, edits)
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})
                return
            except (FileNotFoundError, OSError):
                self._send(404, "text/plain", b"detection not found")
                return
            self._json(200, result)

        def _handle_upload_remove(self, body: dict) -> None:
            label = body.get("label", "") if isinstance(body, dict) else ""
            bucket = tracking_source(body.get("bucket", "calibration")) if isinstance(body, dict) else "calibration"
            tracking_run = body.get("tracking_run") if isinstance(body, dict) else None
            if not label:
                self._json(400, {"error": "missing 'label'"})
                return
            if bucket == "tracking":
                remove_tracking_upload(cfg, label, tracking_run)
            else:
                remove_upload(cfg, label)
            self._json(200, {"removed": safe_stem(label)})

        def _handle_calibration_camera(self, body: dict) -> None:
            if not isinstance(body, dict) or not body.get("video"):
                self._json(400, {"error": "camera needs a 'video' label"})
                return
            try:
                n = upsert_camera(cfg.uploads_calibration_json, body)
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"n_cameras": n, "video": body["video"]})

        def _handle_click(self, body: dict) -> None:
            if clicks.active is None:
                self._json(400, {"error": "no active source"})
                return
            try:
                payload = clicks.active.apply_click(
                    int(body["index"]), body["x"], body["y"], body["frame"])
            except (KeyError, ValueError, IndexError):
                self._send(400, "text/plain", b"bad index")
                return
            self._send(200, "application/json", payload)

        def _handle_skip(self, body: dict) -> None:
            if clicks.active is None:
                self._json(400, {"error": "no active source"})
                return
            try:
                payload = clicks.active.toggle_skip(int(body["index"]))
            except (KeyError, ValueError, IndexError):
                self._send(400, "text/plain", b"bad index")
                return
            self._send(200, "application/json", payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"FireTrack dashboard at http://{host}:{port}")
    print(f"  work_root={cfg.work_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

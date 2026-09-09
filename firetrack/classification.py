"""Run-scoped classification of completed no-mocap trajectories."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from .classify import build_classifier_predictor, load_model_manifest
from .results import _safe_subdir, trajectory_file

MIN_VALID_SAMPLES = 10


@lru_cache(maxsize=128)
def _inspect_trajectory(path: str, digest: str) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            times = np.asarray(data["times_s"], dtype=float)
            xyz = np.asarray(data["trajectory_smooth"], dtype=float)
        if times.ndim != 1 or xyz.shape != (len(times), 3):
            raise ValueError("Expected times_s (N,) and trajectory_smooth (N, 3).")
        valid = np.isfinite(times) & np.isfinite(xyz).all(axis=1)
        count = len(np.unique(times[valid]))
        return {
            "trajectory_ready": count >= MIN_VALID_SAMPLES,
            "n_valid_samples": count,
            "trajectory_sha256": digest,
            "reason": "Ready" if count >= MIN_VALID_SAMPLES else
            f"Need at least {MIN_VALID_SAMPLES} finite samples with distinct timestamps; found {count}.",
        }
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        return {"trajectory_ready": False, "reason": f"Invalid trajectory: {exc}"}


class ClassificationService:
    """Own the single in-memory CPU model and run-scoped prediction files."""

    def __init__(self, triangulation_root: Path, output_root: Path) -> None:
        self.triangulation_root = Path(triangulation_root)
        self.output_root = Path(output_root)
        self.cache_dir = Path(os.environ.get(
            "FIRETRACK_CLASSIFIER_CACHE",
            str(self.output_root.parent / ".cache" / "huggingface" / "hub"),
        )).expanduser()
        self._predictor = None
        self._load_error: str | None = None

    def preload(self) -> None:
        """Load once at server startup; a missing optional asset must not kill the UI."""
        if self._predictor is not None:
            return
        try:
            if find_spec("fire_moonshot_classifier") is None:
                raise RuntimeError("The fire-moonshot-classifier package is not installed.")
            print("Loading trajectory classifier on CPU...")
            self._predictor = build_classifier_predictor(
                device="cpu", local_files_only=True, cache_dir=self.cache_dir,
            )
            self._load_error = None
            print("Trajectory classifier ready.")
        except Exception as exc:  # classification is optional until its button is used
            self._load_error = str(exc)
            print(f"Trajectory classifier unavailable: {exc}")

    def _output_path(self, rel: str) -> Path:
        return _safe_subdir(self.output_root, rel) / "prediction.json"

    def clear(self, rel: str) -> None:
        output = self._output_path(rel)
        output.unlink(missing_ok=True)
        try:
            output.parent.rmdir()
        except OSError:
            pass

    def _model_status(self) -> dict:
        manifest = load_model_manifest()
        if self._predictor is not None:
            return {"ready": True, "reason": "Classifier loaded on CPU.", "sha256": manifest["sha256"]}
        return {
            "ready": False,
            "reason": self._load_error or "Classifier has not loaded.",
            "sha256": manifest["sha256"],
        }

    def status(self, rel: str) -> dict:
        model = self._model_status()
        state = {
            "ready": False,
            "trajectory_ready": False,
            "reason": "Triangulate this tracking run first.",
            "result": None,
            "model": model,
        }
        try:
            path = trajectory_file(self.triangulation_root, rel, "npz")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state.update(_inspect_trajectory(str(path), digest))
            summary = json.loads(path.with_name("summary.json").read_text())
            if not isinstance(summary, dict) or summary.get("error") or summary.get("n_raw_triangulated", 0) <= 0:
                raise ValueError("incomplete triangulation")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state.update(trajectory_ready=False, reason="Complete triangulation with valid 3D points first.")

        if state["trajectory_ready"]:
            try:
                result = json.loads(self._output_path(rel).read_text())
                if (
                    isinstance(result, dict)
                    and result.get("trajectory_sha256") == state["trajectory_sha256"]
                    and result.get("model_sha256") == model["sha256"]
                ):
                    state["result"] = result
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        state["ready"] = bool(model["ready"] and state["trajectory_ready"])
        if state["trajectory_ready"] and not model["ready"]:
            state["reason"] = model["reason"]
        return state

    def result(self, rel: str) -> dict | None:
        return self.status(rel).get("result")

    def run(self, rel: str, progress=None) -> dict:
        if self._predictor is None:
            self.preload()
        state = self.status(rel)
        if not state["ready"]:
            raise ValueError(state["reason"])
        path = trajectory_file(self.triangulation_root, rel, "npz")
        if progress:
            progress({"label": "Classifying trajectory", "frame": 0, "total": 1})
        result = self._predictor.predict_trajectory(path, measurement_type="vision")
        if hashlib.sha256(path.read_bytes()).hexdigest() != state["trajectory_sha256"]:
            raise ValueError("Trajectory changed during classification; run Classify again.")
        result = {**result, "trajectory_sha256": state["trajectory_sha256"]}
        output = self._output_path(rel)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        if progress:
            progress({"label": "Classifying trajectory", "frame": 1, "total": 1})
        print(f"Classification: {result['prediction']} ({result['reason']}) -> {output}")
        return result

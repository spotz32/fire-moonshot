"""Load the pinned trajectory classifier and its verified model bundle."""
from __future__ import annotations

import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fire_moonshot_classifier.inference import TrajectoryPredictor


def load_model_manifest() -> dict[str, str]:
    manifest = json.loads(
        resources.files("vendor.moonshot_classifier_assets")
        .joinpath("fire-moonshot-classifier-model.json")
        .read_text(encoding="utf-8")
    )
    required = ("repo_id", "filename", "revision", "sha256")
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), str) or not manifest[key] for key in required
    ):
        raise ValueError(f"Classifier model manifest requires: {', '.join(required)}")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Classifier model revision must be a full Hugging Face commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["sha256"]):
        raise ValueError("Classifier model sha256 must be a SHA256 digest")
    return manifest


def _download_model(
    manifest: dict[str, str],
    *,
    local_files_only: bool,
    cache_dir: Path | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        model_path = Path(hf_hub_download(
            repo_id=manifest["repo_id"],
            filename=manifest["filename"],
            revision=manifest["revision"],
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        ))
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            "The pinned trajectory-classifier model is not cached. Start the app "
            "once while online, or prefetch it with scripts/prepare_classifier.py."
        ) from exc

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"Classifier model SHA256 mismatch: {model_path}")
    return model_path


def download_classifier_model(
    *, local_files_only: bool = False, cache_dir: Path | None = None,
) -> Path:
    """Download and checksum the small model without constructing the network."""
    return _download_model(
        load_model_manifest(), local_files_only=local_files_only, cache_dir=cache_dir,
    )


def build_classifier_predictor(
    *,
    device: str = "cpu",
    batch_size: int = 128,
    local_files_only: bool = False,
    cache_dir: Path | None = None,
) -> TrajectoryPredictor:
    """Build one predictor that the web process can retain and reuse."""
    manifest = load_model_manifest()
    model_path = _download_model(
        manifest, local_files_only=local_files_only, cache_dir=cache_dir,
    )
    from fire_moonshot_classifier.inference import TrajectoryPredictor

    return TrajectoryPredictor(
        model_path,
        device=device,
        batch_size=batch_size,
        expected_sha256=manifest["sha256"],
    )

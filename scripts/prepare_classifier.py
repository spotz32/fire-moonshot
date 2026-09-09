#!/usr/bin/env python3
"""Install the pinned classifier when needed and cache its verified model."""
from __future__ import annotations

import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

WHEEL_URL = (
    "https://github.com/mhcho1994/FIRE_moonshot_classifier/releases/download/v0.1.0/"
    "fire_moonshot_classifier-0.1.0-py3-none-any.whl"
)
WHEEL_SHA256 = "b6c868b9b1eb58e91e849e7e393e8f249820e5617498436b31521fe38968a84f"


def install_classifier() -> None:
    if find_spec("fire_moonshot_classifier") is not None:
        return
    with tempfile.TemporaryDirectory() as temporary:
        wheel = Path(temporary) / Path(WHEEL_URL).name
        digest = hashlib.sha256()
        with urlopen(WHEEL_URL) as response, wheel.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != WHEEL_SHA256:
            raise RuntimeError("Classifier wheel SHA256 mismatch")
        command = [sys.executable, "-m", "pip", "install", "--no-deps"]
        # The released inference code works on 3.9, but v0.1.0's wheel metadata
        # declares >=3.10. This override is limited to that pinned, verified wheel.
        if sys.version_info < (3, 10):
            command.append("--ignore-requires-python")
        subprocess.run([*command, str(wheel)], check=True)


def main() -> int:
    install_classifier()
    from firetrack.classify import download_classifier_model

    repo_root = Path(__file__).resolve().parents[1]
    sibling_work = repo_root.parent / "firetrack-work"
    default_work = sibling_work if sibling_work.exists() else repo_root / "firetrack-work"
    cache_dir = Path(os.environ.get(
        "FIRETRACK_CLASSIFIER_CACHE",
        str(default_work / ".cache" / "huggingface" / "hub"),
    )).expanduser()
    try:
        path = download_classifier_model(local_files_only=True, cache_dir=cache_dir)
    except RuntimeError:
        path = download_classifier_model(cache_dir=cache_dir)
    print(f"Classifier package and model ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

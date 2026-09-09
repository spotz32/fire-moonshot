"""No-mocap classification state, persistence, and run isolation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from firetrack.classification import ClassificationService, load_model_manifest
from firetrack import webui


class TrackingClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cfg = webui.WebConfig(Path(temporary.name))
        self.service = ClassificationService(
            self.cfg.triangulation_root, self.cfg.classification_root,
        )
        self.predictor = Mock()
        self.predictor.predict_trajectory.return_value = {
            "prediction": "PX4",
            "reason": "classified",
            "model_sha256": load_model_manifest()["sha256"],
            "n_windows": 4,
            "n_accepted": 3,
            "vote_fraction": 1.0,
        }
        self.service._predictor = self.predictor

    def trajectory(self, run: str, value: float = 1.0) -> Path:
        directory = self.cfg.triangulation_root / "tracking_uploads" / run
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trajectory.npz"
        np.savez(
            path,
            times_s=np.arange(12, dtype=float),
            trajectory_smooth=np.full((12, 3), value),
        )
        (directory / "summary.json").write_text(json.dumps({"n_raw_triangulated": 12}))
        return path

    def test_selected_run_is_classified_and_persisted(self) -> None:
        path = self.trajectory("run_a")
        self.trajectory("run_b")

        result = self.service.run("tracking_uploads/run_a")

        self.assertEqual(result["prediction"], "PX4")
        self.predictor.predict_trajectory.assert_called_once_with(path, measurement_type="vision")
        self.assertIsNotNone(self.service.result("tracking_uploads/run_a"))
        self.assertIsNone(self.service.result("tracking_uploads/run_b"))

    def test_changed_reconstruction_hides_old_prediction(self) -> None:
        self.trajectory("run_a")
        self.service.run("tracking_uploads/run_a")
        self.trajectory("run_a", value=2.0)

        self.assertIsNone(self.service.result("tracking_uploads/run_a"))

    def test_clear_outputs_removes_classification(self) -> None:
        run = "run_a"
        root = self.cfg.tracking_uploads_root / run / "cam1"
        root.mkdir(parents=True)
        (root / "video.mp4").touch()
        self.trajectory(run)
        self.service.run(f"tracking_uploads/{run}")

        webui.clear_tracking_outputs(self.cfg, run)

        self.assertFalse((self.cfg.classification_root / "tracking_uploads" / run).exists())

    def test_preload_constructs_cpu_predictor_once(self) -> None:
        service = ClassificationService(self.cfg.triangulation_root, self.cfg.classification_root)
        predictor = Mock()
        with patch("firetrack.classification.build_classifier_predictor", return_value=predictor) as build:
            service.preload()
            service.preload()
        build.assert_called_once_with(
            device="cpu", local_files_only=True, cache_dir=service.cache_dir,
        )


if __name__ == "__main__":
    unittest.main()

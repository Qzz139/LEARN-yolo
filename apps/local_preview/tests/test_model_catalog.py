from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from model_catalog import ModelCatalogError, catalog_lines, resolve_model


class ModelCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_path = self.root / "active.onnx"
        self.model_path.write_bytes(b"model")
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "active_model": "small",
                    "classes": ["keyboard", "monitor", "mouse"],
                    "input_size": 640,
                    "models": {
                        "small": {
                            "status": "active_candidate",
                            "artifacts": {"onnx": "active.onnx"},
                        },
                        "medium": {
                            "status": "planned_training",
                            "artifacts": {},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_active_model_is_selected(self) -> None:
        selection = resolve_model(manifest_path=self.manifest_path)
        self.assertEqual(selection.model_id, "small")
        self.assertEqual(selection.model_path, self.model_path.resolve())
        self.assertEqual(selection.labels, ("keyboard", "monitor", "mouse"))

    def test_planned_model_cannot_run_without_artifact(self) -> None:
        with self.assertRaisesRegex(ModelCatalogError, "no ONNX artifact"):
            resolve_model(
                manifest_path=self.manifest_path,
                explicit_model_id="medium",
            )

    def test_explicit_path_has_highest_priority(self) -> None:
        custom_path = self.root / "custom.onnx"
        selection = resolve_model(
            manifest_path=self.manifest_path,
            explicit_model=custom_path,
            explicit_model_id="medium",
        )
        self.assertEqual(selection.model_id, "custom")
        self.assertEqual(selection.model_path, custom_path.resolve())

    def test_catalog_marks_active_model(self) -> None:
        lines = catalog_lines(self.manifest_path)
        self.assertTrue(lines[0].startswith("* small:"))
        self.assertIn("not exported", lines[1])


if __name__ == "__main__":
    unittest.main()

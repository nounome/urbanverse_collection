"""Synthetic file contracts only; these fixtures are not simulation evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urbanverse.dynamic_agents.admission.capture_contract import CAMERAS, audit_capture, audit_capture_isolated


class CaptureContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "metadata").mkdir()
        self.row = dict(frame_index=0, timestamp_s=.02, step_index=1,
                        base_position_world=[0, 0, 1],
                        base_quaternion_wxyz_world=[1, 0, 0, 0], modalities={})
        for name in CAMERAS:
            folder = self.root / name
            folder.mkdir()
            Image.new("RGB", (4, 3)).save(folder / "frame_0000.png")
            np.save(folder / "frame_0000.npy", np.ones((3, 4), dtype=np.float32))
            self.row["modalities"][name] = dict(rgb=f"{name}/frame_0000.png",
                                               depth_float32=f"{name}/frame_0000.npy")

    def check(self, rows=None):
        rows = [self.row] if rows is None else rows
        (self.root / "metadata/frame_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        return audit_capture(self.root)

    def test_complete_records_not_visual_acceptance(self):
        result = self.check()
        self.assertEqual(result["status"], "passed")
        self.assertIn("pixel_timestamp_alignment", result["not_verified"])

    def test_isolated_audit_matches_and_preserves_failure(self):
        self.assertEqual(self.check(), audit_capture_isolated(self.root))
        del self.row['modalities'][CAMERAS[1]]
        self.assertEqual(self.check(), audit_capture_isolated(self.root))

    def test_missing_side_camera(self):
        del self.row["modalities"][CAMERAS[1]]
        self.assertEqual(self.check()["status"], "failed")

    def test_depth_dtype_and_shape(self):
        np.save(self.root / CAMERAS[0] / "frame_0000.npy", np.ones((2, 4)))
        self.assertEqual(self.check()["status"], "failed")

    def test_sky_invalid_is_not_automatic_rejection(self):
        np.save(self.root / CAMERAS[0] / "frame_0000.npy",
                np.full((3, 4), np.inf, dtype=np.float32))
        self.assertEqual(self.check()["status"], "passed_with_warnings")

    def test_invalid_pose_and_duplicate_clock(self):
        self.row["base_quaternion_wxyz_world"] = [0, 0, 0, 0]
        self.assertEqual(self.check()["status"], "failed")
        self.row["base_quaternion_wxyz_world"] = [1, 0, 0, 0]
        second = dict(self.row, frame_index=1)
        self.assertEqual(self.check([self.row, second])["status"], "failed")

    def test_external_path_and_empty_capture(self):
        self.row["modalities"][CAMERAS[0]]["rgb"] = "../frame_0000.png"
        self.assertEqual(self.check()["status"], "failed")
        self.assertEqual(self.check([])["status"], "failed")


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wifo_inference import load_official_base_state, load_upstream_factory


class InferenceSafetyContracts(unittest.TestCase):
    def test_unknown_checkpoint_bytes_are_rejected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unknown.pkl"
            checkpoint.write_bytes(b"not an official checkpoint")
            with patch("torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "official WiFo base asset"):
                    load_official_base_state(checkpoint)
                loader.assert_not_called()

    def test_tensor_load_is_restricted_and_rejects_non_tensor_payloads(self):
        import torch
        with patch("wifo_inference.file_sha256", return_value="5681e6dfbe3ebd358b01510f476b18a25a6616dcb5b88a31595b288aeba22fce"):
            with patch("torch.load", return_value={"weight": torch.ones(1)}) as loader:
                load_official_base_state(Path("checkpoint.pkl"))
                self.assertTrue(loader.call_args.kwargs["weights_only"])
            with patch("torch.load", return_value={"unexpected": "not a tensor"}):
                with self.assertRaisesRegex(ValueError, "tensor-only"):
                    load_official_base_state(Path("checkpoint.pkl"))

    def test_missing_upstream_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "required source module"):
                load_upstream_factory(directory)

    def test_help_needs_no_assets_or_upstream_code(self):
        runner = Path(__file__).resolve().parents[1] / "wifo_inference.py"
        result = subprocess.run([sys.executable, "-B", str(runner), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--upstream-source", result.stdout)


if __name__ == "__main__":
    unittest.main()

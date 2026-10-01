"""Optional CUDA check for the recovered inference kernel used by table runs."""
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA PyTorch required")
class RoPE3DTests(unittest.TestCase):
    def test_fused_matches_reference_with_odd_axis_sizes_and_pose_token(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/croco"))
        from models.curope import cuRoPE3D
        from models.pos_embed_con import RoPE3DContinuous

        torch.manual_seed(0)
        for dim in (64, 72):
            for pose in (0, 1):
                with self.subTest(head_dim=dim, pose_token=pose), torch.no_grad():
                    tokens = torch.randn(2, 4, 37 + pose, dim, device="cuda")
                    positions = torch.randn(2, 37, 3, device="cuda")
                    actual = cuRoPE3D()(tokens, positions)
                    expected = RoPE3DContinuous()(tokens, positions)
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                    if pose:
                        torch.testing.assert_close(actual[:, :, :1], tokens[:, :, :1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

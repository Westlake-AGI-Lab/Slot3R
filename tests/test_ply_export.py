"""Check export geometry and file contents without a GPU or checkpoint."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_ply as export

try:
    import numpy as np
except ImportError:
    np = None


class ExportCLITests(unittest.TestCase):
    def test_help_runs_outside_repository_without_model_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, str(ROOT / "tools/export_ply.py"), "--help"],
                               cwd=tmp, capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--save_trajectory", p.stdout)

    def test_selection_aliases_validation_and_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("frame10.png", "frame2.png", "frame1.JPG", "notes.txt"):
                (root / name).touch()
            (root / "nested.png").mkdir()
            self.assertEqual([p.name for p in export.select_frames(root, 2, 0)], ["frame1.JPG", "frame10.png"])
            self.assertEqual([p.name for p in export.select_frames(root, 1, 2)], ["frame1.JPG", "frame2.png"])
            with self.assertRaises(FileExistsError):
                export.prepare_output(root)
            empty = root / "empty"
            export.prepare_output(empty)
            with self.assertRaises(ValueError):
                export.select_frames(empty, 1, 200)
        required = ["--image_dir", "images", "--output_dir", "out", "--weights", "model.pth"]
        self.assertEqual(export.parse_args(required + ["--model", "ours_rayma"]).model, "vpc_a")
        for option, value in (("--kf_every", "0"), ("--max_frames", "-1"),
                              ("--conf_quantile", "1"), ("--frustum_scale", "nan")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                export.parse_args(required + [option, value])


@unittest.skipUnless(np is not None, "NumPy required for geometry checks")
class ExportGeometryTests(unittest.TestCase):
    def test_collection_uses_shared_points_and_image_rgb(self):
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def __getitem__(self, key): return Tensor(self.data[key])
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
            def permute(self, *axes): return Tensor(self.data.transpose(axes))
        view = {'img': Tensor([[[[-1, 1]], [[0, -1]], [[1, 0]]]])}
        pred = {'pts3d_in_other_view': Tensor([[[[1, 2, 3], [4, 5, 6]]]]),
                'conf': Tensor([[[2, 4]]])}
        xyz, rgb, conf = export.collect_cloud([view], [pred])
        np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(rgb, [[0, .5, 1], [1, 0, .5]])
        np.testing.assert_array_equal(conf, [2, 4])
        with self.assertRaises(ValueError):
            export.collect_cloud([view], [pred, pred])
        pred['conf'] = Tensor([[[2]]])
        with self.assertRaises(ValueError):
            export.collect_cloud([view], [pred])

    def test_filter_keeps_colors_paired_and_binary_ply_roundtrips(self):
        xyz = np.arange(30, dtype=float).reshape(10, 3)
        rgb = np.column_stack([np.arange(10)/10, np.zeros(10), np.ones(10)])
        conf = np.arange(10, dtype=float)
        xyz[0, 0] = np.nan
        a, c, threshold = export.filter_cloud(xyz, rgb, conf, .5, 3, 17)
        b, d, _ = export.filter_cloud(xyz, rgb, conf, .5, 3, 17)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(c, d)
        np.testing.assert_allclose(c[:, 0], a[:, 0]/30)
        self.assertEqual(threshold, 5)
        self.assertTrue((a[:, 0] >= 15).all())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"cloud.ply"
            export.write_cloud(p, a, c)
            header, payload = p.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 3", header)
            self.assertEqual(len(payload), 3*15)
            rows = np.frombuffer(payload, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
            np.testing.assert_allclose(rows["xyz"], a)
            np.testing.assert_array_equal(rows["rgb"], np.rint(c*255).astype('uint8'))
        with self.assertRaises(ValueError):
            export.filter_cloud(np.full((2, 3), np.nan), np.zeros((2, 3)), np.ones(2), 0, 0, 0)

    def test_camera_export_applies_rotation_translation_and_valid_edges(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        poses[1, :3, 3] = [2, 3, 4]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export.export_cameras(root, poses, 12, 2)
            np.testing.assert_array_equal(np.load(root/'c2w.npy'), poses)
            body = (root/'camera_frustums.ply').read_text().split('end_header\n')[1].splitlines()
            vertices = np.array([[float(x) for x in line.split()[:3]] for line in body[:10]])
            np.testing.assert_allclose(vertices[5], [2, 3, 4])
            # First corner (-1.5, -1, 2), rotated +90 around Z then translated.
            np.testing.assert_allclose(vertices[6], [3, 1.5, 6])
            edges = np.array([[int(x) for x in line.split()] for line in body[10:]])
            self.assertEqual(edges.shape, (16, 2))
            self.assertTrue(((edges >= 0) & (edges < 10)).all())
            trajectory = (root/'trajectory.ply').read_text()
            self.assertIn('element edge 1', trajectory)
            self.assertTrue(trajectory.endswith('0 1\n'))
            files = [m.attrib['filename'] for m in ET.parse(root/'scene.mlp').findall('.//MLMesh')]
            self.assertEqual(files, ['cloud.ply', 'trajectory.ply', 'camera_frustums.ply'])
            export.export_cameras(root, poses[:1], 1, 2)
            self.assertIn('element edge 0', (root/'trajectory.ply').read_text())
        with self.assertRaises(ValueError):
            export.export_cameras(Path('.'), np.zeros((2, 3, 3)), 12, 1)


if __name__ == '__main__':
    unittest.main()

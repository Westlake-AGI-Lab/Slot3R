"""CPU-only checks for evaluation wiring; these do not claim numerical reproduction."""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_BIN") or shutil.which("bash")


def functions(path, names, namespace=None):
    """Load pure protocol helpers without importing the CUDA model stack."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing helpers in {path}")
    ns = {"Path": Path, "re": re, "argparse": argparse, **(namespace or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class ProtocolTests(unittest.TestCase):
    def test_python_syntax_and_no_visualization_exports(self):
        forbidden = {"save", "savez", "savez_compressed", "imwrite", "imsave",
                     "write_point_cloud", "savefig", "plot_trajectory", "save_scene_pointclouds"}
        for path in (ROOT / "eval").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(name, forbidden, f"{path}:{node.lineno}")

    def test_evaluated_model_bytes(self):
        for line in (ROOT / "MODEL_CHECKSUMS.sha256").read_text().splitlines():
            expected, name = line.split()
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_kitti_pairs_identity_and_missing_frames(self):
        pair = functions("eval/depth/kitti.py", {"scene_pairs"})["scene_pairs"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root / "image_gathered/scene"
            gt = root / "groundtruth_depth_gathered/scene"
            rgb.mkdir(parents=True)
            gt.mkdir(parents=True)
            (rgb / "drive_image_000002_image_02.png").touch()
            (gt / "drive_groundtruth_depth_000002_image_02.png").touch()
            images, depths = pair(root, "scene")
            self.assertEqual(len(images), 1)
            self.assertIn("groundtruth_depth", depths[0].name)
            (rgb / "drive_image_000003_image_02.png").touch()
            with self.assertRaisesRegex(ValueError, "unmatched"):
                pair(root, "scene")

    def test_pose_dataset_layouts(self):
        get_files = functions("eval/pose/scannet_tum.py", {"scannet_files"})["scannet_files"]
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "test"
            for dataset, folder, pose in (("scannet", "color_90", "pose_90.txt"),
                                           ("tum", "rgb_90", "groundtruth_90.txt")):
                (scene / folder).mkdir(parents=True)
                (scene / folder / "0001.png").touch()
                (scene / pose).touch()
                files, pose_file = get_files(tmp, "test", 1, dataset)
                self.assertEqual(Path(files[0]).parent.name, folder)
                self.assertEqual(pose_file.name, pose)

    def test_malformed_pose_metrics_cannot_be_reported_as_zero(self):
        parse = functions("eval/relpose/evo_utils.py", {"extract_metrics"})["extract_metrics"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.txt"
            path.write_text("inference failed")
            with self.assertRaises(ValueError):
                parse(path)


@unittest.skipUnless(BASH, "Set BASH_BIN to a Linux or Git Bash executable")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.root = Path(self.temp.name)
        self.fake = self.root / "fake-python.sh"
        self.fake.write_text('#!/usr/bin/env bash\n'
                             'printf "ARG=%s\\n" "$@"\n'
                             'printf "MODULE=%s\\n" "${POINT3R_POSE_MODEL_MODULE:-}"\n'
                             'printf "TOKENS=%s\\n" "${POINT3R_SPARSE_MAX_TOKENS:-}"\n'
                             'exit "${FAKE_EXIT:-0}"\n', newline="\n")
        self.fake.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def run_launcher(self, task, dataset, model="core", **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("POINT3R_")}
        env.update(SLOT3R_REPO=ROOT.as_posix(), PYTHON_BIN=self.fake.as_posix(),
                   MODEL=model, DATASET=dataset, WEIGHTS="/fixture/checkpoint.pth",
                   DATA_ROOT="/fixture/data root", OUTPUT_DIR=(self.root / "out").as_posix(),
                   SCENES="scene_one scene_two", **extra)
        return subprocess.run([BASH, (ROOT / f"eval/{task}/run.sh").as_posix()],
                              cwd=self.root, env=env, text=True, capture_output=True)

    def test_all_table_routes_and_model_selection(self):
        modules = {"core": "point3r_kway_frame_sparse_q35_confselect",
                   "vpc_m": "point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose",
                   "vpc_a": "point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose"}
        for model, module in modules.items():
            for task, datasets in {"pointcloud": ("7scenes", "nrgbd"),
                                   "pose": ("scannet", "tum", "sintel"),
                                   "depth": ("bonn", "scannet", "kitti")}.items():
                for dataset in datasets:
                    with self.subTest(model=model, task=task, dataset=dataset):
                        result = self.run_launcher(task, dataset, model)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        args = [line[4:] for line in result.stdout.splitlines() if line.startswith("ARG=")]
                        script = next(arg for arg in args if arg.endswith(".py"))
                        if os.name == "nt" and re.match(r"^/[a-zA-Z]/", script):
                            script = script[1] + ":" + script[2:]
                        self.assertTrue(Path(script).is_file(), script)
                        self.assertIn("/fixture/data root", args)
                        self.assertIn("TOKENS=640", result.stdout)
                        if task != "pointcloud":
                            self.assertIn("MODULE=dust3r." + module, result.stdout)
                        if task == "pose" and dataset == "tum":
                            self.assertEqual(args[args.index("--dataset") + 1], "tum")
                        if task == "depth" and dataset != "kitti":
                            self.assertEqual(args[args.index("--max_depth") + 1], "5")
                            self.assertEqual(args[args.index("--align") + 1], "sequence_scale_shift")

    def test_failure_exit_code_and_existing_output_guard(self):
        result = self.run_launcher("pointcloud", "nrgbd", FAKE_EXIT="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        (self.root / "out/stale_metric.txt").touch()
        result = self.run_launcher("pose", "sintel")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not empty", result.stderr)

    def test_invalid_model_or_dataset_fails(self):
        self.assertNotEqual(self.run_launcher("pose", "sintel", "unknown").returncode, 0)
        self.assertNotEqual(self.run_launcher("depth", "unknown").returncode, 0)

    def test_shell_syntax(self):
        for path in (ROOT / "eval").rglob("*.sh"):
            result = subprocess.run([BASH, "-n", path.as_posix()], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

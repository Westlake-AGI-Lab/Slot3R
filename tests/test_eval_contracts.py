"""CPU-only checks for evaluation wiring; these do not claim numerical reproduction."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for real import checks")
    def test_unified_entrypoints_and_backbone_imports(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT, ROOT / "src/croco", ROOT / "src")))
        commands = [
            [sys.executable, "-m", "eval.mv_recon.launch", "--help"],
            [sys.executable, "eval/mv_recon/launch.py", "--help"],
            [sys.executable, "-c", (
                "import sys, runpy, importlib; "
                "sys.path.insert(0, 'eval/mv_recon'); "
                "ns = runpy.run_path('eval/mv_recon/launch.py'); "
                "[getattr(importlib.import_module(m), 'Point3R') "
                "for m in ns['MODEL_MODULES'].values()]; "
                "import models.blocks"
            )],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_unified_cli_selects_model_dataset_and_defaults(self):
        def literal(path, name):
            tree = ast.parse((ROOT / path).read_text())
            return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        modules = literal("eval/mv_recon/model_registry.py", "MODEL_MODULES")
        aliases = literal("eval/mv_recon/model_registry.py", "MODEL_ALIASES")
        defaults = literal("eval/mv_recon/data.py", "DEFAULT_SCENES")
        parse = functions("eval/mv_recon/launch.py", {"parse_args"}, {
            "os": os, "__file__": str(ROOT / "eval/mv_recon/launch.py"),
            "MODELS": tuple(modules) + ("ghost", "cut3r", "ttt3r"),
            "MODEL_MODULES": modules, "DEFAULT_SCENES": defaults,
            "canonical_model": lambda value: aliases.get(value, value),
        })["parse_args"]
        for model in ("core", "vpc_m", "vpc_a", "point3r"):
            self.assertTrue((ROOT / "src" / (modules[model].replace(".", "/") + ".py")).is_file())
            for dataset, count in (("nrgbd", 9), ("7scenes", 18)):
                args = parse(["--model", model, "--dataset", dataset,
                              "--weights", "weights.pth", "--data_root", "data root"])
                self.assertEqual((args.model, args.dataset, len(args.scenes)), (model, dataset, count))
                self.assertEqual(args.sparse_max_tokens, 640)
        args = parse(["--model", "ours_rayma", "--weights", "weights.pth", "--data_root", ".",
                      "--scenes", "room", "--max_frames", "200"])
        self.assertEqual((args.model, args.scenes, args.max_frames), ("vpc_a", ["room"], 200))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--weights", "weights.pth", "--data_root", ".", "--kf_every", "0"])

    def test_model_profiles_and_switching_do_not_leak_variant_settings(self):
        source = (ROOT / "eval/mv_recon/model_registry.py").read_text()
        assignments = {"MODEL_MODULES", "MODEL_ALIASES", "RAY_SETTINGS", "VARIANT_SETTINGS"}
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in assignments for t in n.targets)]
        ns = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "profiles", "exec"), ns)
        ns = functions("eval/mv_recon/model_registry.py",
                       {"canonical_model", "clear_point3r_env", "configure_model"}, ns)
        args = SimpleNamespace(model="vpc_a", kway_slots=8, theta_bins=16, phi_bins=8,
                               rho_bins=32, sparse_max_tokens=640, sparse_global_anchors=128,
                               sparse_neighbor_range=1, encode_chunk_size=100, drop_quantile=0.25)
        with patch.dict(os.environ, {}, clear=True):
            for model, interval in (("vpc_a", "1"), ("vpc_m", "4"), ("core", None)):
                args.model = model
                ns["configure_model"](args)
                self.assertEqual(os.environ["POINT3R_SPARSE_MAX_TOKENS"], "640")
                self.assertEqual(os.environ["POINT3R_CGMC_DROP_QUANTILE"], "0.25")
                self.assertEqual(os.environ.get("POINT3R_RAY_BANK_UPDATE_EVERY"), interval)
                if model == "vpc_a":
                    self.assertEqual(os.environ["POINT3R_V106_POSE_INPUT_WEIGHT"], "0.15")
                else:
                    self.assertNotIn("POINT3R_V106_POSE_INPUT_WEIGHT", os.environ)
            args.model = "point3r"
            ns["configure_model"](args)
            self.assertNotIn("POINT3R_SPARSE_READOUT", os.environ)
        self.assertEqual([ns["canonical_model"](x) for x in ("ours", "ours_ray", "ours_rayma")],
                         ["core", "vpc_m", "vpc_a"])

    def test_dataset_factory_pairs_available_frames_and_separates_protocols(self):
        class Dataset:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        build = functions("eval/mv_recon/data.py", {"build_dataset"},
                          {"NRGBD": Dataset, "SevenScenes": Dataset})["build_dataset"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("images", "depth"):
                (root / "room" / folder).mkdir(parents=True)
            for i in range(8):
                (root / "room/images" / f"img{i}.png").touch()
                if i != 3:
                    (root / "room/depth" / f"depth{i}.png").touch()
            args = SimpleNamespace(dataset="nrgbd", data_root=tmp, size=512, kf_every=2)
            dataset = build(args, "room")
            self.assertEqual(dataset.tuple_list, ["room 0 2 5 7"])
            self.assertEqual(dataset.kwargs["ROOT"], tmp)
            args.dataset = "7scenes"
            dataset = build(args, "chess/seq-03")
            self.assertEqual(dataset.kwargs["test_id"], "chess")
            self.assertEqual(dataset.kwargs["seq_id"], "seq-03")
            self.assertEqual(dataset.kwargs["kf_every"], 2)
            with self.assertRaises(ValueError):
                build(args, "chess")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for point sampling test")
    def test_pointcloud_sampling_preserves_correspondences(self):
        import numpy as np
        sample = functions("eval/mv_recon/metrics.py",
                           {"subsample_correspondences"}, {"np": np})["subsample_correspondences"]
        pred = np.arange(300).reshape(100, 3)
        gt, color = pred + 1000, pred + 2000
        actual = sample(pred, gt, color, 17, 42)
        self.assertEqual(len(actual[0]), 17)
        self.assertEqual(len(np.unique(actual[0][:, 0])), 17)
        np.testing.assert_array_equal(actual[1] - actual[0], np.full((17, 3), 1000))
        np.testing.assert_array_equal(actual[2] - actual[0], np.full((17, 3), 2000))
        np.testing.assert_array_equal(actual[0], sample(pred, gt, color, 17, 42)[0])
        self.assertIs(sample(pred, gt, color, 0, 42)[0], pred)
        with self.assertRaises(ValueError):
            sample(pred, gt[:-1], color, 17, 42)

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
            for task, datasets in {"mv_recon": ("7scenes", "nrgbd"),
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
                        if task != "mv_recon":
                            self.assertIn("TOKENS=640", result.stdout)
                            self.assertIn("MODULE=dust3r." + module, result.stdout)
                        if task == "mv_recon":
                            self.assertEqual(Path(script).name, "launch.py")
                            self.assertEqual(args[args.index("--model") + 1], model)
                            self.assertEqual(args[args.index("--dataset") + 1], dataset)
                        if task == "pose" and dataset == "tum":
                            self.assertEqual(args[args.index("--dataset") + 1], "tum")
                        if task == "depth" and dataset != "kitti":
                            self.assertEqual(args[args.index("--max_depth") + 1], "5")
                            self.assertEqual(args[args.index("--align") + 1], "sequence_scale_shift")

    def test_failure_exit_code_and_existing_output_guard(self):
        result = self.run_launcher("mv_recon", "nrgbd", FAKE_EXIT="7")
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

"""Unified task CLI, configuration, failure semantics and numerical protocol checks."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.model_config import configure_task_model
from eval.pose.launch import parse_args as pose_args
from eval.depth.launch import parse_args as depth_args, aggregate_metrics
from eval.runtime import run_scenes, mean_metrics

class UnifiedTaskTests(unittest.TestCase):
    def args(self, task, dataset, *extra):
        parse = pose_args if task == "pose" else depth_args
        return parse(["--dataset", dataset, "--data_root", "fixture", "--weights", "weights.pth",
                      "--output_dir", "out", *extra])

    def test_all_task_cli_combinations_and_direct_entrypoints(self):
        for task, datasets in (("pose", ("sintel", "scannet", "tum")), ("depth", ("bonn", "scannet", "kitti"))):
            for dataset in datasets:
                for model in ("core", "vpc_m", "vpc_a", "point3r"):
                    args = self.args(task, dataset, "--model", model)
                    self.assertEqual((args.model, args.dataset, args.sparse_max_tokens), (model, dataset, 640))
            for entry in ([f"eval/{task}/launch.py"], ["-m", f"eval.{task}.launch"]):
                result = subprocess.run([sys.executable, *entry, "--help"], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.args("pose", "tum", "--model", "ours_rayma").model, "vpc_a")

    def test_dataset_defaults_and_invalid_alignment(self):
        for dataset, scale in (("bonn", 5000.), ("scannet", 1000.)):
            args = self.args("depth", dataset)
            self.assertEqual((args.depth_scale, args.align, args.max_depth), (scale, "sequence_scale_shift", 5.))
        args = self.args("depth", "kitti")
        self.assertEqual((args.depth_scale, args.align, args.max_depth), (256., "scale_shift", None))
        for dataset, option in (("kitti", "none"), ("bonn", "metric")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("depth", dataset, "--align", option)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("depth", "kitti", "--max_depth", "5")

    def test_task_model_state_configuration_and_switching(self):
        with patch.dict(os.environ, {}, clear=True):
            for model in ("vpc_a", "vpc_m", "core", "point3r"):
                args = self.args("pose", "sintel", "--model", model)
                configure_task_model(args)
                if model.startswith("vpc"):
                    self.assertEqual(os.environ["POINT3R_LC_STATE_TRANS_STRENGTH"], "30")
                    self.assertEqual(os.environ["POINT3R_LC_ENABLED"], "1")
                elif model == "core":
                    self.assertEqual(os.environ["POINT3R_LC_ENABLED"], "0")
                    self.assertNotIn("POINT3R_V106_POSE_INPUT_WEIGHT", os.environ)
                    self.assertNotIn("POINT3R_LC_STATE_TRANS_STRENGTH", os.environ)
                else:
                    self.assertNotIn("POINT3R_LC_ENABLED", os.environ)
                    self.assertNotIn("POINT3R_SPARSE_READOUT", os.environ)

    def test_incomplete_or_nonfinite_runs_do_not_publish_aggregate(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": fake_torch}):
            args = SimpleNamespace(model="core", output_dir=str(Path(tmp)/"out"))
            with contextlib.redirect_stdout(io.StringIO()):
                status = run_scenes(args, ["good", "bad"], lambda scene: {"Acc": 1. if scene == "good" else float("nan")}, mean_metrics)
            self.assertEqual(status, 1)
            self.assertFalse((Path(args.output_dir)/"result.json").exists())
            self.assertIn("bad", json.loads((Path(args.output_dir)/"scenes.json").read_text())["failures"])
            with self.assertRaises(FileExistsError):
                run_scenes(args, ["new"], lambda s: {"Acc": 0.}, mean_metrics)
            args.output_dir=str(Path(tmp)/"complete")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_scenes(args,["a","b"],lambda s:{"Acc":2. if s=="a" else 4.},mean_metrics),0)
            self.assertEqual(json.loads((Path(args.output_dir)/"result.json").read_text())["Acc"],3.)

    def test_kitti_aggregate_is_pixel_weighted(self):
        rows=[{"AbsRel":1.,"valid_pixels":10.,"frames":1.},{"AbsRel":3.,"valid_pixels":30.,"frames":1.}]
        self.assertEqual(aggregate_metrics(rows,"kitti")["AbsRel"],2.5)
        self.assertEqual(aggregate_metrics(rows,"bonn")["AbsRel"],2.)

@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "Torch/NumPy required")
class NumericalProtocolTests(unittest.TestCase):
    def test_indoor_sequence_alignment_retains_frame_mean(self):
        import numpy as np
        import torch
        from eval.depth.metrics import depth_metrics
        gt=np.linspace(1.,3.,64,dtype=np.float32).reshape(8,8)
        batch=[{"depthmap":torch.tensor(gt)[None,None]} for _ in range(2)]
        preds=[{"depth":torch.tensor((gt-0.2)/2)[None,None]} for _ in range(2)]
        args=SimpleNamespace(model="core",min_depth=1e-3,max_depth=5.,center_crop=0,align="sequence_scale_shift")
        values=depth_metrics(preds,batch,args)
        self.assertLess(values["AbsRel"],1e-6)
        self.assertEqual(values["Delta1"],1.)
        args.align="none"
        self.assertGreater(depth_metrics(preds,batch,args)["AbsRel"],0.5)

    def test_pose_rpe_statistic_selection(self):
        import numpy as np
        from eval.pose.metrics import score_trajectory
        # Use the actual evo evaluator: a curved trajectory plus one displaced frame.
        n=8; poses=np.tile(np.eye(4),(n,1,1));t=np.arange(n,dtype=float)
        poses[:,:3,3]=np.stack((t,t*t/10,np.sin(t)),axis=-1)
        gtpose=poses.copy();poses[3,0,3]+=.3
        from eval.relpose.utils import get_tum_poses
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            mean=score_trajectory(poses,get_tum_poses(gtpose),"fixture",Path(tmp)/"mean.txt","mean")
            rmse=score_trajectory(poses,get_tum_poses(gtpose),"fixture",Path(tmp)/"rmse.txt","rmse")
        self.assertAlmostEqual(rmse["RPE_t"],mean["RPE_t_RMSE"])
        self.assertGreater(rmse["RPE_t"],mean["RPE_t"])

if __name__ == "__main__":
    unittest.main()

"""Sim(3)-aligned pose scoring with dataset-specific RPE reporting."""
from pathlib import Path
import re
import numpy as np

def parse_color90_metric_file(metric_path: str | Path) -> dict[str, float]:
    txt = Path(metric_path).read_text(errors="ignore")

    def block(title: str) -> str:
        m = re.search(title + r".*?(?=\n[A-Z][A-Za-z ]+ w\.r\.t\.|\Z)", txt, re.S)
        return m.group(0) if m else ""

    def val(text: str, key: str) -> float:
        m = re.search(rf"^\s*{key}\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else float("nan")

    ape = block(r"APE w\.r\.t\. translation part")
    rper = block(r"RPE w\.r\.t\. rotation angle")
    rpet = block(r"RPE w\.r\.t\. translation part")
    return {
        "ATE_RMSE": val(ape, "rmse"),
        "ATE_mean": val(ape, "mean"),
        "RPE_t": val(rpet, "mean"),
        "RPE_t_RMSE": val(rpet, "rmse"),
        "RPE_rot": val(rper, "mean"),
        "RPE_rot_RMSE": val(rper, "rmse"),
    }


def score_trajectory(pred, gt, scene, path, rpe_stat):
    from eval.relpose.utils import get_tum_poses
    from eval.relpose.evo_utils import eval_metrics
    eval_metrics(get_tum_poses(pred), gt, seq=scene, filename=str(path))
    result = parse_color90_metric_file(path)
    if rpe_stat == "rmse":
        result["RPE_t"] = result["RPE_t_RMSE"]
        result["RPE_rot"] = result["RPE_rot_RMSE"]
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError(f"Non-finite or malformed pose metrics for {scene}")
    return result

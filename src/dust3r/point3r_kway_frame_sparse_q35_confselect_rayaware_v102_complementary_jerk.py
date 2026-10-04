"""v102: v76 strength on smooth motion, v82k strength on high jerk.

This keeps a single decoder forward.  The only change from v82k is the scalar
weight applied to its existing ray-aware pose-token readout.
"""

import os

import torch
import torch.nn.functional as F

from dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v101_safe_translation import (
    Point3R as V101Point3R,
)


class Point3R(V101Point3R):
    def _ray_pose_input_readout(self, pose_feat, query_pos):
        enabled = os.environ.get("POINT3R_RAY_POSE_INPUT_ONLY", "0").lower() in (
            "1", "true", "yes", "on"
        )
        if (
            not enabled
            or pose_feat is None
            or query_pos is None
            or self._dual_ray_bank is None
        ):
            return pose_feat

        budget = max(16, int(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_TOKENS", "128"
        )))
        smooth_weight = max(0.0, min(0.35, float(os.environ.get(
            "POINT3R_V102_SMOOTH_WEIGHT", "0.15"
        ))))
        jerk_weight = max(0.0, min(0.35, float(os.environ.get(
            "POINT3R_V102_JERK_WEIGHT", "0.025"
        ))))
        temperature = max(0.02, float(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_TEMPERATURE", "0.10"
        )))
        jerk_low = max(0.0, float(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_JERK_LOW", "0.010"
        )))
        jerk_high = max(jerk_low + 1e-6, float(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_JERK_HIGH", "0.035"
        )))

        jerk_gate, motion_jerk = 0.0, 0.0
        history = [pose for pose in self._pose_trajectory if pose is not None]
        if len(history) >= 3:
            r0 = history[-3][:3, :3].float()
            r1 = history[-2][:3, :3].float()
            r2 = history[-1][:3, :3].float()
            increment_0 = r0.transpose(-1, -2) @ r1
            increment_1 = r1.transpose(-1, -2) @ r2
            increment_change = increment_0.transpose(-1, -2) @ increment_1
            cosine = ((torch.trace(increment_change) - 1.0) * 0.5).clamp(-1.0, 1.0)
            motion_jerk = float(torch.acos(cosine).detach().cpu())
            jerk_gate = max(0.0, min(
                1.0, (motion_jerk - jerk_low) / (jerk_high - jerk_low)
            ))
        scheduled_weight = (
            smooth_weight * (1.0 - jerk_gate) + jerk_weight * jerk_gate
        )

        outputs, audit = [], []
        for batch_index in range(pose_feat.shape[0]):
            ray_feat = self._dual_ray_bank["feat"][batch_index]
            ray_pos = self._dual_ray_bank["pos"][batch_index]
            if ray_feat is None or ray_pos is None or ray_pos.numel() == 0:
                outputs.append(pose_feat[batch_index:batch_index + 1])
                audit.append(f"b{batch_index}:empty")
                continue
            query = query_pos[batch_index].float()
            sample_count = min(192, int(query.shape[0]))
            sample_ids = torch.linspace(
                0, query.shape[0] - 1, sample_count, device=query.device
            ).round().long()
            distance = torch.cdist(
                ray_pos.float(), query[sample_ids]
            ).min(dim=1).values
            keep = torch.topk(
                -distance, k=min(budget, int(ray_pos.shape[0])), sorted=False
            ).indices
            candidates = ray_feat[keep].float()
            query_token = pose_feat[batch_index, 0].float()
            similarity = F.cosine_similarity(
                candidates, query_token.unsqueeze(0), dim=-1
            )
            attention = torch.softmax(similarity / temperature, dim=0)
            pooled = (attention[:, None] * candidates).sum(dim=0)
            pooled = F.normalize(pooled, dim=-1) * query_token.norm().clamp_min(1e-6)
            agreement = ((similarity.max() + 1.0) * 0.5).clamp(0.0, 1.0).pow(4.0)
            weight = scheduled_weight * agreement
            mixed = (1.0 - weight) * query_token + weight * pooled
            outputs.append(mixed.to(pose_feat.dtype)[None, None])
            audit.append(
                f"b{batch_index}:weight={float(weight.detach().cpu()):.6f} "
                f"schedule={scheduled_weight:.6f} sim={float(similarity.max().detach().cpu()):.6f}"
            )
        if len(self._pose_trajectory) % 10 == 0:
            self._lc_emit(
                f"[V102_COMPLEMENTARY_JERK] jerk={motion_jerk:.6f} "
                f"jerk_gate={jerk_gate:.6f} " + " ".join(audit)
            )
        return torch.cat(outputs, dim=0)

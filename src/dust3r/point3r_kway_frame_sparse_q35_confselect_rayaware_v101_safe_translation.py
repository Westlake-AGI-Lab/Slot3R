"""One-forward, pose-only, confidence-gated ray translation residual.

The v82k decoder, K-way memory, sparse readout, ConfSelect and dense heads are
unchanged. A second pose-head evaluation (not a decoder forward) produces an
auxiliary camera candidate. Only its translation can affect the output pose.
"""

import os

import torch
import torch.nn.functional as F

import dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose as _v82k_module
from dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose import (
    Point3R as V82KPoint3R,
)


class Point3R(V82KPoint3R):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """Ensure the legacy eval-based checkpoint loader constructs v101.

        The inherited loader resolves ``Point3R`` in the v82k module globals;
        without this scoped substitution it silently instantiates the base
        class and none of this subclass's pose-only logic runs.
        """
        original_class = _v82k_module.Point3R
        _v82k_module.Point3R = cls
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kw)
        finally:
            _v82k_module.Point3R = original_class

    def _v101_ray_pose_candidate(self, pose_token, query_pos):
        if pose_token is None or query_pos is None or self._dual_ray_bank is None:
            return pose_token, None, []

        budget = max(16, int(os.environ.get("POINT3R_V101_TOKENS", "128")))
        token_weight = max(0.0, min(0.35, float(os.environ.get(
            "POINT3R_V101_TOKEN_WEIGHT", "0.15"
        ))))
        temperature = max(0.02, float(os.environ.get(
            "POINT3R_V101_TEMPERATURE", "0.10"
        )))
        similarity_power = max(1.0, float(os.environ.get(
            "POINT3R_V101_SIM_POWER", "4.0"
        )))
        distance_scale = max(1e-3, float(os.environ.get(
            "POINT3R_V101_DISTANCE_SCALE", "0.20"
        )))

        outputs, qualities, diagnostics = [], [], []
        for batch_index in range(pose_token.shape[0]):
            ray_feat = self._dual_ray_bank["feat"][batch_index]
            ray_pos = self._dual_ray_bank["pos"][batch_index]
            if ray_feat is None or ray_pos is None or ray_pos.numel() == 0:
                outputs.append(pose_token[batch_index:batch_index + 1])
                qualities.append(pose_token.new_zeros(()).float())
                diagnostics.append("empty")
                continue

            query = query_pos[batch_index].float()
            sample_count = min(192, int(query.shape[0]))
            sample_ids = torch.linspace(
                0, query.shape[0] - 1, sample_count, device=query.device
            ).round().long()
            nearest = torch.cdist(ray_pos.float(), query[sample_ids]).min(dim=1).values
            keep = torch.topk(
                -nearest, k=min(budget, int(ray_pos.shape[0])), sorted=False
            ).indices
            candidates = ray_feat[keep].float()
            base_token = pose_token[batch_index, 0].float()
            similarity = F.cosine_similarity(
                candidates, base_token.unsqueeze(0), dim=-1
            )
            attention = torch.softmax(similarity / temperature, dim=0)
            pooled = (attention[:, None] * candidates).sum(dim=0)
            pooled = F.normalize(pooled, dim=-1) * base_token.norm().clamp_min(1e-6)

            # Match the calibration used by the verified v76/v82k readout.
            # Raw cross-bank cosine values are centered near zero; treating
            # 0.55 as a hard cosine threshold incorrectly disables all rays.
            feature_quality = (
                (similarity.max() + 1.0) * 0.5
            ).clamp(0.0, 1.0).pow(similarity_power)
            query_center = query.median(dim=0).values
            scene_scale = torch.linalg.norm(
                query - query_center, dim=-1
            ).median().clamp_min(1e-4)
            normalized_distance = nearest[keep].median() / scene_scale
            spatial_quality = torch.exp(
                -0.5 * (normalized_distance / distance_scale).square()
            )
            quality = feature_quality * spatial_quality
            mix = token_weight * quality
            mixed = (1.0 - mix) * base_token + mix * pooled
            outputs.append(mixed.to(pose_token.dtype)[None, None])
            qualities.append(quality)
            diagnostics.append(
                f"sim={float(similarity.max().detach().cpu()):.4f},"
                f"nd={float(normalized_distance.detach().cpu()):.4f},"
                f"q={float(quality.detach().cpu()):.4f}"
            )

        return torch.cat(outputs, dim=0), torch.stack(qualities), diagnostics

    def _apply_geometry_safe_pose_readout(
        self, res, decoder_pose_token, query_pos, frame_i
    ):
        enabled = os.environ.get("POINT3R_V101_SAFE_TRANSLATION", "0").lower() in (
            "1", "true", "yes", "on"
        )
        if not enabled:
            return super()._apply_geometry_safe_pose_readout(
                res, decoder_pose_token, query_pos, frame_i
            )
        if (
            decoder_pose_token is None
            or query_pos is None
            or "camera_pose" not in res
            or not hasattr(self.downstream_head, "forward_pose_only")
        ):
            return res

        candidate_token, token_quality, diagnostics = self._v101_ray_pose_candidate(
            decoder_pose_token, query_pos
        )
        if token_quality is None or float(token_quality.max().detach().cpu()) <= 0.0:
            return res

        try:
            from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera

            candidate_encoding = self.downstream_head.forward_pose_only(
                candidate_token[:, 0].float()
            )
            base_c2w = pose_encoding_to_camera(res["camera_pose"]).float()
            candidate_c2w = pose_encoding_to_camera(candidate_encoding).float()

            relative_rotation = (
                base_c2w[:, :3, :3].transpose(-1, -2)
                @ candidate_c2w[:, :3, :3]
            )
            cosine = (
                (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            rotation_delta_deg = torch.rad2deg(torch.acos(cosine))
            translation_delta = torch.linalg.norm(
                candidate_c2w[:, :3, 3] - base_c2w[:, :3, 3], dim=-1
            )

            rotation_gate_deg = max(1e-3, float(os.environ.get(
                "POINT3R_V101_ROT_GATE_DEG", "3.0"
            )))
            max_step_ratio = max(1e-3, float(os.environ.get(
                "POINT3R_V101_MAX_STEP_RATIO", "1.0"
            )))
            minimum_quality = max(0.0, min(1.0, float(os.environ.get(
                "POINT3R_V101_MIN_QUALITY", "0.15"
            ))))
            max_blend = max(0.0, min(1.0, float(os.environ.get(
                "POINT3R_V101_MAX_BLEND", "1.0"
            ))))

            rotation_gate = (
                1.0 - (rotation_delta_deg / rotation_gate_deg).square()
            ).clamp(0.0, 1.0)
            motion_gate = torch.ones_like(rotation_gate)
            direction_gate = torch.ones_like(rotation_gate)
            if self._pose_trajectory and self._pose_trajectory[-1] is not None:
                previous = self._pose_trajectory[-1].to(base_c2w.device).float()
                previous_t = previous[:3, 3].unsqueeze(0)
                base_step_vector = base_c2w[:, :3, 3] - previous_t
                candidate_step_vector = candidate_c2w[:, :3, 3] - previous_t
                base_step = torch.linalg.norm(base_step_vector, dim=-1).clamp_min(1e-4)
                step_ratio = translation_delta / base_step
                motion_gate = (
                    1.0 - (step_ratio / max_step_ratio).square()
                ).clamp(0.0, 1.0)
                direction_cosine = F.cosine_similarity(
                    base_step_vector, candidate_step_vector, dim=-1, eps=1e-6
                )
                direction_gate = (
                    ((direction_cosine + 1.0) * 0.5).clamp(0.0, 1.0).square()
                )

            finite = (
                torch.isfinite(rotation_delta_deg)
                & torch.isfinite(translation_delta)
                & torch.isfinite(token_quality)
            )
            # token_quality already scales the latent candidate above. Do not
            # multiply it a second time here; that squares the evidence and
            # makes a calibrated v76-strength candidate numerically inert.
            quality = rotation_gate * motion_gate * direction_gate
            quality = torch.where(
                finite & (quality >= minimum_quality),
                quality,
                torch.zeros_like(quality),
            )
            blend = max_blend * quality

            refined = base_c2w.clone()
            refined[:, :3, 3] = (
                (1.0 - blend[:, None]) * base_c2w[:, :3, 3]
                + blend[:, None] * candidate_c2w[:, :3, 3]
            )
            res["camera_pose"] = camera_to_pose_encoding(refined).to(
                res["camera_pose"].dtype
            )

            if frame_i % 10 == 0:
                self._lc_emit(
                    f"[V101_SAFE_TRANSLATION] frame={frame_i} "
                    f"blend={float(blend.mean().detach().cpu()):.6f} "
                    f"token_q={float(token_quality.mean().detach().cpu()):.6f} "
                    f"rot_deg={float(rotation_delta_deg.mean().detach().cpu()):.6f} "
                    f"trans_delta={float(translation_delta.mean().detach().cpu()):.6f} "
                    f"detail={'|'.join(diagnostics)} decoder_forwards=1 geometry_reused=1"
                )
        except Exception as error:
            self._lc_emit(f"[V101_SAFE_TRANSLATION] frame={frame_i} failed={error}")
        return res

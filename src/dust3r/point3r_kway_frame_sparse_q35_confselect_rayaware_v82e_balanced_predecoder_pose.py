import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.point3r_blocks import (
    Block,
    MemoryDecoderBlock,
    DecoderBlock,
    PosDecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """
    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def from_dust3r_to_ours(state_dict):
    
    new_state_dict = OrderedDict()
        
    for k, v in state_dict.items():
        if k.startswith("dec_blocks2."):
            k = k.replace("dec_blocks2.", "dec_blocks_memory.")
        elif k.startswith("downstream_head1.dpt."):
            k = k.replace("downstream_head1.dpt.", "downstream_head.dpt_self.")
        elif k.startswith("downstream_head2.dpt."):
            k = k.replace("downstream_head2.dpt.", "downstream_head.dpt_cross.")
        name = k
        new_state_dict[name] = v
        
    return new_state_dict

def load_model(model_path, device):
    
    print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    ckpt_args = ckpt["args"]
    if isinstance(ckpt_args, dict):
        model_args = ckpt_args["model"]
    else:
        model_args = ckpt_args.model

    args = model_args.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    print(s)
    return net.to(device)


class Point3RConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="dpt",
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        local_mem_size=256,
        memory_dec_num_heads=16,
        memory_update_mode="ordered_kway",
        kway_num_slots=8,
        kway_len_unit=20,
        kway_appearance_threshold=None,
        kway_full_policy="nearest",
        depth_head=False,
        pose_conf_head=False,
        pose_head=False,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.memory_dec_num_heads = memory_dec_num_heads
        self.local_mem_size = local_mem_size
        self.memory_update_mode = memory_update_mode
        self.kway_num_slots = kway_num_slots
        self.kway_len_unit = kway_len_unit
        self.kway_appearance_threshold = kway_appearance_threshold
        self.kway_full_policy = kway_full_policy
        self.depth_head = depth_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs

# thanks to CUT3R (https://github.com/CUT3R)
class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query) 
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class Point3R(CroCoNet):
    config_class = Point3RConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: Point3RConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True)
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,)
        
        self._set_memory_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.memory_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )

        self._set_value_encoder(
            enc_depth=6, 
            enc_embed_dim=1024, 
            out_dim=1024, 
            enc_num_heads=16,
            mlp_ratio=4, 
            norm_layer=self.croco_args.get("norm_layer", None),
        )

        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.memory_attn_head = nn.Sequential(
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim+self.dec_embed_dim),
            nn.GELU(),
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim))
        self.memory_update_mode = "ordered_kway"
        self.kway_num_slots = 8
        self.kway_len_unit = float(getattr(config, "kway_len_unit", 20))
        self.kway_appearance_threshold = getattr(config, "kway_appearance_threshold", None)
        self.kway_full_policy = getattr(config, "kway_full_policy", "nearest")
        self.ordered_theta_bins = 16
        self.ordered_phi_bins = 8
        self.ordered_rho_bins = 32
        self.ordered_rho_min = float(getattr(config, "ordered_rho_min", 0.05))
        self.ordered_rho_max = float(getattr(config, "ordered_rho_max", 50.0))
        self.ordered_neighbor_range = int(getattr(config, "ordered_neighbor_range", 1))
        self.ordered_way_policy = getattr(config, "ordered_way_policy", "appearance")
        self.ordered_app_threshold = getattr(config, "ordered_app_threshold", None)
        self.ordered_update_impl = "tensor"
        self.sparse_readout_enabled = True
        self.sparse_readout_neighbor_range = int(getattr(config, "sparse_readout_neighbor_range", 1))
        self.sparse_readout_global_anchors = 128
        self.sparse_readout_max_tokens = int(getattr(config, "sparse_readout_max_tokens", 0))
        self.sparse_readout_min_tokens = int(getattr(config, "sparse_readout_min_tokens", 1))
        self.sparse_readout_dense_fallback = bool(getattr(config, "sparse_readout_dense_fallback", True))
        self._apply_memory_update_env_overrides()
        print(
            "[CGMC_SPARSE512_EFFECTIVE_CONFIG] "
            f"slots={self.kway_num_slots} "
            f"bins={self.ordered_theta_bins}x{self.ordered_phi_bins}x{self.ordered_rho_bins} "
            f"impl={self.ordered_update_impl} mode={self.memory_update_mode} "
            f"sparse={int(self.sparse_readout_enabled)} anchors={self.sparse_readout_global_anchors}",
            flush=True,
        )
        self.memory_update_stats = []
        self.sparse_readout_stats = []
        self._last_sparse_readout_stats = None
        self._ordered_slot_tables = None
        self._ordered_slot_counts = None
        self._ordered_slot_confs = None
        self._ordered_slot_rays = None
        self._ordered_slot_times = None
        self._ordered_slot_locals = None
        self._static_gate_dynamic_ema = None
        self._static_gate_dynamic_latched = None
        # v28: keep the ray-diverse and stable CGMC policies alive in
        # parallel.  A late scene-policy decision must never inherit a memory
        # that was already mutated by the other policy.
        self._dual_ray_bank = None
        self._dual_stable_bank = None
        # v54: an optional second decoder reads the ray-aware bank only for
        # the pose head.  The stable decoder remains authoritative for dense
        # geometry, recurrent pose memory and persistent-memory writes.
        self._last_pose_only_dec = None

        # ---- Loop Closure state (default OFF, set POINT3R_LC_ENABLED=1 to enable) ----
        # _lc_pos[j]:      list of [M,3] world pos，每帧 q25 后写入，有容量上限
        # _lc_fid[j]:      list of [M] frame id
        # _lc_local[j]:    list of [M,3] 相机坐标系局部坐标 R^T @ (x_w - t)
        # _pose_trajectory: list of [4,4] c2w，每帧保存
        # _lc_candidates:  list of (fi, fj, T_ij_4x4, info_6x6)，独立估计的几何约束
        self._lc_pos = None
        self._lc_fid = None
        self._lc_local = None
        self._lc_ray = None
        self._lc_feat = None
        self._pose_trajectory = None
        self._lc_candidates = []
        self._lc_odometry_edges = {}
        self._lc_rotation_edges = {}
        self._lc_translation_edges = {}
        self._lc_diag = None
        self._lc_pgo_executed = False
        self._online_pose_refinements = {}
        # v70 keeps raw (unrefined) v56 and token trajectories only for one
        # frame.  The token branch may correct the current relative motion,
        # but its low-frequency drift is never accumulated into the anchor.
        self._v70_prev_stable_c2w = None
        self._v70_prev_token_c2w = None
        self._v71_base_score_ema = None
        self._v71_token_score_ema = None
        self._v71_use_token = None
        self._v74_relative_gain_ema = None
        self._v74_effect_size_ema = None
        self._v74_observability_ema = None
        self._v74_use_token = None

        self.confselect_merge_threshold = float(os.environ.get("POINT3R_CONFSELECT_MERGE_THRESHOLD", "0.90"))
        self.confselect_default_conf = float(os.environ.get("POINT3R_CONFSELECT_DEFAULT_CONF", "1.0"))
        self.confselect_decay_gamma = float(os.environ.get("POINT3R_CONFSELECT_DECAY_GAMMA", "1.0"))
        self.confselect_conf_margin = float(os.environ.get("POINT3R_CONFSELECT_CONF_MARGIN", "0.0"))
        self.confselect_collect_stats = os.environ.get("POINT3R_CONFSELECT_STATS", "0").lower() in ("1", "true", "yes", "on")
        print(
            "[V82E_BALANCED_PREDECODER_POSE] "
            "single_forward=1 post_decoder_only_default=0 "
            "drop_quantile=0.35 merge_threshold="
            f"{self.confselect_merge_threshold:.3f}",
            flush=True,
        )

        self.set_freeze(config.freeze)

    def _apply_memory_update_env_overrides(self):
        mode = os.environ.get("POINT3R_MEMORY_UPDATE_MODE")
        if mode:
            self.memory_update_mode = mode
        num_slots = os.environ.get("POINT3R_KWAY_NUM_SLOTS")
        if num_slots:
            self.kway_num_slots = int(num_slots)
        len_unit = os.environ.get("POINT3R_KWAY_LEN_UNIT")
        if len_unit:
            self.kway_len_unit = float(len_unit)
        appearance_threshold = os.environ.get("POINT3R_KWAY_APPEARANCE_THRESHOLD")
        if appearance_threshold:
            self.kway_appearance_threshold = float(appearance_threshold)
        full_policy = os.environ.get("POINT3R_KWAY_FULL_POLICY")
        if full_policy:
            self.kway_full_policy = full_policy
        theta_bins = os.environ.get("POINT3R_ORDERED_THETA_BINS")
        if theta_bins:
            self.ordered_theta_bins = int(theta_bins)
        phi_bins = os.environ.get("POINT3R_ORDERED_PHI_BINS")
        if phi_bins:
            self.ordered_phi_bins = int(phi_bins)
        rho_bins = os.environ.get("POINT3R_ORDERED_RHO_BINS")
        if rho_bins:
            self.ordered_rho_bins = int(rho_bins)
        rho_min = os.environ.get("POINT3R_ORDERED_RHO_MIN")
        if rho_min:
            self.ordered_rho_min = float(rho_min)
        rho_max = os.environ.get("POINT3R_ORDERED_RHO_MAX")
        if rho_max:
            self.ordered_rho_max = float(rho_max)
        ordered_neighbor_range = os.environ.get("POINT3R_ORDERED_NEIGHBOR_RANGE")
        if ordered_neighbor_range:
            self.ordered_neighbor_range = int(ordered_neighbor_range)
        ordered_way_policy = os.environ.get("POINT3R_ORDERED_WAY_POLICY")
        if ordered_way_policy:
            self.ordered_way_policy = ordered_way_policy
        ordered_app_threshold = os.environ.get("POINT3R_ORDERED_APP_THRESHOLD")
        if ordered_app_threshold:
            self.ordered_app_threshold = float(ordered_app_threshold)
        ordered_update_impl = os.environ.get("POINT3R_ORDERED_UPDATE_IMPL")
        if ordered_update_impl:
            self.ordered_update_impl = ordered_update_impl
        sparse_readout = os.environ.get("POINT3R_SPARSE_READOUT")
        if sparse_readout is not None:
            self.sparse_readout_enabled = sparse_readout.lower() not in ("0", "false", "no", "off", "")
        sparse_neighbor_range = os.environ.get("POINT3R_SPARSE_NEIGHBOR_RANGE")
        if sparse_neighbor_range:
            self.sparse_readout_neighbor_range = int(sparse_neighbor_range)
        sparse_global_anchors = os.environ.get("POINT3R_SPARSE_GLOBAL_ANCHORS")
        if sparse_global_anchors:
            self.sparse_readout_global_anchors = int(sparse_global_anchors)
        sparse_max_tokens = os.environ.get("POINT3R_SPARSE_MAX_TOKENS")
        if sparse_max_tokens:
            self.sparse_readout_max_tokens = int(sparse_max_tokens)
        sparse_min_tokens = os.environ.get("POINT3R_SPARSE_MIN_TOKENS")
        if sparse_min_tokens:
            self.sparse_readout_min_tokens = int(sparse_min_tokens)
        sparse_dense_fallback = os.environ.get("POINT3R_SPARSE_DENSE_FALLBACK")
        if sparse_dense_fallback is not None:
            self.sparse_readout_dense_fallback = sparse_dense_fallback.lower() not in ("0", "false", "no", "off", "")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(Point3R, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.pts_patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_memory_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_memory = dec_depth
        self.dec_embed_dim_memory = dec_embed_dim
        self.decoder_embed_memory = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_memory = nn.ModuleList(
            [
                MemoryDecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_memory = norm_layer(dec_embed_dim)
    
    def _set_value_encoder(
        self,
        enc_depth, 
        enc_embed_dim, 
        out_dim, 
        enc_num_heads,
        mlp_ratio, 
        norm_layer
    ):
        self.value_encoder = nn.ModuleList(
            [
                Block(
                    enc_embed_dim, 
                    enc_num_heads, 
                    mlp_ratio, 
                    qkv_bias=True, 
                    norm_layer=norm_layer, 
                    rope=self.rope
                )
                for i in range(enc_depth)
            ]
        )
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_memory") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_memory")] = value
        if not any(k.startswith("pts_patch_embed") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("patch_embed"):
                    new_ckpt[key.replace("patch_embed", "pts_patch_embed")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze): 
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_views(self, views, img_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        img_mask = torch.stack(
            [view["img_mask"] for view in views], dim=0
        ) 
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        
        if selected_imgs.size(0) > 0:
            encode_chunk_size = int(os.environ.get("POINT3R_ENCODE_CHUNK_SIZE", "100"))
            if encode_chunk_size > 0 and selected_imgs.shape[0] > encode_chunk_size:
                img_out_chunks = []
                img_pos_chunks = []
                for start in range(0, selected_imgs.shape[0], encode_chunk_size):
                    end = min(start + encode_chunk_size, selected_imgs.shape[0])
                    chunk_out, chunk_pos, _ = self._encode_image(
                        selected_imgs[start:end], selected_shapes[start:end]
                    )
                    img_out_chunks.append(chunk_out)
                    img_pos_chunks.append(chunk_pos)
                img_out = [
                    torch.cat([chunk[level] for chunk in img_out_chunks], dim=0)
                    for level in range(len(img_out_chunks[0]))
                ]
                img_pos = torch.cat(img_pos_chunks, dim=0)
            else:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
        full_pos[img_masks_flat] += img_pos
        
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, i, mask_memory, f_memory, pos_memory, f_img, pos_img, f_pose, point3r_tag=False):
        if isinstance(f_memory, torch.Tensor):
            assert f_memory.shape[-1] == self.dec_embed_dim
        else:
            assert f_memory[-1].shape[-1] == self.dec_embed_dim
        
        final_output = [(f_memory, f_img)] 
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
        final_output.append((f_memory, f_img))
        
        for blk_memory, blk_img in zip(self.dec_blocks_memory, self.dec_blocks):
            f_memory, _ = blk_memory(i, *final_output[-1][::+1], mask_memory, pos_memory, pos_img, point3r_tag=point3r_tag)
            f_img, _ = blk_img(i, *final_output[-1][::-1], mask_memory, pos_img, pos_memory, point3r_tag=point3r_tag)
            final_output.append((f_memory, f_img))
        del final_output[1] 
        final_output[-1] = (
            self.dec_norm_memory(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_memory(self, image_tokens, image_pos):
        
        memory_feat = self.decoder_embed_memory(image_tokens)
        return memory_feat, None

    def _sparse_readout_select_indices(
        self,
        memory_pos_j,
        query_pos_j,
        valid_mask_j=None,
        memory_ray_j=None,
        camera_pose=None,
    ):
        device = memory_pos_j.device
        if memory_pos_j is None or query_pos_j is None:
            return None, {"sparse_enabled": 0, "sparse_applied": 0}

        if valid_mask_j is None:
            valid_idx = torch.arange(memory_pos_j.shape[0], device=device)
        else:
            valid_idx = torch.nonzero(valid_mask_j.bool(), as_tuple=False).flatten()

        valid_memory = int(valid_idx.numel())
        if valid_memory == 0:
            return None, {
                "sparse_enabled": int(self.sparse_readout_enabled),
                "sparse_applied": 0,
                "sparse_memory": 0,
                "sparse_selected": 0,
                "sparse_ratio": 0.0,
                "sparse_local": 0,
                "sparse_anchor": 0,
                "sparse_fallback": 1,
            }

        mem_pos = memory_pos_j[valid_idx]
        mem_keys = self._ordered_pack_bins(self._ordered_spatial_bins(mem_pos))
        query_bins = torch.unique(self._ordered_spatial_bins(query_pos_j), dim=0)

        radius = max(0, int(self.sparse_readout_neighbor_range))
        offsets = []
        for dt in range(-radius, radius + 1):
            for dp in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    offsets.append((dt, dp, dr))
        offsets = torch.tensor(offsets, dtype=torch.long, device=device)

        neighbor_bins = (query_bins[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        keep = (
            (neighbor_bins[:, 0] >= 0)
            & (neighbor_bins[:, 0] < int(self.ordered_theta_bins))
            & (neighbor_bins[:, 1] >= 0)
            & (neighbor_bins[:, 1] < int(self.ordered_phi_bins))
            & (neighbor_bins[:, 2] >= 0)
            & (neighbor_bins[:, 2] < int(self.ordered_rho_bins))
        )
        neighbor_bins = neighbor_bins[keep]
        if neighbor_bins.numel() == 0:
            local_idx = valid_idx.new_empty((0,))
        else:
            neighbor_keys = torch.sort(torch.unique(self._ordered_pack_bins(neighbor_bins))).values
            search_pos = torch.searchsorted(neighbor_keys, mem_keys).clamp(max=max(0, neighbor_keys.numel() - 1))
            local_mask = neighbor_keys[search_pos] == mem_keys
            local_idx = valid_idx[torch.nonzero(local_mask, as_tuple=False).flatten()]

        anchor_count = max(0, int(self.sparse_readout_global_anchors))
        if anchor_count > 0 and valid_memory > 0:
            anchor_count = min(anchor_count, valid_memory)
            anchor_pos = torch.linspace(0, valid_memory - 1, steps=anchor_count, device=device).round().long()
            anchor_idx = valid_idx[anchor_pos]
        else:
            anchor_idx = valid_idx.new_empty((0,))

        selected = torch.unique(torch.cat((local_idx, anchor_idx), dim=0))
        ray_rank_applied = 0
        max_tokens = int(self.sparse_readout_max_tokens)
        if max_tokens > 0 and selected.numel() > max_tokens:
            if anchor_idx.numel() >= max_tokens:
                selected = torch.unique(anchor_idx)[:max_tokens]
            else:
                is_anchor = torch.zeros(memory_pos_j.shape[0], dtype=torch.bool, device=device)
                is_anchor[anchor_idx] = True
                rest = selected[~is_anchor[selected]]
                rest_budget = max_tokens - torch.unique(anchor_idx).numel()
                ray_rank = os.environ.get(
                    "POINT3R_RAY_READOUT_RANK", "0"
                ).lower() in ("1", "true", "yes", "on")
                joint_ray_readout = os.environ.get(
                    "POINT3R_RAY_JOINT_READOUT", "0"
                ).lower() in ("1", "true", "yes", "on")
                if (
                    (ray_rank or joint_ray_readout)
                    and rest_budget > 0
                    and memory_ray_j is not None
                    and memory_ray_j.shape[0] == memory_pos_j.shape[0]
                    and camera_pose is not None
                ):
                    camera_center = camera_pose[:3, 3].to(
                        device=device, dtype=memory_pos_j.dtype
                    )
                    current_rays = F.normalize(
                        memory_pos_j[rest] - camera_center.unsqueeze(0), dim=-1
                    )
                    stored_rays = F.normalize(
                        memory_ray_j[rest].to(device=device).float(), dim=-1
                    )
                    # Prefer pointers whose stored observation direction is
                    # compatible with the current viewpoint.  This makes ray
                    # metadata affect the actual decoder readout instead of
                    # only the memory-write policy.
                    ray_scores = (
                        current_rays.float() * stored_rays
                    ).sum(dim=-1)
                    if joint_ray_readout and query_pos_j.numel() > 0:
                        # The write policy alone cannot make the decoder use
                        # the right pointer. Rank candidates jointly by their
                        # proximity to the current query geometry and viewing
                        # direction, while keeping the selected features
                        # unscaled for decoder distribution compatibility.
                        query_sample_count = min(256, int(query_pos_j.shape[0]))
                        query_sample_ids = torch.linspace(
                            0, query_pos_j.shape[0] - 1,
                            query_sample_count, device=device,
                        ).round().long()
                        query_sample = query_pos_j[query_sample_ids].float()
                        spatial_distance = torch.cdist(
                            memory_pos_j[rest].float(), query_sample
                        ).min(dim=1).values
                        spatial_scale = spatial_distance.median().clamp_min(1e-4)
                        lambda_pos = float(os.environ.get(
                            "POINT3R_RAY_READOUT_LAMBDA_POS", "1.0"
                        ))
                        lambda_ray = float(os.environ.get(
                            "POINT3R_RAY_READOUT_LAMBDA_RAY", "0.25"
                        ))
                        joint_score = (
                            -lambda_pos * spatial_distance / spatial_scale
                            + lambda_ray * ray_scores
                        )
                        rank = torch.argsort(joint_score, descending=True)
                    else:
                        rank = torch.argsort(ray_scores, descending=True)
                    rest = rest[rank[:rest_budget]]
                    ray_rank_applied = 1
                else:
                    rest = rest[:rest_budget]
                selected = torch.cat((torch.unique(anchor_idx), rest), dim=0)

        fallback = 0
        min_tokens = max(1, int(self.sparse_readout_min_tokens))
        if selected.numel() < min_tokens:
            if self.sparse_readout_dense_fallback:
                selected = valid_idx
                fallback = 1
            else:
                selected = valid_idx[: min(min_tokens, valid_memory)]

        stats = {
            "sparse_enabled": int(self.sparse_readout_enabled),
            "sparse_applied": int(selected.numel() < valid_memory),
            "sparse_memory": valid_memory,
            "sparse_selected": int(selected.numel()),
            "sparse_ratio": float(selected.numel()) / float(max(1, valid_memory)),
            "sparse_local": int(local_idx.numel()),
            "sparse_anchor": int(anchor_idx.numel()),
            "sparse_fallback": int(fallback),
            "sparse_ray_rank": int(ray_rank_applied),
            "sparse_joint_ray": int(
                os.environ.get("POINT3R_RAY_JOINT_READOUT", "0").lower()
                in ("1", "true", "yes", "on")
            ),
        }
        return selected, stats

    def _maybe_sparse_readout(self, memory_feat, memory_pos, mask_memory, query_pos):
        self._last_sparse_readout_stats = None
        if not self.sparse_readout_enabled:
            return memory_feat, memory_pos, mask_memory, False
        if memory_pos is None or query_pos is None or not isinstance(memory_feat, torch.Tensor):
            return memory_feat, memory_pos, mask_memory, False

        feat_list = []
        pos_list = []
        mask_list = []
        stats_list = []
        applied_any = False

        for j in range(memory_feat.shape[0]):
            valid_mask_j = mask_memory[j].bool() if mask_memory is not None else None
            memory_ray_j = None
            if (
                self._ordered_slot_rays is not None
                and j < len(self._ordered_slot_rays)
            ):
                memory_ray_j = self._ordered_slot_rays[j]
            camera_pose = (
                self._pose_trajectory[-1]
                if self._pose_trajectory
                else None
            )
            selected, stats = self._sparse_readout_select_indices(
                memory_pos[j],
                query_pos[j],
                valid_mask_j,
                memory_ray_j=memory_ray_j,
                camera_pose=camera_pose,
            )
            stats["batch"] = int(j)
            stats_list.append(stats)

            if selected is None:
                feat_j = memory_feat[j] if mask_memory is None else memory_feat[j][valid_mask_j]
                pos_j = memory_pos[j] if mask_memory is None else memory_pos[j][valid_mask_j]
            else:
                feat_j = memory_feat[j][selected]
                pos_j = memory_pos[j][selected]
                applied_any = applied_any or bool(stats.get("sparse_applied", 0))
            feat_list.append(feat_j)
            pos_list.append(pos_j)

        if not applied_any:
            self._last_sparse_readout_stats = stats_list
            self.sparse_readout_stats.append(stats_list)
            return memory_feat, memory_pos, mask_memory, False

        max_len = max(x.shape[0] for x in feat_list)
        for j in range(len(feat_list)):
            pad_len = max_len - feat_list[j].shape[0]
            if pad_len > 0:
                feat_pad = feat_list[j].new_zeros((pad_len, feat_list[j].shape[-1]))
                pos_pad = pos_list[j].new_zeros((pad_len, pos_list[j].shape[-1]))
                feat_list[j] = torch.cat((feat_list[j], feat_pad), dim=0)
                pos_list[j] = torch.cat((pos_list[j], pos_pad), dim=0)
            valid = torch.ones(max_len - pad_len, device=feat_list[j].device)
            invalid = torch.zeros(pad_len, device=feat_list[j].device)
            mask_list.append(torch.cat((valid, invalid), dim=0))

        self._last_sparse_readout_stats = stats_list
        self.sparse_readout_stats.append(stats_list)
        return (
            torch.stack(feat_list, dim=0),
            torch.stack(pos_list, dim=0),
            torch.stack(mask_list, dim=0),
            True,
        )

    def _hybrid_dual_bank_readout(self, query_pos):
        """Build a bounded decoder context from both persistent banks.

        The earlier dual-bank implementation updated two banks but exposed
        only one of them to the decoder.  Consequently the ray-aware branch
        often had no effect on the pose token.  This readout reserves an
        explicit quota for spatially stable pointers and a second quota for
        viewpoint-compatible ray pointers.  A small uniform anchor set from
        each bank preserves global context.
        """
        enabled = os.environ.get(
            "POINT3R_RAY_HYBRID_READOUT", "0"
        ).lower() in ("1", "true", "yes", "on")
        if (
            not enabled
            or query_pos is None
            or self._dual_stable_bank is None
            or self._dual_ray_bank is None
        ):
            return None

        total_budget = max(
            32,
            int(os.environ.get(
                "POINT3R_RAY_HYBRID_TOKENS",
                str(self.sparse_readout_max_tokens or 640),
            )),
        )
        stable_fraction = max(0.1, min(0.9, float(os.environ.get(
            "POINT3R_RAY_HYBRID_STABLE_FRAC", "0.60"
        ))))
        anchor_fraction = max(0.0, min(0.4, float(os.environ.get(
            "POINT3R_RAY_HYBRID_ANCHOR_FRAC", "0.10"
        ))))
        ray_weight = float(os.environ.get(
            "POINT3R_RAY_HYBRID_RAY_WEIGHT", "0.25"
        ))

        feat_batches, pos_batches, mask_batches = [], [], []
        audit_parts = []
        batch_size = int(query_pos.shape[0])
        for j in range(batch_size):
            stable_feat = self._dual_stable_bank["feat"][j]
            stable_pos = self._dual_stable_bank["pos"][j]
            ray_feat = self._dual_ray_bank["feat"][j]
            ray_pos = self._dual_ray_bank["pos"][j]
            ray_dirs = self._dual_ray_bank["ray"][j]
            if (
                stable_feat is None or stable_pos is None
                or ray_feat is None or ray_pos is None
                or stable_pos.numel() == 0 or ray_pos.numel() == 0
            ):
                return None

            query_j = query_pos[j].float()
            sample_count = min(192, int(query_j.shape[0]))
            sample_ids = torch.linspace(
                0, query_j.shape[0] - 1, sample_count,
                device=query_j.device,
            ).round().long()
            query_sample = query_j[sample_ids]

            stable_quota = min(
                int(stable_pos.shape[0]),
                int(round(total_budget * stable_fraction)),
            )
            ray_quota = min(
                int(ray_pos.shape[0]), total_budget - stable_quota
            )
            if stable_quota + ray_quota < total_budget:
                stable_quota = min(
                    int(stable_pos.shape[0]), total_budget - ray_quota
                )

            def _rank_with_anchors(pos, score, quota):
                if quota <= 0:
                    return torch.empty(
                        (0,), dtype=torch.long, device=pos.device
                    )
                anchor_count = min(
                    quota, int(round(quota * anchor_fraction))
                )
                if anchor_count > 0:
                    anchors = torch.linspace(
                        0, pos.shape[0] - 1, anchor_count,
                        device=pos.device,
                    ).round().long().unique()
                else:
                    anchors = torch.empty(
                        (0,), dtype=torch.long, device=pos.device
                    )
                is_anchor = torch.zeros(
                    pos.shape[0], dtype=torch.bool, device=pos.device
                )
                is_anchor[anchors] = True
                ranked = torch.argsort(score, descending=True)
                ranked = ranked[~is_anchor[ranked]]
                return torch.cat(
                    (anchors, ranked[: max(0, quota - anchors.numel())]),
                    dim=0,
                )

            stable_distance = torch.cdist(
                stable_pos.float(), query_sample
            ).min(dim=1).values
            stable_idx = _rank_with_anchors(
                stable_pos, -stable_distance, stable_quota
            )

            ray_distance = torch.cdist(
                ray_pos.float(), query_sample
            ).min(dim=1).values
            ray_scale = ray_distance.median().clamp_min(1e-4)
            ray_score = -ray_distance / ray_scale
            camera_pose = self._pose_trajectory[-1] if self._pose_trajectory else None
            if (
                camera_pose is not None
                and ray_dirs is not None
                and ray_dirs.shape[0] == ray_pos.shape[0]
            ):
                camera_center = camera_pose[:3, 3].to(
                    device=ray_pos.device, dtype=ray_pos.dtype
                )
                current_ray = F.normalize(
                    ray_pos - camera_center.unsqueeze(0), dim=-1
                )
                stored_ray = F.normalize(ray_dirs.float(), dim=-1)
                ray_score = ray_score + ray_weight * (
                    current_ray.float() * stored_ray
                ).sum(dim=-1)
            ray_idx = _rank_with_anchors(ray_pos, ray_score, ray_quota)

            feat_j = torch.cat(
                (stable_feat[stable_idx], ray_feat[ray_idx]), dim=0
            )
            pos_j = torch.cat(
                (stable_pos[stable_idx], ray_pos[ray_idx]), dim=0
            )
            feat_batches.append(feat_j)
            pos_batches.append(pos_j)
            mask_batches.append(torch.ones(
                feat_j.shape[0], device=feat_j.device
            ))
            audit_parts.append(
                f"b{j}:stable={stable_idx.numel()}/{stable_pos.shape[0]} "
                f"ray={ray_idx.numel()}/{ray_pos.shape[0]}"
            )

        max_len = max(int(value.shape[0]) for value in feat_batches)
        for j in range(batch_size):
            pad = max_len - int(feat_batches[j].shape[0])
            if pad > 0:
                feat_batches[j] = torch.cat((
                    feat_batches[j],
                    feat_batches[j].new_zeros((pad, feat_batches[j].shape[-1])),
                ), dim=0)
                pos_batches[j] = torch.cat((
                    pos_batches[j],
                    pos_batches[j].new_zeros((pad, 3)),
                ), dim=0)
                mask_batches[j] = torch.cat((
                    mask_batches[j], mask_batches[j].new_zeros((pad,))
                ), dim=0)
        if len(self._pose_trajectory) % 10 == 0:
            self._lc_emit(
                "[HYBRID_READOUT] " + " ".join(audit_parts)
                + f" total={max_len}"
            )
        return (
            torch.stack(feat_batches, dim=0),
            torch.stack(pos_batches, dim=0),
            torch.stack(mask_batches, dim=0),
        )

    def _ray_pose_input_readout(self, pose_feat, query_pos):
        """Training-free ray-bank readout for the input pose token only.

        This deliberately does not alter the decoder memory context: dense
        image tokens keep the verified stable sparse K-way readout.  The
        selected ray features are pooled by pose-token cosine attention and
        injected before the sole decoder forward.
        """
        enabled = os.environ.get(
            "POINT3R_RAY_POSE_INPUT_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if (
            not enabled or pose_feat is None or query_pos is None
            or self._dual_ray_bank is None
        ):
            return pose_feat
        budget = max(16, int(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_TOKENS", "128"
        )))
        max_weight = max(0.0, min(0.35, float(os.environ.get(
            "POINT3R_RAY_POSE_INPUT_MAX_WEIGHT", "0.025"
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
        motion_gate = 0.0
        motion_jerk = 0.0
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
            motion_gate = max(0.0, min(
                1.0, (motion_jerk - jerk_low) / (jerk_high - jerk_low)
            ))
        outputs = []
        audit = []
        for j in range(pose_feat.shape[0]):
            ray_feat = self._dual_ray_bank["feat"][j]
            ray_pos = self._dual_ray_bank["pos"][j]
            if ray_feat is None or ray_pos is None or ray_pos.numel() == 0:
                outputs.append(pose_feat[j:j + 1])
                audit.append(f"b{j}:empty")
                continue
            query = query_pos[j].float()
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
            query_token = pose_feat[j, 0].float()
            similarity = F.cosine_similarity(
                candidates, query_token.unsqueeze(0), dim=-1
            )
            attention = torch.softmax(similarity / temperature, dim=0)
            pooled = (attention[:, None] * candidates).sum(dim=0)
            pooled = F.normalize(pooled, dim=-1) * query_token.norm().clamp_min(1e-6)
            agreement = ((similarity.max() + 1.0) * 0.5).clamp(0.0, 1.0).pow(4.0)
            weight = max_weight * agreement * motion_gate
            mixed = (1.0 - weight) * query_token + weight * pooled
            outputs.append(mixed.to(pose_feat.dtype)[None, None])
            audit.append(
                f"b{j}:tokens={keep.numel()} weight={float(weight.detach().cpu()):.6f} "
                f"sim={float(similarity.max().detach().cpu()):.6f} "
                f"motion_gate={motion_gate:.6f} jerk={motion_jerk:.6f}"
            )
        if len(self._pose_trajectory) % 10 == 0:
            self._lc_emit("[SINGLE_POSE_READOUT] " + " ".join(audit))
        return torch.cat(outputs, dim=0)

    def _apply_geometry_safe_pose_readout(
        self, res, decoder_pose_token, query_pos, frame_i
    ):
        """Refine only ``camera_pose`` after the sole decoder forward.

        v77 injected the ray-bank residual before the decoder.  Because the
        decoder pose token self-attends with image tokens and later conditions
        ``pts3d_in_other_view``, that changed dense geometry as a side effect.
        Here the verified decoder and both point-map branches run unchanged.
        The ray residual is applied only to the final pose token, which is then
        passed directly through the existing pose head.
        """
        enabled = os.environ.get(
            "POINT3R_RAY_POSE_POST_DECODER_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if (
            not enabled
            or decoder_pose_token is None
            or query_pos is None
            or "camera_pose" not in res
            or not hasattr(self.downstream_head, "forward_pose_only")
        ):
            return res
        mixed_pose_token = self._ray_pose_input_readout(
            decoder_pose_token, query_pos
        )
        res["camera_pose"] = self.downstream_head.forward_pose_only(
            mixed_pose_token[:, 0].float()
        )
        if frame_i % 10 == 0:
            delta = torch.linalg.norm(
                (mixed_pose_token - decoder_pose_token).float(), dim=-1
            ).mean()
            self._lc_emit(
                f"[GEOMETRY_SAFE_POSE_READOUT] frame={frame_i} "
                f"token_delta={float(delta.detach().cpu()):.6f} "
                "pointmaps_reused=1 decoder_forwards=1"
            )
        return res

    def _recurrent_rollout(
        self,
        i,
        mask_memory,
        memory_feat,
        memory_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        pose_only_feat=None,
        point3r_tag=False,
    ):
        pose_only_ensemble = os.environ.get(
            "POINT3R_RAY_POSE_ONLY_ENSEMBLE", "0"
        ).lower() in ("1", "true", "yes", "on")
        hybrid = self._hybrid_dual_bank_readout(current_pos)
        self._last_pose_only_dec = None
        pose_input_only = os.environ.get(
            "POINT3R_RAY_POSE_INPUT_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        post_decoder_only = os.environ.get(
            "POINT3R_RAY_POSE_POST_DECODER_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if pose_input_only and not post_decoder_only:
            pose_feat = self._ray_pose_input_readout(pose_feat, current_pos)
        if hybrid is not None and not pose_only_ensemble and not pose_input_only:
            read_memory_feat, read_memory_pos, read_mask_memory = hybrid
            sparse_applied = True
        else:
            read_memory_feat, read_memory_pos, read_mask_memory, sparse_applied = self._maybe_sparse_readout(
                memory_feat,
                memory_pos,
                mask_memory,
                current_pos,
            )
        new_memory_feat, dec = self._decoder(
            i,
            read_mask_memory,
            read_memory_feat, read_memory_pos, current_feat, current_pos, pose_feat,
            point3r_tag=point3r_tag,
        )
        new_memory_feat = new_memory_feat[-1]
        if sparse_applied:
            new_memory_feat = memory_feat
        if hybrid is not None and pose_only_ensemble:
            hybrid_feat, hybrid_pos, hybrid_mask = hybrid
            _, pose_only_dec = self._decoder(
                i,
                hybrid_mask,
                hybrid_feat,
                hybrid_pos,
                current_feat,
                current_pos,
                pose_only_feat if pose_only_feat is not None else pose_feat,
                point3r_tag=point3r_tag,
            )
            self._last_pose_only_dec = pose_only_dec
        return new_memory_feat, dec

    @staticmethod
    def _pose_only_consensus_fuse(
        base_c2w,
        ray_c2w,
        max_weight_env="POINT3R_RAY_POSE_MAX_WEIGHT",
        default_max_weight="0.35",
    ):
        """Safely mix a ray-bank pose candidate into the stable prediction.

        The ray branch is an auxiliary observation, never a replacement.  A
        continuous agreement score gives it at most ``MAX_WEIGHT`` influence
        and exactly zero influence outside the rotation/translation gates.
        Rotations are projected back to SO(3) after chordal interpolation.
        """
        rot_gate_deg = max(1e-4, float(os.environ.get(
            "POINT3R_RAY_POSE_ROT_GATE_DEG", "4.0"
        )))
        trans_gate = max(1e-6, float(os.environ.get(
            "POINT3R_RAY_POSE_TRANS_GATE", "0.20"
        )))
        max_weight = max(0.0, min(0.5, float(os.environ.get(
            max_weight_env, default_max_weight
        ))))

        base = base_c2w.float()
        ray = ray_c2w.float()
        rel = base[:, :3, :3].transpose(-1, -2) @ ray[:, :3, :3]
        cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        rot_deg = torch.rad2deg(torch.acos(cosine))
        trans_delta = torch.linalg.norm(
            ray[:, :3, 3] - base[:, :3, 3], dim=-1
        )
        finite = torch.isfinite(rot_deg) & torch.isfinite(trans_delta)
        rot_agreement = (1.0 - (rot_deg / rot_gate_deg).square()).clamp(0.0, 1.0)
        trans_agreement = (1.0 - (trans_delta / trans_gate).square()).clamp(0.0, 1.0)
        weight = max_weight * rot_agreement * trans_agreement * finite.float()

        mixed = base.clone()
        w = weight[:, None, None]
        rotation_chord = (1.0 - w) * base[:, :3, :3] + w * ray[:, :3, :3]
        u, _, vh = torch.linalg.svd(rotation_chord)
        rotation = u @ vh
        det = torch.linalg.det(rotation)
        if (det < 0).any():
            u = u.clone()
            u[det < 0, :, -1] *= -1.0
            rotation = u @ vh
        mixed[:, :3, :3] = rotation
        wt = weight[:, None]
        mixed[:, :3, 3] = (
            (1.0 - wt) * base[:, :3, 3] + wt * ray[:, :3, 3]
        )
        return mixed.to(dtype=base_c2w.dtype), weight, rot_deg, trans_delta

    @staticmethod
    def _pose_pointmap_consistency(candidate_res, candidate_c2w):
        """Robust, scale-normalized same-frame pose/pointmap residual.

        Local and world pointmaps are pixel-aligned outputs of the same head,
        so this avoids the unreliable cross-frame pointer correspondence used
        by the rejected v52/v53/v55 branches.  GT is never consulted.
        """
        required = (
            "pts3d_in_self_view",
            "pts3d_in_other_view",
            "conf_self",
            "conf",
        )
        if any(key not in candidate_res for key in required):
            return None
        local = candidate_res["pts3d_in_self_view"].float().flatten(1, -2)
        world = candidate_res["pts3d_in_other_view"].float().flatten(1, -2)
        conf_self = candidate_res["conf_self"].float().flatten(1)
        conf_world = candidate_res["conf"].float().flatten(1)
        pose = candidate_c2w.float()
        scores = []
        dispersions = []
        observabilities = []
        max_points = max(128, int(os.environ.get(
            "POINT3R_RAY_POSE_GEOM_MAX_POINTS", "2048"
        )))
        for batch_index in range(local.shape[0]):
            local_b = local[batch_index]
            world_b = world[batch_index]
            weight_b = (
                torch.log(conf_self[batch_index].clamp_min(1.0 + 1e-6))
                * torch.log(conf_world[batch_index].clamp_min(1.0 + 1e-6))
            )
            valid = (
                torch.isfinite(local_b).all(dim=-1)
                & torch.isfinite(world_b).all(dim=-1)
                & torch.isfinite(weight_b)
                & (weight_b > 0)
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() < 32:
                scores.append(torch.tensor(
                    float("inf"), device=local.device, dtype=torch.float32
                ))
                dispersions.append(torch.tensor(
                    float("inf"), device=local.device, dtype=torch.float32
                ))
                observabilities.append(torch.tensor(
                    0.0, device=local.device, dtype=torch.float32
                ))
                continue
            if valid_indices.numel() > max_points:
                valid_weight = weight_b[valid_indices]
                keep = torch.topk(
                    valid_weight, k=max_points, largest=True, sorted=False
                ).indices
                valid_indices = valid_indices[keep]
            local_sel = local_b[valid_indices]
            world_sel = world_b[valid_indices]
            weight_sel = weight_b[valid_indices].clamp_min(1e-8)
            predicted_world = (
                local_sel @ pose[batch_index, :3, :3].transpose(-1, -2)
                + pose[batch_index, :3, 3]
            )
            residual = torch.linalg.norm(predicted_world - world_sel, dim=-1)
            scale = torch.median(
                torch.linalg.norm(local_sel, dim=-1)
            ).clamp_min(1e-4)
            normalized = residual / scale
            # Trim the worst 20% even within the high-confidence subset.
            cutoff = torch.quantile(normalized, 0.80)
            robust = normalized <= cutoff
            robust_weight = weight_sel[robust]
            score = (
                (robust_weight * normalized[robust]).sum()
                / robust_weight.sum().clamp_min(1e-8)
            )
            scores.append(score)
            robust_residual = normalized[robust]
            residual_median = torch.median(robust_residual)
            residual_mad = 1.4826 * torch.median(
                torch.abs(robust_residual - residual_median)
            )
            dispersions.append(residual_mad.clamp_min(1e-6))

            normalized_weight = weight_sel / weight_sel.sum().clamp_min(1e-8)
            local_center = (
                normalized_weight[:, None] * local_sel
            ).sum(dim=0, keepdim=True)
            centered_local = local_sel - local_center
            covariance = centered_local.transpose(0, 1) @ (
                normalized_weight[:, None] * centered_local
            )
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            observability = eigenvalues[0] / eigenvalues[-1].clamp_min(1e-8)
            observabilities.append(observability)
        return (
            torch.stack(scores),
            torch.stack(dispersions),
            torch.stack(observabilities),
        )

    def _pose_pointmap_procrustes(self, candidate_res):
        """Estimate a causal same-frame SE(3) directly from token pointmaps."""
        required = ("pts3d_in_self_view", "pts3d_in_other_view", "conf_self", "conf")
        if any(key not in candidate_res for key in required):
            return None
        local = candidate_res["pts3d_in_self_view"].float().flatten(1, -2)
        world = candidate_res["pts3d_in_other_view"].float().flatten(1, -2)
        conf = (
            torch.log(candidate_res["conf_self"].float().flatten(1).clamp_min(1.0 + 1e-6))
            * torch.log(candidate_res["conf"].float().flatten(1).clamp_min(1.0 + 1e-6))
        )
        estimates = []
        max_points = max(128, int(os.environ.get("POINT3R_RAY_POSE_GEOM_MAX_POINTS", "2048")))
        for b in range(local.shape[0]):
            valid = (
                torch.isfinite(local[b]).all(dim=-1)
                & torch.isfinite(world[b]).all(dim=-1)
                & torch.isfinite(conf[b])
                & (conf[b] > 0)
            )
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if indices.numel() < 32:
                return None
            if indices.numel() > max_points:
                indices = indices[torch.topk(conf[b, indices], max_points, sorted=False).indices]
            src, dst = local[b, indices], world[b, indices]
            transform = self._umeyama_se3(src, dst)
            if transform is None:
                return None
            residual = torch.linalg.norm(
                src @ transform[:3, :3].transpose(-1, -2) + transform[:3, 3] - dst,
                dim=-1,
            )
            keep = residual <= torch.quantile(residual, 0.80)
            transform = self._umeyama_se3(src[keep], dst[keep])
            if transform is None:
                return None
            estimates.append(transform)
        return torch.stack(estimates).to(dtype=local.dtype)

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)
  
    def enc_pts_value(self, pts, shape):
        out, pos = self.pts_patch_embed(pts.permute(0, 3, 1, 2), true_shape=shape)
        for block in self.value_encoder:
            out = block(out, pos)
        out = self.value_norm(out)
        out = self.value_out(out)
        return out

    def _record_point3r_memory_stats(self, i, img_pos, memory_pos, mode, batch_stats=None):
        if batch_stats is None:
            if isinstance(memory_pos, torch.Tensor):
                final_lengths = [int(memory_pos.shape[1]) for _ in range(memory_pos.shape[0])]
            else:
                final_lengths = [int(memory_pos_j.shape[0]) for memory_pos_j in memory_pos]
            batch_stats = []
            for batch_idx, final_memory in enumerate(final_lengths):
                batch_stats.append(
                    {
                        "frame": int(i),
                        "batch": int(batch_idx),
                        "mode": mode,
                        "new": int(img_pos.shape[1]),
                        "miss": int(img_pos.shape[1]) if i == 0 else 0,
                        "hit_merge": 0,
                        "final_memory": final_memory,
                    }
                )
        self.memory_update_stats.append(batch_stats)

    def _forward_addmemory(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        ):
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)
        
        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key+feat_pts)
        memory_add = memory_add.float()

        if i == 0:
            memory_feat = memory_add
            init_memory_feat = memory_feat.clone().detach()
            chosen_pts = img_pos
        else:
            memory_feat = torch.cat((memory_feat, memory_add), dim=1)
            init_memory_feat = torch.cat((init_memory_feat, memory_add.clone().detach()), dim=1)
            chosen_pts = torch.cat((memory_pos, img_pos), dim=1)

        self._record_point3r_memory_stats(
            i,
            img_pos,
            chosen_pts,
            mode="point3r_append",
        )
          
        return memory_feat, chosen_pts, init_memory_feat, img_pos






    def _ordered_spatial_bins(self, pos):
        pos_f = pos.float()
        radius = torch.norm(pos_f, dim=-1).clamp_min(float(self.ordered_rho_min))
        theta = torch.atan2(pos_f[..., 1], pos_f[..., 0])
        xy_radius = torch.norm(pos_f[..., :2], dim=-1).clamp_min(torch.finfo(pos_f.dtype).eps)
        phi = torch.atan2(pos_f[..., 2], xy_radius)

        theta_norm = (theta + torch.pi) / (2 * torch.pi)
        phi_norm = (phi + torch.pi / 2) / torch.pi
        rho_min = max(float(self.ordered_rho_min), torch.finfo(pos_f.dtype).eps)
        rho_max = max(float(self.ordered_rho_max), rho_min * 1.01)
        rho_norm = torch.log(radius / rho_min) / torch.log(
            torch.tensor(rho_max / rho_min, device=pos_f.device, dtype=pos_f.dtype)
        )

        theta_bin = torch.floor(theta_norm * self.ordered_theta_bins).long()
        phi_bin = torch.floor(phi_norm * self.ordered_phi_bins).long()
        rho_bin = torch.floor(rho_norm * self.ordered_rho_bins).long()
        theta_bin = theta_bin.clamp(0, self.ordered_theta_bins - 1)
        phi_bin = phi_bin.clamp(0, self.ordered_phi_bins - 1)
        rho_bin = rho_bin.clamp(0, self.ordered_rho_bins - 1)
        return torch.stack((theta_bin, phi_bin, rho_bin), dim=-1)

    def _ordered_bin_bits(self):
        theta_bits = (max(1, int(self.ordered_theta_bins)) - 1).bit_length()
        phi_bits = (max(1, int(self.ordered_phi_bins)) - 1).bit_length()
        rho_bits = (max(1, int(self.ordered_rho_bins)) - 1).bit_length()
        return theta_bits, phi_bits, rho_bits

    def _ordered_num_buckets(self):
        theta_bits, phi_bits, rho_bits = self._ordered_bin_bits()
        return 1 << (theta_bits + phi_bits + rho_bits)



    def _ordered_pack_bins(self, bins):
        bins_l = bins.long()
        _, phi_bits, rho_bits = self._ordered_bin_bits()
        theta_part = bins_l[..., 0] << (phi_bits + rho_bits)
        phi_part = bins_l[..., 1] << rho_bits
        return theta_part | phi_part | bins_l[..., 2]


















    def _ordered_update_single_tensor(
        self,
        memory_feat_j,
        memory_pos_j,
        new_feat_j,
        img_pos_j,
        slot_table_j,
        slot_count_j,
        slot_conf_j=None,
        new_conf_j=None,
        memory_ray_j=None,
        new_ray_j=None,
        memory_time_j=None,
        new_time_j=None,
        memory_local_j=None,
        new_local_j=None,
        ray_adaptive_strength=None,
        rayaware_override=None,
    ):
        num_slots = max(1, int(self.kway_num_slots))
        num_new = int(img_pos_j.shape[0])
        num_buckets = int(self._ordered_num_buckets())
        device = img_pos_j.device
        rayaware = os.environ.get("POINT3R_RAYAWARE_UPDATE", "0").lower() in ("1", "true", "yes", "on")
        if rayaware_override is not None:
            rayaware = bool(rayaware_override)
        paper_update = rayaware and os.environ.get(
            "POINT3R_RAY_PAPER_UPDATE", "0"
        ).lower() in ("1", "true", "yes", "on")
        ray_kway_update = rayaware and os.environ.get(
            "POINT3R_RAY_KWAY_DIVERSE_UPDATE", "0"
        ).lower() in ("1", "true", "yes", "on")
        stats = {
            "new": num_new,
            "miss": 0,
            "hit_free": 0,
            "hit_full": 0,
            "app_reject": 0,
            "final_memory": 0,
            "lookup": "ordered_tensor_bucket",
            "update_impl": "tensor",
            "rows": 0,
            "neighbor_rows": 0,
            "candidate_sum": 0,
            "candidate_max": num_slots,
            "neighbor_hit": 0,
            "hit_update": 0,
            "skip_low_conf": 0,
            "merge_sim": 0,
            "append_distinct": 0,
            "replace_low_conf": 0,
            "empty_store": 0,
            "distinct_store": 0,
            "merge_better_conf": 0,
            "policy_reject": 0,
            "detail_stats": bool(self.confselect_collect_stats),
            "merge_threshold": float(self.confselect_merge_threshold),
            "num_buckets": num_buckets,
            "recent_keys": [],
        }

        if slot_conf_j is None or slot_conf_j.shape[0] != memory_pos_j.shape[0]:
            slot_conf_j = torch.full(
                (memory_pos_j.shape[0],),
                float(self.confselect_default_conf),
                dtype=torch.float32,
                device=device,
            )
        if slot_conf_j.numel() > 0 and self.confselect_decay_gamma < 1.0:
            slot_conf_j = slot_conf_j * max(0.0, self.confselect_decay_gamma)

        if slot_table_j is None or slot_table_j.shape != (num_buckets, num_slots):
            slot_table_j = torch.full((num_buckets, num_slots), -1, dtype=torch.long, device=device)
            slot_count_j = torch.zeros(num_buckets, dtype=torch.long, device=device)
            if memory_pos_j.shape[0] > 0:
                mem_keys = self._ordered_pack_bins(self._ordered_spatial_bins(memory_pos_j)).long()
                sorted_keys, sort_idx = torch.sort(mem_keys)
                unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
                starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
                ranks = torch.arange(sort_idx.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
                keep = ranks < num_slots
                if keep.numel() > 0:
                    slot_table_j[sorted_keys[keep], ranks[keep]] = sort_idx[keep]
                    slot_count_j[unique_keys] = torch.minimum(counts, counts.new_full((), num_slots))

        if num_new == 0:
            stats["rows"] = -1
            stats["final_memory"] = int(memory_pos_j.shape[0])
            return memory_feat_j, memory_pos_j, slot_table_j, slot_count_j, slot_conf_j, memory_ray_j, memory_time_j, memory_local_j, stats

        if new_conf_j is None:
            new_conf_j = torch.full((num_new,), float(self.confselect_default_conf), dtype=torch.float32, device=device)
        else:
            new_conf_j = new_conf_j.reshape(-1).to(device=device, dtype=torch.float32)
            if new_conf_j.shape[0] != num_new:
                new_conf_j = torch.full((num_new,), float(self.confselect_default_conf), dtype=torch.float32, device=device)
        new_conf_j = torch.nan_to_num(new_conf_j, nan=0.0, posinf=0.0, neginf=0.0)

        if memory_ray_j is None or memory_ray_j.shape[0] != memory_pos_j.shape[0]:
            memory_ray_j = F.normalize(memory_pos_j.float(), dim=-1) if memory_pos_j.shape[0] > 0 else memory_pos_j.new_empty((0, 3)).float()
        if new_ray_j is None or new_ray_j.shape[0] != num_new:
            new_ray_j = F.normalize(img_pos_j.float(), dim=-1)
        else:
            new_ray_j = F.normalize(new_ray_j.float(), dim=-1)
        if memory_time_j is None or memory_time_j.shape[0] != memory_pos_j.shape[0]:
            memory_time_j = torch.full((memory_pos_j.shape[0],), -1, dtype=torch.long, device=device)
        if new_time_j is None:
            new_time_j = torch.zeros((num_new,), dtype=torch.long, device=device)
        else:
            new_time_j = torch.as_tensor(new_time_j, dtype=torch.long, device=device).reshape(-1)
            if new_time_j.numel() == 1:
                new_time_j = new_time_j.expand(num_new)
        if memory_local_j is None or memory_local_j.shape[0] != memory_pos_j.shape[0]:
            memory_local_j = memory_pos_j.float().clone()
        if new_local_j is None or new_local_j.shape[0] != num_new:
            new_local_j = img_pos_j.float()

        img_bins = self._ordered_spatial_bins(img_pos_j)
        img_keys = self._ordered_pack_bins(img_bins).long()
        sorted_keys, sort_idx = torch.sort(img_keys)
        unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

        # Capacity and insertion always refer to the query's own bucket.  RayAway
        # matching, however, must search adjacent buckets: two nearby 3D points
        # commonly straddle a quantization boundary.
        old_counts_for_token = slot_count_j[img_keys].clamp_max(num_slots)
        own_cand_idx = slot_table_j[img_keys].clamp_min(0)
        if paper_update or ray_kway_update:
            neighbor_range = max(0, int(os.environ.get("POINT3R_RAY_NEIGHBOR_RANGE", "1")))
            axis = torch.arange(-neighbor_range, neighbor_range + 1, device=device, dtype=torch.long)
            offsets = torch.cartesian_prod(axis, axis, axis)
            if offsets.ndim == 1:
                offsets = offsets.unsqueeze(0)
            neighbor_bins = img_bins[:, None, :] + offsets[None, :, :]
            # Azimuth is periodic; elevation/radius are not.
            neighbor_bins[..., 0] = torch.remainder(neighbor_bins[..., 0], int(self.ordered_theta_bins))
            neighbor_valid = (
                (neighbor_bins[..., 1] >= 0)
                & (neighbor_bins[..., 1] < int(self.ordered_phi_bins))
                & (neighbor_bins[..., 2] >= 0)
                & (neighbor_bins[..., 2] < int(self.ordered_rho_bins))
            )
            neighbor_bins[..., 1] = neighbor_bins[..., 1].clamp(0, int(self.ordered_phi_bins) - 1)
            neighbor_bins[..., 2] = neighbor_bins[..., 2].clamp(0, int(self.ordered_rho_bins) - 1)
            neighbor_keys = self._ordered_pack_bins(neighbor_bins).long()
            neighbor_counts = slot_count_j[neighbor_keys].clamp_max(num_slots)
            cand_idx = slot_table_j[neighbor_keys].clamp_min(0).reshape(num_new, -1)
            local_ids = torch.arange(num_slots, device=device, dtype=torch.long).view(1, 1, -1)
            valid_slots = (
                (local_ids < neighbor_counts.unsqueeze(-1))
                & neighbor_valid.unsqueeze(-1)
            ).reshape(num_new, -1)
        else:
            cand_idx = own_cand_idx
            local_ids = torch.arange(num_slots, device=device, dtype=torch.long).unsqueeze(0)
            valid_slots = local_ids < old_counts_for_token.unsqueeze(1)
        has_existing = valid_slots.any(dim=1)
        best_sim = torch.full((num_new,), -float("inf"), dtype=torch.float32, device=device)
        best_local = torch.zeros((num_new,), dtype=torch.long, device=device)
        best_conf = torch.full((num_new,), -float("inf"), dtype=torch.float32, device=device)
        min_conf = torch.full((num_new,), float("inf"), dtype=torch.float32, device=device)
        best_ray_sim = torch.ones((num_new,), dtype=torch.float32, device=device)
        best_pos_dist = torch.full((num_new,), float("inf"), dtype=torch.float32, device=device)
        best_target = torch.full((num_new,), -1, dtype=torch.long, device=device)
        pointer_pair_mask = None
        row_ids = torch.arange(num_new, device=device)

        if memory_feat_j.shape[0] > 0:
            cand_feat = memory_feat_j[cand_idx]
            query = F.normalize(new_feat_j.float(), dim=-1)
            keys = F.normalize(cand_feat.float(), dim=-1)
            sims = (query.unsqueeze(1) * keys).sum(dim=-1).masked_fill(~valid_slots, -float("inf"))
            if paper_update or ray_kway_update:
                cand_pos = memory_pos_j[cand_idx].float()
                pos_dist = torch.norm(img_pos_j.float().unsqueeze(1) - cand_pos, dim=-1)
                cand_ray = memory_ray_j[cand_idx]
                ray_sims = (new_ray_j.unsqueeze(1) * cand_ray).sum(dim=-1).clamp(-1.0, 1.0)
                lambda_pos = float(os.environ.get("POINT3R_RAY_LAMBDA_POS", "1.0"))
                lambda_ang = float(os.environ.get("POINT3R_RAY_LAMBDA_ANG", "0.1"))
                joint_dist = lambda_pos * pos_dist + lambda_ang * (1.0 - ray_sims)
                joint_dist = joint_dist.masked_fill(~valid_slots, float("inf"))
                _, best_local = joint_dist.min(dim=1)
                best_sim = sims[row_ids, best_local]
                best_ray_sim = ray_sims[row_ids, best_local]
                best_pos_dist = pos_dist[row_ids, best_local]
            elif rayaware:
                cand_ray = memory_ray_j[cand_idx]
                ray_sims = (new_ray_j.unsqueeze(1) * cand_ray).sum(dim=-1).masked_fill(~valid_slots, -1.0)
                ray_weight = float(os.environ.get("POINT3R_RAY_MATCH_WEIGHT", "0.10"))
                joint_sims = sims - ray_weight * (1.0 - ray_sims)
                _, best_local = joint_sims.max(dim=1)
                best_sim = sims[row_ids, best_local]
                best_ray_sim = ray_sims[row_ids, best_local]
            else:
                best_sim, best_local = sims.max(dim=1)
            best_target = cand_idx[row_ids, best_local]
            own_valid = torch.arange(num_slots, device=device).unsqueeze(0) < old_counts_for_token.unsqueeze(1)
            cand_conf = slot_conf_j[own_cand_idx].masked_fill(~own_valid, float("inf"))
            min_conf, _ = cand_conf.min(dim=1)
            best_conf = slot_conf_j[cand_idx[
                torch.arange(num_new, device=device), best_local
            ]].float()

        paper_radius = float(os.environ.get("POINT3R_RAY_SPATIAL_RADIUS", "0.05"))
        paper_neighbor = has_existing & (best_pos_dist < paper_radius)
        best_ang_dist = 1.0 - best_ray_sim
        redundant_ang = float(os.environ.get("POINT3R_RAY_REDUNDANT_ANG", "0.02"))
        loop_ang = float(os.environ.get("POINT3R_RAY_LOOP_ANG", "0.10"))
        loop_delta_t = int(os.environ.get("POINT3R_RAY_LOOP_DELTA_T", "15"))
        best_old_time = torch.full_like(new_time_j, -1)
        valid_best = best_target >= 0
        if valid_best.any():
            best_old_time[valid_best] = memory_time_j[best_target[valid_best]]
        paper_redundant = paper_neighbor & (best_ang_dist <= redundant_ang)
        loop_feat_thresh = float(os.environ.get(
            "POINT3R_RAY_LOOP_FEAT_SIM", "0.60"
        ))
        paper_loop = (
            paper_neighbor
            & (best_ang_dist >= loop_ang)
            & ((new_time_j - best_old_time).abs() > loop_delta_t)
            & (best_sim >= loop_feat_thresh)
        )
        if (
            paper_update
            and memory_feat_j.shape[0] > 0
            and cand_idx.numel() > 0
        ):
            candidate_old_time = memory_time_j[cand_idx]
            pointer_pair_mask = (
                valid_slots
                & (pos_dist < paper_radius)
                & ((1.0 - ray_sims) >= loop_ang)
                & (
                    (
                        new_time_j.unsqueeze(1)
                        - candidate_old_time
                    ).abs() > loop_delta_t
                )
                & (sims >= loop_feat_thresh)
            )
        paper_novel = ~paper_redundant & ~paper_loop
        appearance_similar = has_existing & (best_sim >= float(self.confselect_merge_threshold))
        ray_merge_distance = float(os.environ.get("POINT3R_RAY_MERGE_DISTANCE", "0.05"))
        ray_compatible = (1.0 - best_ray_sim) <= ray_merge_distance
        adaptive_average = rayaware and os.environ.get(
            "POINT3R_RAY_ADAPTIVE_AVERAGE", "0"
        ).lower() in ("1", "true", "yes", "on")
        # In adaptive mode ray disagreement changes the update strength instead
        # of making the merge decision discontinuous at a fixed threshold.
        similar_mask = appearance_similar & (ray_compatible | ~rayaware | adaptive_average)
        conf_margin = float(self.confselect_conf_margin)
        merge_mask = similar_mask & (new_conf_j >= best_conf + conf_margin)
        no_average = rayaware and os.environ.get("POINT3R_RAY_RETAIN_REPLACE", "1").lower() in ("1", "true", "yes", "on")
        if no_average or paper_update or ray_kway_update:
            merge_mask = torch.zeros_like(merge_mask)
        if paper_update or ray_kway_update:
            # No spatial neighbour means novel geometry and must grow memory.
            # Loop detection consumes the triage result before the stochastic
            # update below, matching Sec. 3.4/3.5 ordering in RayAway.
            if ray_kway_update:
                # K-way adaptation: a spatial neighbour seen from a genuinely
                # different ray is useful evidence, not a redundant pointer.
                distinct_candidate_mask = has_existing & ~paper_redundant
            else:
                distinct_candidate_mask = has_existing & ~paper_neighbor
            empty_candidate_mask = ~has_existing
        else:
            distinct_candidate_mask = (
                has_existing
                & (
                    (best_sim < float(self.confselect_merge_threshold))
                    | ((~ray_compatible) & (not adaptive_average))
                    | no_average
                )
                & (new_conf_j >= min_conf + conf_margin)
            )
            empty_candidate_mask = ~has_existing
        free_candidate_mask = (
            (empty_candidate_mask | distinct_candidate_mask)
            & (old_counts_for_token < num_slots)
        )

        append_mask = torch.zeros((num_new,), dtype=torch.bool, device=device)
        free_indices = torch.nonzero(free_candidate_mask, as_tuple=False).flatten()
        if free_indices.numel() > 0:
            free_keys = img_keys[free_indices]
            free_sorted_keys, free_sort_idx = torch.sort(free_keys)
            free_unique_keys, free_counts = torch.unique_consecutive(free_sorted_keys, return_counts=True)
            free_starts = torch.cat([free_counts.new_zeros(1), free_counts.cumsum(0)[:-1]])
            free_rank_sorted = torch.arange(free_indices.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(free_starts, free_counts)
            free_rank = torch.empty_like(free_rank_sorted)
            free_rank[free_sort_idx] = free_rank_sorted
            free_old_counts = slot_count_j[free_keys].clamp_max(num_slots)
            free_space = (num_slots - free_old_counts).clamp_min(0)
            keep_free = free_rank < free_space
            append_mask[free_indices[keep_free]] = True

        memory_feat_all = memory_feat_j
        memory_pos_all = memory_pos_j
        slot_conf_all = slot_conf_j
        memory_ray_all = memory_ray_j
        memory_time_all = memory_time_j
        memory_local_all = memory_local_j

        append_indices = torch.nonzero(append_mask, as_tuple=False).flatten()
        if append_indices.numel() > 0:
            start_mem = int(memory_pos_j.shape[0])
            appended_global = torch.arange(
                start_mem,
                start_mem + int(append_indices.numel()),
                dtype=torch.long,
                device=device,
            )
            append_keys = img_keys[append_indices]
            append_old_counts = slot_count_j[append_keys].clamp_max(num_slots)
            append_sorted_keys, append_sort_idx = torch.sort(append_keys)
            append_unique_keys, append_counts = torch.unique_consecutive(append_sorted_keys, return_counts=True)
            append_starts = torch.cat([append_counts.new_zeros(1), append_counts.cumsum(0)[:-1]])
            append_rank_sorted = torch.arange(append_indices.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(append_starts, append_counts)
            append_rank = torch.empty_like(append_rank_sorted)
            append_rank[append_sort_idx] = append_rank_sorted
            append_slots = append_old_counts + append_rank
            slot_table_j[append_keys, append_slots] = appended_global
            slot_count_j[append_unique_keys] = torch.minimum(
                slot_count_j[append_unique_keys] + append_counts,
                append_counts.new_full((), num_slots),
            )
            memory_feat_all = torch.cat([memory_feat_all, new_feat_j[append_indices]], dim=0)
            memory_pos_all = torch.cat([memory_pos_all, img_pos_j[append_indices]], dim=0)
            slot_conf_all = torch.cat([slot_conf_all, new_conf_j[append_indices].float()], dim=0)
            memory_ray_all = torch.cat([memory_ray_all, new_ray_j[append_indices]], dim=0)
            memory_time_all = torch.cat([memory_time_all, new_time_j[append_indices]], dim=0)
            memory_local_all = torch.cat([memory_local_all, new_local_j[append_indices]], dim=0)

        merge_indices = torch.nonzero(merge_mask, as_tuple=False).flatten()
        if merge_indices.numel() > 0:
            merge_cand_idx = slot_table_j[img_keys[merge_indices]].clamp_min(0)
            merge_target = merge_cand_idx[
                torch.arange(merge_cand_idx.shape[0], device=device),
                best_local[merge_indices],
            ].long()
            rem_feat = new_feat_j[merge_indices]
            rem_pos = img_pos_j[merge_indices]
            rem_conf = new_conf_j[merge_indices].to(dtype=rem_feat.dtype).clamp_min(1e-6)

            unique_targets, inverse = torch.unique(merge_target, return_inverse=True)
            feat_sum = torch.zeros((unique_targets.numel(), rem_feat.shape[-1]), dtype=rem_feat.dtype, device=device)
            pos_sum = torch.zeros((unique_targets.numel(), rem_pos.shape[-1]), dtype=rem_pos.dtype, device=device)
            conf_sum = torch.zeros((unique_targets.numel(), 1), dtype=rem_feat.dtype, device=device)
            feat_sum.index_add_(0, inverse, rem_feat * rem_conf.unsqueeze(-1))
            pos_sum.index_add_(0, inverse, rem_pos * rem_conf.unsqueeze(-1))
            conf_sum.index_add_(0, inverse, rem_conf.unsqueeze(-1))
            feat_mean = feat_sum / conf_sum.clamp_min(1e-6)
            pos_mean = pos_sum / conf_sum.to(pos_sum.dtype).clamp_min(1e-6)
            conf_mean = (conf_sum.squeeze(-1) / torch.zeros_like(conf_sum.squeeze(-1)).index_add_(0, inverse, torch.ones_like(rem_conf)).clamp_min(1)).float()
            old_conf = slot_conf_all[unique_targets].float().clamp_min(1e-6)
            ray_sum = torch.zeros((unique_targets.numel(), 3), dtype=memory_ray_all.dtype, device=device)
            ray_sum.index_add_(0, inverse, new_ray_j[merge_indices].to(memory_ray_all.dtype) * rem_conf.unsqueeze(-1).to(memory_ray_all.dtype))
            ray_mean = F.normalize(ray_sum / conf_sum.to(ray_sum.dtype).clamp_min(1e-6), dim=-1)
            beta_base = (conf_mean / (old_conf + conf_mean + 1e-6)).to(dtype=memory_feat_all.dtype).unsqueeze(-1)
            beta = beta_base
            if adaptive_average:
                old_ray = F.normalize(memory_ray_all[unique_targets].float(), dim=-1)
                new_ray = F.normalize(ray_mean.float(), dim=-1)
                ray_cos = (old_ray * new_ray).sum(dim=-1).clamp(-1.0, 1.0)
                ray_angle = torch.acos(ray_cos)
                sigma_deg = float(os.environ.get("POINT3R_RAY_AVG_SIGMA_DEG", "20.0"))
                sigma_rad = max(sigma_deg * torch.pi / 180.0, 1e-4)
                ray_floor = float(os.environ.get("POINT3R_RAY_AVG_FLOOR", "0.05"))
                ray_gate = torch.exp(-0.5 * (ray_angle / sigma_rad) ** 2)
                ray_gate = ray_floor + (1.0 - ray_floor) * ray_gate
                if ray_adaptive_strength is not None:
                    motion_strength = torch.as_tensor(
                        ray_adaptive_strength, device=ray_gate.device, dtype=ray_gate.dtype
                    ).clamp(0.0, 1.0)
                    # Fast camera motion should approach the original averaging
                    # rule; nearly stationary views receive the full ray gate.
                    ray_gate = 1.0 - motion_strength * (1.0 - ray_gate)
                beta = beta * ray_gate.to(dtype=beta.dtype).unsqueeze(-1)
            memory_feat_all[unique_targets] = (1.0 - beta) * memory_feat_all[unique_targets] + beta * feat_mean
            feature_only_gate = adaptive_average and os.environ.get(
                "POINT3R_RAY_FEATURE_ONLY_GATE", "0"
            ).lower() in ("1", "true", "yes", "on")
            position_mix = float(os.environ.get(
                "POINT3R_RAY_POSITION_GATE_MIX", "1.0" if feature_only_gate else "0.0"
            ))
            position_mix = max(0.0, min(1.0, position_mix))
            beta_pos_source = (1.0 - position_mix) * beta + position_mix * beta_base
            beta_pos = beta_pos_source.to(dtype=memory_pos_all.dtype)
            memory_pos_all[unique_targets] = (1.0 - beta_pos) * memory_pos_all[unique_targets] + beta_pos * pos_mean
            memory_ray_all[unique_targets] = F.normalize((1.0 - beta_pos) * memory_ray_all[unique_targets] + beta_pos * ray_mean, dim=-1)
            # Metadata follows the newest representative observation.
            latest_order = torch.zeros((unique_targets.numel(),), dtype=torch.long, device=device)
            latest_order.scatter_reduce_(0, inverse, new_time_j[merge_indices], reduce="amax", include_self=True)
            memory_time_all[unique_targets] = latest_order
            local_sum = torch.zeros((unique_targets.numel(), 3), dtype=memory_local_all.dtype, device=device)
            local_sum.index_add_(0, inverse, new_local_j[merge_indices].to(memory_local_all.dtype) * rem_conf.unsqueeze(-1).to(memory_local_all.dtype))
            memory_local_all[unique_targets] = local_sum / conf_sum.to(local_sum.dtype).clamp_min(1e-6)
            slot_conf_all[unique_targets] = ((1.0 - beta.squeeze(-1).float()) * old_conf + beta.squeeze(-1).float() * conf_mean).float()

        # Full buckets need a bounded replacement path; otherwise decay only
        # lowers the threshold but cannot admit distinct observations.
        if paper_update:
            # Seeded stochastic retain-or-replace: P(replace)=0.5.  Resolve
            # collisions so at most one incoming observation replaces a given
            # matched pointer in this vectorized update.
            paper_coin = torch.rand((num_new,), device=device)
            same_bucket = torch.zeros((num_new,), dtype=torch.bool, device=device)
            if valid_best.any():
                matched_keys = self._ordered_pack_bins(
                    self._ordered_spatial_bins(memory_pos_j[best_target[valid_best]])
                ).long()
                same_bucket[valid_best] = matched_keys == img_keys[valid_best]
            # A neighbour may lie across a quantization boundary.  The paper's
            # radius search is continuous, so rejecting every cross-bucket
            # replacement introduces an unintended retain bias.  Cross-bucket
            # winners are rehashed below when their destination has capacity.
            paper_raw = paper_neighbor & (paper_coin < 0.5)
            paper_indices = torch.nonzero(paper_raw, as_tuple=False).flatten()
            paper_accept = torch.zeros_like(paper_raw)
            if paper_indices.numel() > 0:
                paper_targets = best_target[paper_indices].long()
                min_coin = torch.full(
                    (memory_feat_all.shape[0],), float("inf"), dtype=paper_coin.dtype, device=device
                )
                min_coin.scatter_reduce_(0, paper_targets, paper_coin[paper_indices], reduce="amin", include_self=True)
                keep_one = paper_coin[paper_indices] <= min_coin[paper_targets]
                paper_accept[paper_indices[keep_one]] = True
            replace_candidate_mask = paper_accept
        elif ray_kway_update:
            # Retain same-view redundant pointers.  A view-novel observation
            # may enter a full bucket by evicting one geometrically redundant
            # ray slot below.
            replace_candidate_mask = (
                has_existing
                & ~paper_redundant
                & (old_counts_for_token >= num_slots)
                & ~append_mask
            )
        else:
            replace_candidate_mask = (
                has_existing
                & (best_sim < float(self.confselect_merge_threshold))
                & (old_counts_for_token >= num_slots)
                & (new_conf_j >= min_conf + conf_margin)
                & ~merge_mask
            )
        replace_indices = torch.nonzero(replace_candidate_mask, as_tuple=False).flatten()
        replace_count = 0
        replace_accept_mask = torch.zeros((num_new,), dtype=torch.bool, device=device)
        if replace_indices.numel() > 0:
            if not paper_update:
                replace_keys = img_keys[replace_indices]
                replace_sorted_keys, replace_sort_idx = torch.sort(replace_keys)
                replace_unique_keys, replace_counts = torch.unique_consecutive(replace_sorted_keys, return_counts=True)
                replace_starts = torch.cat([replace_counts.new_zeros(1), replace_counts.cumsum(0)[:-1]])
                replace_rank_sorted = torch.arange(replace_indices.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(replace_starts, replace_counts)
                replace_rank = torch.empty_like(replace_rank_sorted)
                replace_rank[replace_sort_idx] = replace_rank_sorted
                replace_indices = replace_indices[replace_rank == 0]
            replace_accept_mask[replace_indices] = True
            if paper_update:
                replace_targets = best_target[replace_indices]
                source_keys = self._ordered_pack_bins(
                    self._ordered_spatial_bins(
                        memory_pos_all[replace_targets]
                    )
                ).long()
                destination_keys = img_keys[replace_indices]
                cross_bucket = source_keys != destination_keys
                if cross_bucket.any():
                    cross_ids = torch.nonzero(
                        cross_bucket, as_tuple=False
                    ).flatten()
                    cross_dest = destination_keys[cross_ids]
                    # Respect destination capacity.  This is conservative when
                    # another accepted move simultaneously frees that bucket,
                    # but never creates an unindexed memory pointer.
                    cross_sorted, cross_order = torch.sort(cross_dest)
                    cross_unique, cross_counts = torch.unique_consecutive(
                        cross_sorted, return_counts=True
                    )
                    cross_starts = torch.cat(
                        [
                            cross_counts.new_zeros(1),
                            cross_counts.cumsum(0)[:-1],
                        ]
                    )
                    cross_rank_sorted = (
                        torch.arange(
                            cross_ids.numel(), device=device, dtype=torch.long
                        )
                        - torch.repeat_interleave(cross_starts, cross_counts)
                    )
                    cross_rank = torch.empty_like(cross_rank_sorted)
                    cross_rank[cross_order] = cross_rank_sorted
                    cross_space = (
                        num_slots
                        - slot_count_j[cross_dest].clamp_max(num_slots)
                    ).clamp_min(0)
                    cross_keep = cross_rank < cross_space
                    keep_replace = ~cross_bucket
                    keep_replace[cross_ids[cross_keep]] = True
                    replace_accept_mask[replace_indices] = False
                    replace_indices = replace_indices[keep_replace]
                    replace_targets = replace_targets[keep_replace]
                    source_keys = source_keys[keep_replace]
                    destination_keys = destination_keys[keep_replace]
                    cross_bucket = source_keys != destination_keys
                    replace_accept_mask[replace_indices] = True

                # Move cross-bucket targets in the slot table before updating
                # their positions.  Compact vacated source rows, then append
                # the same target indices to their destination rows.
                if cross_bucket.any():
                    moved_targets = replace_targets[cross_bucket]
                    moved_source = source_keys[cross_bucket]
                    moved_dest = destination_keys[cross_bucket]
                    source_rows = slot_table_j[moved_source]
                    source_slots = (
                        source_rows == moved_targets.unsqueeze(1)
                    ).long().argmax(dim=1)
                    slot_table_j[moved_source, source_slots] = -1
                    touched_source = torch.unique(moved_source)
                    rows = slot_table_j[touched_source]
                    compact_order = torch.argsort(
                        (rows < 0).long(), dim=1, stable=True
                    )
                    slot_table_j[touched_source] = rows.gather(
                        1, compact_order
                    )
                    slot_count_j[touched_source] = (
                        slot_table_j[touched_source] >= 0
                    ).sum(dim=1)

                    moved_sorted, moved_order = torch.sort(moved_dest)
                    moved_unique, moved_counts = torch.unique_consecutive(
                        moved_sorted, return_counts=True
                    )
                    moved_starts = torch.cat(
                        [
                            moved_counts.new_zeros(1),
                            moved_counts.cumsum(0)[:-1],
                        ]
                    )
                    moved_rank_sorted = (
                        torch.arange(
                            moved_targets.numel(),
                            device=device,
                            dtype=torch.long,
                        )
                        - torch.repeat_interleave(moved_starts, moved_counts)
                    )
                    moved_rank = torch.empty_like(moved_rank_sorted)
                    moved_rank[moved_order] = moved_rank_sorted
                    destination_slots = (
                        slot_count_j[moved_dest] + moved_rank
                    )
                    slot_table_j[
                        moved_dest, destination_slots
                    ] = moved_targets
                    slot_count_j[moved_unique] = torch.minimum(
                        slot_count_j[moved_unique] + moved_counts,
                        moved_counts.new_full((), num_slots),
                    )
            else:
                replace_keys = img_keys[replace_indices]
                replace_cand_idx = slot_table_j[replace_keys].clamp_min(0)
                if ray_kway_update:
                    # Evict a slot from the most mutually redundant ray pair.
                    # This preserves angular coverage instead of randomly
                    # destroying the bucket's existing view diversity.
                    replace_rays = F.normalize(
                        memory_ray_all[replace_cand_idx].float(), dim=-1
                    )
                    pair_sim = torch.einsum(
                        "nkd,njd->nkj", replace_rays, replace_rays
                    )
                    diagonal = torch.eye(
                        num_slots, dtype=torch.bool, device=device
                    ).unsqueeze(0)
                    pair_sim = pair_sim.masked_fill(diagonal, -float("inf"))
                    ray_redundancy = pair_sim.max(dim=-1).values
                    replace_local = ray_redundancy.argmax(dim=1)
                else:
                    replace_conf = slot_conf_all[replace_cand_idx]
                    replace_local = replace_conf.argmin(dim=1)
                replace_targets = replace_cand_idx[torch.arange(replace_indices.shape[0], device=device), replace_local]
            if ray_kway_update and replace_indices.numel() > 0:
                min_conf_ratio = max(
                    0.0, float(os.environ.get("POINT3R_RAY_KWAY_MIN_CONF_RATIO", "0.0"))
                )
                if min_conf_ratio > 0.0:
                    quality_ok = new_conf_j[replace_indices] >= (
                        min_conf_ratio * slot_conf_all[replace_targets]
                    )
                    replace_accept_mask[replace_indices] = False
                    replace_indices = replace_indices[quality_ok]
                    replace_targets = replace_targets[quality_ok]
                    replace_accept_mask[replace_indices] = True
            memory_feat_all[replace_targets] = new_feat_j[replace_indices]
            memory_pos_all[replace_targets] = img_pos_j[replace_indices]
            memory_ray_all[replace_targets] = new_ray_j[replace_indices]
            memory_time_all[replace_targets] = new_time_j[replace_indices]
            memory_local_all[replace_targets] = new_local_j[replace_indices]
            slot_conf_all[replace_targets] = new_conf_j[replace_indices]
            replace_count = int(replace_indices.numel())

        # Keep the formal FPS path free of per-frame GPU-host synchronizations.
        # Detailed counters are opt-in through POINT3R_CONFSELECT_STATS=1.
        stats["miss"] = -1
        stats["hit_free"] = -1
        stats["hit_full"] = -1
        stats["replace_low_conf"] = 0
        stats["candidate_sum"] = -1
        stats["rows"] = -1
        stats["final_memory"] = int(memory_pos_all.shape[0])
        stats["candidate_avg"] = -1.0
        stats["neighbor_rows_avg"] = -1.0
        if self.confselect_collect_stats:
            merge_count = int(merge_mask.sum().detach().cpu())
            stats["merge_sim"] = merge_count
            stats["append_distinct"] = int(append_mask.sum().detach().cpu())
            stats["empty_store"] = int((append_mask & empty_candidate_mask).sum().detach().cpu())
            stats["distinct_store"] = int((append_mask & distinct_candidate_mask).sum().detach().cpu())
            stats["merge_better_conf"] = merge_count
            stats["replace_low_conf"] = replace_count
            accepted_mask = append_mask | merge_mask | replace_accept_mask
            stats["policy_reject"] = int((~accepted_mask).sum().detach().cpu())
            stats["app_reject"] = stats["policy_reject"]
            stats["slot_conf_mean"] = float(slot_conf_all.mean().detach().cpu()) if slot_conf_all.numel() > 0 else 0.0
            stats["incoming_conf_mean"] = float(new_conf_j.mean().detach().cpu()) if new_conf_j.numel() > 0 else 0.0
        else:
            stats["merge_sim"] = -1
            stats["append_distinct"] = -1
            stats["empty_store"] = -1
            stats["distinct_store"] = -1
            stats["merge_better_conf"] = -1
            stats["policy_reject"] = -1
            stats["app_reject"] = -1
        stats["rayaware_update"] = int(rayaware)
        stats["retain_replace"] = int(no_average)
        stats["ray_adaptive_average"] = int(adaptive_average)
        stats["ray_paper_update"] = int(paper_update)
        stats["ray_kway_diverse_update"] = int(ray_kway_update)
        ray_reasoner_active = paper_update or ray_kway_update
        stats["ray_redundant"] = int(paper_redundant.sum().item()) if ray_reasoner_active else 0
        stats["ray_loop"] = int(paper_loop.sum().item()) if ray_reasoner_active else 0
        stats["ray_novel"] = int(paper_novel.sum().item()) if ray_reasoner_active else 0
        pointer_loop_graph = os.environ.get(
            "POINT3R_RAY_POINTER_LOOP_GRAPH", "0"
        ).lower() in ("1", "true", "yes", "on")
        if pointer_loop_graph:
            frame_value = int(new_time_j[0].item()) if new_time_j.numel() else -1
            loop_count = int(paper_loop.sum().item())
            if loop_count > 0 or frame_value % 10 == 0:
                self._lc_emit(
                    f"[RAY_POINTER_TRIAGE] frame={frame_value} "
                    f"redundant={int(paper_redundant.sum().item())} "
                    f"loop={loop_count} novel={int(paper_novel.sum().item())} "
                    f"radius={paper_radius:.6f} feat_thresh={loop_feat_thresh:.4f}"
                )
        if (
            pointer_loop_graph
            and paper_update
            and pointer_pair_mask is not None
            and pointer_pair_mask.any()
        ):
            pair_new, pair_slot = torch.nonzero(
                pointer_pair_mask, as_tuple=True
            )
            if pair_new.numel() > 0:
                loop_targets = cand_idx[pair_new, pair_slot].long()
                stats["ray_loop_pairs"] = int(pair_new.numel())
                # Private payload: the dual-bank caller consumes and removes
                # it before stats are persisted, so tensor references cannot
                # leak into the experiment summaries.
                stats["_pointer_loop_payload"] = {
                    "new_local": new_local_j[pair_new].detach(),
                    "old_local": memory_local_j[loop_targets].detach(),
                    "old_world": memory_pos_j[loop_targets].detach(),
                    "old_frame": memory_time_j[loop_targets].detach(),
                    "old_target": loop_targets.detach(),
                    "new_target": pair_new.detach(),
                    "feat_sim": sims[pair_new, pair_slot].detach(),
                    "pos_dist": pos_dist[pair_new, pair_slot].detach(),
                    "ang_dist": (1.0 - ray_sims[pair_new, pair_slot]).detach(),
                }
        return memory_feat_all, memory_pos_all, slot_table_j, slot_count_j, slot_conf_all, memory_ray_all, memory_time_all, memory_local_all, stats

    # ---------------------------------------------------------------

    @staticmethod
    def _lc_emit(message):
        print(message, flush=True)
        audit_path = os.environ.get("POINT3R_LC_AUDIT_FILE", "").strip()
        if audit_path:
            try:
                with open(audit_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            except Exception:
                pass

    def _lc_pointer_loop_edges_from_payload(
        self, frame_i, payload, device
    ):
        """Turn RayAware long-time revisit pointers into verified SE(3) edges.

        The pointer update already performed bounded spatial-hash lookup plus
        ray and feature triage.  This function only does the missing geometric
        verification/graph handoff; it never creates adjacent odometry edges.
        """
        enabled = os.environ.get(
            "POINT3R_RAY_POINTER_LOOP_GRAPH", "0"
        ).lower() in ("1", "true", "yes", "on")
        if not enabled or payload is None or frame_i <= 0:
            return []
        required = (
            "new_local", "old_local", "old_world", "old_frame", "old_target",
            "new_target",
            "feat_sim", "pos_dist", "ang_dist",
        )
        if any(name not in payload for name in required):
            return []
        old_frame = payload["old_frame"].long()
        valid = (old_frame >= 0) & (old_frame < frame_i)
        if not valid.any():
            return []
        current_pose = (
            self._pose_trajectory[frame_i]
            if frame_i < len(self._pose_trajectory) else None
        )
        if current_pose is None:
            return []

        min_pairs = max(6, int(os.environ.get(
            "POINT3R_RAY_LOOP_MIN_PAIRS", "12"
        )))
        max_corr = max(min_pairs, int(os.environ.get(
            "POINT3R_RAY_LOOP_MAX_CORRESPONDENCES", "256"
        )))
        max_edges = max(1, int(os.environ.get(
            "POINT3R_RAY_LOOP_MAX_EDGES_PER_FRAME", "1"
        )))
        radius = max(1e-6, float(os.environ.get(
            "POINT3R_RAY_SPATIAL_RADIUS", "0.05"
        )))
        candidates = []
        time_window = max(1, int(os.environ.get(
            "POINT3R_RAY_LOOP_TIME_WINDOW", "10"
        )))
        group_id = torch.div(old_frame, time_window, rounding_mode="floor")
        valid_groups = torch.unique(group_id[valid])
        group_sizes = [
            int((valid & (group_id == value)).sum().item())
            for value in valid_groups
        ]
        self._lc_emit(
            f"[RAY_POINTER_GROUP] frame={frame_i} matches={int(valid.sum().item())} "
            f"groups={len(group_sizes)} max_group={max(group_sizes, default=0)} "
            f"window={time_window}"
        )
        for group_tensor in valid_groups:
            ids = torch.nonzero(
                valid & (group_id == group_tensor), as_tuple=False
            ).flatten()
            if ids.numel() < min_pairs:
                continue
            group_frames = torch.sort(torch.unique(old_frame[ids])).values
            target_frame = int(
                group_frames[group_frames.numel() // 2].item()
            )
            target_pose = (
                self._pose_trajectory[target_frame]
                if target_frame < len(self._pose_trajectory) else None
            )
            if target_pose is None:
                continue

            # One incoming token per stored pointer, chosen by the same joint
            # evidence used for RayAware triage.
            cost = (
                payload["pos_dist"][ids].float() / radius
                + 0.25 * (1.0 - payload["feat_sim"][ids].float())
                + 0.10 * payload["ang_dist"][ids].float()
            )
            order = ids[torch.argsort(cost)]
            used_targets = set()
            used_new = set()
            keep = []
            for idx in order.tolist():
                target_id = int(payload["old_target"][idx].item())
                new_id = int(payload["new_target"][idx].item())
                if target_id in used_targets or new_id in used_new:
                    continue
                used_targets.add(target_id)
                used_new.add(new_id)
                keep.append(idx)
                if len(keep) >= max_corr:
                    break
            if len(keep) < min_pairs:
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=unique proposals={ids.numel()} unique={len(keep)}"
                )
                continue
            keep = torch.tensor(keep, device=device, dtype=torch.long)
            src = payload["new_local"][keep].float()
            target_pose_device = target_pose.to(device).float()
            old_world = payload["old_world"][keep].float()
            dst = (
                target_pose_device[:3, :3].T
                @ (
                    old_world
                    - target_pose_device[:3, 3].unsqueeze(0)
                ).T
            ).T
            estimate = self._lc_ransac_se3(
                src,
                dst,
                float(os.environ.get(
                    "POINT3R_RAY_LOOP_RANSAC_THRESH", "0.10"
                )),
                int(os.environ.get(
                    "POINT3R_RAY_LOOP_RANSAC_ITERS", "96"
                )),
                seed=(frame_i + 1) * 100003 + target_frame,
            )
            if estimate is None:
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=ransac_none pairs={src.shape[0]}"
                )
                continue
            transform, inliers, rmse = estimate
            inlier_count = int(inliers.sum().item())
            ratio = inlier_count / max(1, int(src.shape[0]))
            if (
                inlier_count < min_pairs
                or ratio < float(os.environ.get(
                    "POINT3R_RAY_LOOP_MIN_INLIER_RATIO", "0.60"
                ))
            ):
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=inlier pairs={src.shape[0]} inliers={inlier_count} "
                    f"ratio={ratio:.4f} rmse={float(rmse):.6f}"
                )
                continue

            predicted = (
                torch.linalg.inv(target_pose.to(device).float())
                @ current_pose.to(device).float()
            )
            delta = transform.float() @ torch.linalg.inv(predicted)
            cosine = ((torch.trace(delta[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
            correction_deg = float(torch.rad2deg(torch.acos(cosine)).item())
            correction_trans = float(delta[:3, 3].norm().item())
            max_rotation = float(os.environ.get(
                "POINT3R_RAY_LOOP_MAX_ROT_DEG", "3.0"
            ))
            max_translation = float(os.environ.get(
                "POINT3R_RAY_LOOP_MAX_TRANS", "0.20"
            ))
            rotation_fallback = os.environ.get(
                "POINT3R_RAY_LOOP_ROTATION_FALLBACK", "0"
            ).lower() in ("1", "true", "yes", "on")
            rotation_only = False
            if correction_deg > max_rotation:
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=correction corr_deg={correction_deg:.4f} "
                    f"corr_trans={correction_trans:.6f}"
                )
                continue
            if correction_trans > max_translation:
                if rotation_fallback:
                    rotation_only = True
                    transform = transform.clone()
                    transform[:3, 3] = predicted[:3, 3]
                else:
                    self._lc_emit(
                        f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                        f"stage=correction corr_deg={correction_deg:.4f} "
                        f"corr_trans={correction_trans:.6f}"
                    )
                    continue

            src_in = src[inliers]
            dst_in = dst[inliers]
            src_centered = src_in - src_in.mean(dim=0)
            dst_centered = dst_in - dst_in.mean(dim=0)
            src_spread = src_centered.norm(dim=-1).median().clamp_min(1e-5)
            dst_spread = dst_centered.norm(dim=-1).median().clamp_min(1e-5)
            scale_ratio = float((dst_spread / src_spread).item())
            covariance = src_centered.T @ src_centered / max(1, src_in.shape[0])
            eigvals = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            observability = float(
                (eigvals[0] / eigvals[-1].clamp_min(1e-6)).item()
            )
            if (not rotation_only) and not (
                float(os.environ.get(
                    "POINT3R_RAY_LOOP_SCALE_MIN", "0.80"
                )) <= scale_ratio <= float(os.environ.get(
                    "POINT3R_RAY_LOOP_SCALE_MAX", "1.25"
                ))
            ):
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=scale ratio={scale_ratio:.4f}"
                )
                continue
            if (not rotation_only) and observability < float(os.environ.get(
                "POINT3R_RAY_LOOP_MIN_OBSERVABILITY", "0.002"
            )):
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=observability value={observability:.6f}"
                )
                continue

            inlier_rays = F.normalize(src_in.float(), dim=-1)
            ray_information = (
                torch.eye(3, device=device).unsqueeze(0)
                - torch.einsum("ni,nj->nij", inlier_rays, inlier_rays)
            ).mean(dim=0)
            ray_coverage = float(
                torch.linalg.eigvalsh(ray_information)[0].clamp_min(0.0).item()
            )
            common_quality = (
                ratio * ratio
                * math.exp(-((float(rmse) / 0.05) ** 2))
                * math.exp(-((correction_deg / 3.0) ** 2))
            )
            if rotation_only:
                # Absolute inlier count is already hard-gated above.  For an
                # SO(3)-only edge the relevant degeneracy is angular ray
                # coverage, not volumetric 3-D support of a planar surface.
                quality = common_quality * min(1.0, ray_coverage / 0.02)
            else:
                quality = (
                    common_quality
                    * min(1.0, inlier_count / 50.0)
                    * min(1.0, observability / 0.02)
                    * math.exp(-((correction_trans / 0.20) ** 2))
                )
            if quality < float(os.environ.get(
                "POINT3R_RAY_LOOP_MIN_QUALITY", "0.02"
            )):
                self._lc_emit(
                    f"[RAY_POINTER_REJECT] frame={frame_i} target={target_frame} "
                    f"stage=quality value={quality:.6f} rmse={float(rmse):.6f} "
                    f"corr_deg={correction_deg:.4f} corr_trans={correction_trans:.6f}"
                )
                continue
            info = self._lc_fisher_information(
                src_in, transform, float(rmse), ratio
            ) * quality
            if rotation_only:
                trans_scale = float(os.environ.get(
                    "POINT3R_RAY_LOOP_TRANS_INFO_SCALE", "0.01"
                ))
                max_rot_weight = float(os.environ.get(
                    "POINT3R_RAY_LOOP_MAX_ROT_WEIGHT", "5.0"
                ))
                eigval, eigvec = torch.linalg.eigh(ray_information)
                eigval = eigval.clamp_min(eigval.max().clamp_min(1e-6) * 0.05)
                rot_info = (eigvec * eigval.unsqueeze(0)) @ eigvec.T
                rot_info = rot_info * (
                    3.0 * max_rot_weight * quality
                    / rot_info.trace().clamp_min(1e-6)
                )
                info[:3, :3] = rot_info
                info[:3, 3:] = 0.0
                info[3:, :3] = 0.0
                info[3:, 3:] *= trans_scale
            temporal_bonus = 1.0 + min(frame_i - target_frame, 200) / 200.0
            score = quality * inlier_count * temporal_bonus
            candidates.append((
                score,
                (
                    frame_i,
                    target_frame,
                    transform.detach().cpu(),
                    info.detach().cpu(),
                ),
            ))
            self._lc_emit(
                f"[RAY_POINTER_LOOP] frame={frame_i} target={target_frame} "
                f"lag={frame_i-target_frame} pairs={src.shape[0]} "
                f"inliers={inlier_count} ratio={ratio:.4f} rmse={float(rmse):.6f} "
                f"corr_deg={correction_deg:.4f} corr_trans={correction_trans:.6f} "
                f"scale={scale_ratio:.4f} observability={observability:.6f} "
                f"ray_coverage={ray_coverage:.6f} quality={quality:.6f} "
                f"mode={'rotation' if rotation_only else 'se3'}"
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in candidates[:max_edges]]
    # Loop Closure v3 — 修复版
    # 修复项：
    #   1. loop edge 用 3D correspondence + Umeyama 独立估计 T_ij，不再用预测 pose
    #   2. PGO 结果写回 ress[i]["camera_pose"]（通过 _lc_pose_corrections 传出）
    #   3. local_xyz 用正确的 R^T @ (x_w - t)
    #   4. loop index 有容量上限（POINT3R_LC_MAX_INDEX_PER_FRAME 控制）
    #   5. 默认关闭（POINT3R_LC_ENABLED 默认为 0）
    # ---------------------------------------------------------------

    def _lc_update_index(self, j, write_pos_j, write_local_j, write_ray_j, write_feat_j, frame_i, c2w, device):
        """把 q25 筛选后的 token 加入 loop index（上限控制，正确局部坐标）"""
        index_stride = max(1, int(os.environ.get("POINT3R_LC_INDEX_STRIDE", "1")))
        if frame_i % index_stride != 0:
            return
        max_per_frame = int(os.environ.get("POINT3R_LC_MAX_INDEX_PER_FRAME", "256"))
        n = write_pos_j.shape[0]
        if n > max_per_frame:
            # 均匀采样，控制内存
            idx = torch.linspace(0, n - 1, max_per_frame, device=device).long()
            write_pos_j = write_pos_j[idx]
            write_local_j = write_local_j[idx]
            write_ray_j = write_ray_j[idx]
            write_feat_j = write_feat_j[idx]
            n = max_per_frame

        if write_local_j is None and c2w is not None:
            t = c2w[:3, 3].to(device)                       # [3]
            R = c2w[:3, :3].to(device)                      # [3,3]
            local_xyz = (R.T @ (write_pos_j - t.unsqueeze(0)).T).T  # R^T @ (x_w - t) [N,3]
        elif write_local_j is not None:
            local_xyz = write_local_j
        else:
            local_xyz = write_pos_j.clone()

        fids = torch.full((n,), frame_i, dtype=torch.long, device=device)

        if self._lc_pos is None:
            self._lc_pos = [None]
            self._lc_fid = [None]
            self._lc_local = [None]
            self._lc_ray = [None]
            self._lc_feat = [None]
        while len(self._lc_pos) <= j:
            self._lc_pos.append(None)
            self._lc_fid.append(None)
            self._lc_local.append(None)
            self._lc_ray.append(None)
            self._lc_feat.append(None)

        def _cat(old, new):
            return new if old is None else torch.cat([old, new], dim=0)

        adjacent_only = os.environ.get(
            "POINT3R_LC_ADJACENT_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if adjacent_only:
            self._lc_pos[j] = write_pos_j.detach()
            self._lc_fid[j] = fids
            self._lc_local[j] = local_xyz.detach()
            self._lc_ray[j] = write_ray_j.detach()
            self._lc_feat[j] = F.normalize(write_feat_j.detach().float(), dim=-1)
        else:
            self._lc_pos[j]   = _cat(self._lc_pos[j],   write_pos_j.detach())
            self._lc_fid[j]   = _cat(self._lc_fid[j],   fids)
            self._lc_local[j] = _cat(self._lc_local[j], local_xyz.detach())
            self._lc_ray[j]   = _cat(self._lc_ray[j],   write_ray_j.detach())
            self._lc_feat[j]  = _cat(self._lc_feat[j],  F.normalize(write_feat_j.detach().float(), dim=-1))

    def _lc_estimate_adjacent_rotation(self, j, write_pos_j, write_local_j, write_feat_j, frame_i, device, target_frame=None, force_rotation=False):
        """Estimate current->target relative rotation from shared pointers."""
        if frame_i <= 0 or self._lc_pos is None or j >= len(self._lc_pos) or self._lc_pos[j] is None:
            return None
        if target_frame is None:
            target_frame = frame_i - 1
        if target_frame < 0 or target_frame >= frame_i:
            return None
        old_mask = self._lc_fid[j] == target_frame
        if int(old_mask.sum().item()) < 6:
            return None
        old_pos = self._lc_pos[j][old_mask]
        old_local = self._lc_local[j][old_mask]
        old_feat = self._lc_feat[j][old_mask]
        new_feat = F.normalize(write_feat_j.float(), dim=-1)
        eps = float(os.environ.get("POINT3R_LC_ODOM_EPS_POS", "0.30"))
        feat_thresh = float(os.environ.get("POINT3R_LC_ODOM_FEAT_SIM", "0.55"))
        dist = torch.cdist(write_pos_j.float(), old_pos.float())
        feat_sim = new_feat @ old_feat.T
        valid = (dist < eps) & (feat_sim > feat_thresh)
        if int(valid.sum().item()) < 8:
            return None
        feat_weight = float(os.environ.get("POINT3R_LC_ODOM_FEAT_WEIGHT", "0.75"))
        cost = dist / max(eps, 1e-6) + feat_weight * (1.0 - feat_sim)
        cost = cost.masked_fill(~valid, float("inf"))
        new_to_old = cost.argmin(dim=1)
        old_to_new = cost.argmin(dim=0)
        new_ids = torch.arange(write_pos_j.shape[0], device=device)
        mutual = old_to_new[new_to_old] == new_ids
        cost_margin = max(0.0, float(os.environ.get("POINT3R_LC_ODOM_COST_MARGIN", "0.0")))
        if cost_margin > 0.0 and cost.shape[0] >= 2 and cost.shape[1] >= 2:
            new_top2 = torch.topk(cost, k=2, dim=1, largest=False).values
            old_top2 = torch.topk(cost, k=2, dim=0, largest=False).values
            new_unambiguous = (new_top2[:, 1] - new_top2[:, 0]) >= cost_margin
            old_unambiguous = (old_top2[1] - old_top2[0]) >= cost_margin
            mutual = mutual & new_unambiguous & old_unambiguous[new_to_old]
        valid_new = torch.nonzero(mutual, as_tuple=False).flatten()
        valid_old = new_to_old[valid_new]
        pair_ok = valid[valid_new, valid_old]
        valid_new, valid_old = valid_new[pair_ok], valid_old[pair_ok]
        min_pairs = int(os.environ.get("POINT3R_LC_ODOM_MIN_PAIRS", "10"))
        if valid_new.numel() < min_pairs:
            return None
        full_se3 = (not force_rotation) and os.environ.get(
            "POINT3R_LC_FULL_SE3", "0"
        ).lower() in ("1", "true", "yes", "on")
        if full_se3:
            src_local = write_local_j[valid_new].float()
            dst_local = old_local[valid_old].float()
            estimate = self._lc_ransac_se3(
                src_local,
                dst_local,
                float(os.environ.get("POINT3R_LC_ODOM_SE3_RANSAC", "0.10")),
                int(os.environ.get("POINT3R_LC_ODOM_RANSAC_ITERS", "96")),
                seed=(frame_i + 1) * 2654435761 + (target_frame + 1) * 2246822519,
            )
            if estimate is None:
                return None
            measured, inliers, rmse = estimate
            inlier_count = int(inliers.sum().item())
            ratio = inlier_count / max(1, int(valid_new.numel()))
            if inlier_count < min_pairs or ratio < float(os.environ.get(
                "POINT3R_LC_ODOM_SE3_MIN_RATIO",
                os.environ.get("POINT3R_LC_ODOM_MIN_RATIO", "0.60"),
            )):
                return None
            c2w_old = self._pose_trajectory[target_frame] if target_frame < len(self._pose_trajectory) else None
            c2w_new = self._pose_trajectory[frame_i] if frame_i < len(self._pose_trajectory) else None
            if c2w_old is None or c2w_new is None:
                return None
            c2w_old = c2w_old.to(device=device, dtype=torch.float32)
            c2w_new = c2w_new.to(device=device, dtype=torch.float32)
            predicted = torch.linalg.inv(c2w_old) @ c2w_new
            delta = measured @ torch.linalg.inv(predicted)
            delta_deg = float(torch.rad2deg(torch.acos(
                ((torch.trace(delta[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
            )).item())
            delta_trans = float(delta[:3, 3].norm().item())

            # Translation from point maps is accepted only when both point
            # sets have compatible scale and non-degenerate 3-D support.
            src_in = src_local[inliers]
            dst_in = dst_local[inliers]
            src_centered = src_in - src_in.mean(dim=0)
            dst_centered = dst_in - dst_in.mean(dim=0)
            src_spread = src_centered.norm(dim=-1).median().clamp_min(1e-5)
            dst_spread = dst_centered.norm(dim=-1).median().clamp_min(1e-5)
            scale_ratio = float((dst_spread / src_spread).item())
            covariance = src_centered.T @ src_centered / max(1, src_in.shape[0])
            eigvals = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            observability = float((eigvals[0] / eigvals[-1].clamp_min(1e-6)).item())
            if not (
                float(os.environ.get("POINT3R_LC_ODOM_SCALE_MIN", "0.80"))
                <= scale_ratio
                <= float(os.environ.get("POINT3R_LC_ODOM_SCALE_MAX", "1.25"))
            ):
                return None
            if observability < float(os.environ.get(
                "POINT3R_LC_ODOM_MIN_OBSERVABILITY", "0.002"
            )):
                return None
            if delta_deg > float(os.environ.get(
                "POINT3R_LC_ODOM_MAX_CORRECTION_DEG", "5.0"
            )) or delta_trans > float(os.environ.get(
                "POINT3R_LC_ODOM_MAX_TRANS_CORRECTION", "0.20"
            )):
                return None

            rot_blend = max(0.0, min(1.0, float(os.environ.get(
                "POINT3R_LC_ODOM_ROT_BLEND", "1.0"
            ))))
            trans_blend = max(0.0, min(1.0, float(os.environ.get(
                "POINT3R_LC_ODOM_TRANS_BLEND", "0.35"
            ))))
            blended_rot = (
                (1.0 - rot_blend) * predicted[:3, :3]
                + rot_blend * measured[:3, :3]
            )
            u_blend, _, vh_blend = torch.linalg.svd(blended_rot)
            det_fix = torch.eye(3, device=device)
            det_fix[-1, -1] = torch.det(u_blend @ vh_blend)
            measured[:3, :3] = u_blend @ det_fix @ vh_blend
            measured[:3, 3] = (
                (1.0 - trans_blend) * predicted[:3, 3]
                + trans_blend * measured[:3, 3]
            )

            rmse_scale = max(1e-4, float(os.environ.get(
                "POINT3R_LC_ODOM_SE3_QUALITY_RMSE", "0.05"
            )))
            trans_scale = max(1e-4, float(os.environ.get(
                "POINT3R_LC_ODOM_SE3_QUALITY_TRANS", "0.10"
            )))
            support_scale = max(1.0, float(os.environ.get(
                "POINT3R_LC_ODOM_QUALITY_SUPPORT", "50"
            )))
            quality = (
                ratio * ratio
                * min(1.0, inlier_count / support_scale)
                * min(1.0, observability / 0.02)
                * math.exp(-((float(rmse) / rmse_scale) ** 2))
                * math.exp(-((delta_trans / trans_scale) ** 2))
            )
            info = self._lc_fisher_information(
                src_in, measured, float(rmse), ratio
            ) * max(1e-4, quality)
            self._lc_emit(
                f"[LC_ODOM_SE3] frame={frame_i} target={target_frame} "
                f"pairs={valid_new.numel()} inliers={inlier_count} ratio={ratio:.4f} "
                f"rmse={float(rmse):.6f} corr_deg={delta_deg:.4f} "
                f"corr_trans={delta_trans:.6f} scale={scale_ratio:.4f} "
                f"observability={observability:.6f} quality={quality:.6f}"
            )
            return measured.cpu(), info.cpu()
        odom_essential_rotation = os.environ.get(
            "POINT3R_LC_ODOM_ESSENTIAL_ROTATION", "0"
        ).lower() in ("1", "true", "yes", "on")
        odom_centered_rotation = os.environ.get(
            "POINT3R_LC_ODOM_CENTERED_ROTATION", "0"
        ).lower() in ("1", "true", "yes", "on")
        rotation_src = write_local_j[valid_new]
        rotation_dst = old_local[valid_old]
        if odom_centered_rotation and not odom_essential_rotation:
            # Camera-frame bearings cannot be related by a pure rotation when
            # the camera translates: parallax is then misread as orientation.
            # For rigid correspondences p_old = R p_new + t, subtracting the
            # paired centroids removes t exactly and leaves an SO(3)-only
            # relation.  Normalization below also makes the estimate
            # insensitive to an isotropic point-map scale mismatch.
            rotation_src = rotation_src - rotation_src.mean(
                dim=0, keepdim=True
            )
            rotation_dst = rotation_dst - rotation_dst.mean(
                dim=0, keepdim=True
            )
        if odom_essential_rotation:
            estimate = self._lc_essential_so3(
                rotation_src,
                rotation_dst,
                float(os.environ.get(
                    "POINT3R_LC_ODOM_ESSENTIAL_THRESHOLD", "0.003"
                )),
                float(os.environ.get(
                    "POINT3R_LC_ODOM_ESSENTIAL_PROB", "0.999"
                )),
            )
        else:
            estimate = self._lc_ransac_so3(
                rotation_src, rotation_dst,
                float(os.environ.get("POINT3R_LC_ODOM_RANSAC_DEG", "2.0")),
                int(os.environ.get("POINT3R_LC_ODOM_RANSAC_ITERS", "96")),
                seed=(frame_i + 1) * 2654435761 + (target_frame + 1) * 2246822519,
            )
        if estimate is None:
            return None
        T_current_to_old, inliers, rmse = estimate
        ratio = int(inliers.sum().item()) / max(1, int(valid_new.numel()))
        if int(inliers.sum().item()) < min_pairs or ratio < float(os.environ.get("POINT3R_LC_ODOM_MIN_RATIO", "0.60")):
            return None
        c2w_old = self._pose_trajectory[target_frame] if target_frame < len(self._pose_trajectory) else None
        c2w_new = self._pose_trajectory[frame_i] if frame_i < len(self._pose_trajectory) else None
        if c2w_old is None or c2w_new is None:
            return None
        c2w_old = c2w_old.to(device=device, dtype=torch.float32)
        c2w_new = c2w_new.to(device=device, dtype=torch.float32)
        predicted = torch.linalg.inv(c2w_old) @ c2w_new
        delta_r = T_current_to_old[:3, :3] @ predicted[:3, :3].T
        delta_deg = float(torch.rad2deg(torch.acos(((torch.trace(delta_r) - 1.0) * 0.5).clamp(-1.0, 1.0))).item())
        if delta_deg > float(os.environ.get("POINT3R_LC_ODOM_MAX_CORRECTION_DEG", "5.0")):
            return None
        blend = max(0.0, min(1.0, float(os.environ.get("POINT3R_LC_ODOM_ROT_BLEND", "1.0"))))
        blended = (1.0 - blend) * predicted[:3, :3] + blend * T_current_to_old[:3, :3]
        U, _, Vt = torch.linalg.svd(blended)
        correction = torch.eye(3, device=device)
        correction[-1, -1] = torch.det(U @ Vt)
        R_blend = U @ correction @ Vt
        T_current_to_old = predicted.clone()
        T_current_to_old[:3, :3] = R_blend
        # RayAway-style reliability: a geometrically plausible edge should have
        # broad support, a tight SO(3) consensus, and should not violently
        # disagree with the network's adjacent-pose prediction.  The old
        # ratio/rmse score saturated almost every edge, so dynamic-object
        # matches in Sintel received the same graph weight as static geometry.
        rmse_deg = float(rmse * 180.0 / float(torch.pi))
        inlier_count = int(inliers.sum().item())
        rmse_scale = max(1e-3, float(os.environ.get("POINT3R_LC_ODOM_QUALITY_RMSE_DEG", "1.0")))
        correction_scale = max(1e-3, float(os.environ.get("POINT3R_LC_ODOM_QUALITY_CORR_DEG", "2.0")))
        support_scale = max(1.0, float(os.environ.get("POINT3R_LC_ODOM_QUALITY_SUPPORT", "50")))
        max_rot_weight = max(1e-6, float(os.environ.get("POINT3R_LC_ODOM_MAX_ROT_WEIGHT", "5.0")))
        support = min(1.0, inlier_count / support_scale)
        information_vectors = (
            rotation_src[inliers]
            if odom_centered_rotation and not odom_essential_rotation
            else write_local_j[valid_new][inliers]
        )
        inlier_rays = F.normalize(information_vectors.float(), dim=-1)
        ray_information = torch.eye(3, device=device).unsqueeze(0) - torch.einsum(
            "ni,nj->nij", inlier_rays, inlier_rays
        )
        coverage_min_eig = float(
            torch.linalg.eigvalsh(ray_information.mean(dim=0))[0].clamp_min(0.0).item()
        )
        coverage_target = max(
            1e-6, float(os.environ.get("POINT3R_LC_ODOM_COVERAGE_TARGET", "0.02"))
        )
        coverage_floor = max(
            0.0, min(1.0, float(os.environ.get("POINT3R_LC_ODOM_COVERAGE_FLOOR", "0.25")))
        )
        coverage_enabled = os.environ.get(
            "POINT3R_LC_ODOM_COVERAGE_ENABLED", "0"
        ).lower() in ("1", "true", "yes", "on")
        coverage_factor = 1.0
        if coverage_enabled:
            coverage_factor = coverage_floor + (1.0 - coverage_floor) * min(
                1.0, coverage_min_eig / coverage_target
            )
        quality = (
            ratio * ratio * support * coverage_factor
            * math.exp(-((rmse_deg / rmse_scale) ** 2))
            * math.exp(-((delta_deg / correction_scale) ** 2))
        )
        binary_quality = os.environ.get(
            "POINT3R_LC_ODOM_BINARY_QUALITY", "0"
        ).lower() in ("1", "true", "yes", "on")
        quality_threshold = float(os.environ.get("POINT3R_LC_ODOM_QUALITY_THRESHOLD", "0.05"))
        if binary_quality:
            rot_weight = max_rot_weight if quality >= quality_threshold else 0.0
        else:
            rot_weight = max_rot_weight * quality
        trans_weight = float(os.environ.get("POINT3R_LC_ODOM_TRANS_WEIGHT", "0.05"))
        info = torch.eye(6, device=device)
        anisotropic_info = os.environ.get(
            "POINT3R_LC_ANISOTROPIC_INFORMATION", "0"
        ).lower() in ("1", "true", "yes", "on")
        if anisotropic_info and rot_weight > 0.0:
            # Keep the directional observability carried by the inlier rays.
            # Previous versions collapsed this matrix to scalar * I, losing
            # exactly the planar/axial distinction that the coverage audit
            # was intended to measure.
            rot_info = ray_information.mean(dim=0)
            eigval, eigvec = torch.linalg.eigh(rot_info)
            relative_floor = float(os.environ.get(
                "POINT3R_LC_ROT_INFO_EIG_FLOOR", "0.05"
            ))
            eigval = eigval.clamp_min(
                eigval.max().clamp_min(1e-6) * relative_floor
            )
            rot_info = (eigvec * eigval.unsqueeze(0)) @ eigvec.T
            rot_info = rot_info * (
                3.0 * rot_weight / rot_info.trace().clamp_min(1e-6)
            )
            info[:3, :3] = rot_info
        else:
            info[:3, :3] *= rot_weight
        info[3:, 3:] *= trans_weight
        self._lc_emit(
            f"[LC_ODOM] frame={frame_i} target={target_frame} lag={frame_i-target_frame} "
            f"pairs={valid_new.numel()} inliers={int(inliers.sum().item())} "
            f"ratio={ratio:.4f} rmse_deg={rmse_deg:.4f} correction_deg={delta_deg:.4f} "
            f"coverage_eig={coverage_min_eig:.6f} coverage={coverage_factor:.4f} "
            f"quality={quality:.4f} graph_weight={rot_weight:.4f} "
            f"centered={int(odom_centered_rotation and not odom_essential_rotation)} "
            f"essential={int(odom_essential_rotation)}"
        )
        return T_current_to_old.cpu(), info.cpu()

    def _lc_detect_and_estimate(self, j, write_pos_j, write_local_j, write_ray_j, write_feat_j, frame_i, c2w_new, device):
        """
        在 q25 筛选后、memory update 前做 loop detection。
        对每个历史帧，用 3D correspondence + Umeyama 独立估计 T_ij。
        返回 list of (fi, fj, T_ij_4x4, info_6x6)。
        """
        if self._lc_pos is None or j >= len(self._lc_pos):
            return []
        if self._lc_pos[j] is None or self._lc_pos[j].shape[0] == 0:
            return []

        keyframe_stride = max(1, int(os.environ.get("POINT3R_LC_KEYFRAME_STRIDE", "5")))
        if frame_i % keyframe_stride != 0:
            return []

        eps_pos  = float(os.environ.get("POINT3R_LC_EPS_POS",  "0.75"))
        eps_ang  = float(os.environ.get("POINT3R_LC_EPS_ANG",  "0.015"))
        delta_t  = int(os.environ.get("POINT3R_LC_DELTA_T",    "15"))
        min_pairs = int(os.environ.get("POINT3R_LC_MIN_PAIRS", "12"))
        debug = os.environ.get("POINT3R_LC_DEBUG", "0").lower() in ("1", "true", "yes", "on")

        if self._lc_diag is None:
            self._lc_diag = {
                "frames": 0, "history_frames": 0, "spatial_pairs": 0,
                "ray_pairs": 0, "mutual_pairs": 0, "verified_edges": 0,
            }
        self._lc_diag["frames"] += 1

        if c2w_new is not None:
            cam_c_new = c2w_new[:3, 3].to(device)
        else:
            cam_c_new = write_pos_j.new_zeros(3)

        ray_new = write_ray_j

        old_pos = self._lc_pos[j]
        old_fid = self._lc_fid[j]
        old_local = self._lc_local[j]
        old_ray = self._lc_ray[j]
        old_feat = self._lc_feat[j]
        new_feat = F.normalize(write_feat_j.float(), dim=-1)

        scored_candidates = []
        frame_best = {"spatial": 0, "ray": 0, "feature": 0, "mutual": 0, "inliers": 0}
        for fid_val in torch.unique(old_fid).tolist():
            fid_int = int(fid_val)
            if abs(frame_i - fid_int) <= delta_t:
                continue
            if fid_int >= len(self._pose_trajectory) or self._pose_trajectory[fid_int] is None:
                continue
            self._lc_diag["history_frames"] += 1

            c2w_old = self._pose_trajectory[fid_int].to(device)
            cam_c_old = c2w_old[:3, 3]
            mask_old = old_fid == fid_int
            old_pos_f   = old_pos[mask_old]    # [K,3]
            old_local_f = old_local[mask_old]  # [K,3]
            ray_old_f = old_ray[mask_old]
            old_feat_f = old_feat[mask_old]

            # 距离 + ray angle 过滤
            dist = torch.cdist(write_pos_j.float(), old_pos_f.float())  # [N,K]
            d_ang = 1.0 - (ray_new.unsqueeze(1) * ray_old_f.unsqueeze(0)).sum(-1)  # [N,K]
            spatial_mask = dist < eps_pos
            ray_mask = spatial_mask & (d_ang > eps_ang)
            feat_sim = new_feat @ old_feat_f.T
            feat_thresh = float(os.environ.get("POINT3R_LC_FEAT_SIM", "0.50"))
            loop_mask = ray_mask & (feat_sim > feat_thresh)
            n_spatial = int(spatial_mask.sum().item())
            n_ray = int(ray_mask.sum().item())
            n_feature = int(loop_mask.sum().item())
            frame_best["spatial"] = max(frame_best["spatial"], n_spatial)
            frame_best["ray"] = max(frame_best["ray"], n_ray)
            frame_best["feature"] = max(frame_best["feature"], n_feature)
            self._lc_diag["spatial_pairs"] += n_spatial
            self._lc_diag["ray_pairs"] += n_ray

            if int(loop_mask.sum().item()) < min_pairs:
                continue

            # Build one-to-one correspondences. Strict mutual-nearest matching
            # can collapse to zero pairs when several sparse tokens quantize to
            # the same spatial/ray neighbourhood. Keep each new token's best
            # admissible old token, then greedily keep the lowest-cost proposal
            # for every old token. RANSAC and the inlier-ratio gate below remain
            # responsible for geometric verification.
            feat_weight = float(os.environ.get("POINT3R_LC_FEAT_WEIGHT", "0.25"))
            dist_masked = dist / max(eps_pos, 1e-6) + feat_weight * (1.0 - feat_sim)
            dist_masked[~loop_mask] = float('inf')

            pair_new, pair_old = torch.nonzero(loop_mask, as_tuple=True)
            if pair_new.numel() == 0:
                continue
            pair_cost = dist_masked[pair_new, pair_old]
            proposal_cap = max(
                1024, int(os.environ.get("POINT3R_LC_MAX_PAIR_PROPOSALS", "32768"))
            )
            if pair_cost.numel() > proposal_cap:
                _, proposal_order = torch.topk(
                    pair_cost, k=proposal_cap, largest=False, sorted=True
                )
            else:
                proposal_order = torch.argsort(pair_cost)
            max_corr = max(min_pairs, int(os.environ.get("POINT3R_LC_MAX_CORRESPONDENCES", "256")))
            used_new = set()
            used_old = set()
            keep_new = []
            keep_old = []
            ordered_new = pair_new[proposal_order].tolist()
            ordered_old = pair_old[proposal_order].tolist()
            for new_idx, old_idx in zip(ordered_new, ordered_old):
                if new_idx in used_new or old_idx in used_old:
                    continue
                used_new.add(new_idx)
                used_old.add(old_idx)
                keep_new.append(new_idx)
                keep_old.append(old_idx)
                if len(keep_new) >= max_corr:
                    break
            valid_new = torch.tensor(keep_new, device=device, dtype=torch.long)
            valid_old = torch.tensor(keep_old, device=device, dtype=torch.long)

            if valid_new.numel() < min_pairs:
                continue

            n_mutual = int(valid_new.numel())
            frame_best["mutual"] = max(frame_best["mutual"], n_mutual)
            self._lc_diag["mutual_pairs"] += n_mutual

            # 局部坐标
            src_local = write_local_j[valid_new]
            dst_local = old_local_f[valid_old]                                                # [P,3]

            ransac_thresh = float(os.environ.get("POINT3R_LC_RANSAC_THRESH", "0.15"))
            ransac_iters = int(os.environ.get("POINT3R_LC_RANSAC_ITERS", "96"))
            rotation_only = os.environ.get(
                "POINT3R_LC_ROTATION_ONLY", "0"
            ).lower() in ("1", "true", "yes", "on")
            if rotation_only:
                estimate = self._lc_ransac_so3(
                    src_local, dst_local,
                    float(os.environ.get("POINT3R_LC_ROT_RANSAC_DEG", "3.0")),
                    ransac_iters,
                    seed=(frame_i + 1) * 100003 + fid_int,
                )
            else:
                estimate = self._lc_ransac_se3(
                    src_local, dst_local, ransac_thresh, ransac_iters,
                    seed=(frame_i + 1) * 100003 + fid_int,
                )
            if estimate is None:
                continue
            T_ij, inlier_mask, inlier_rmse = estimate
            if rotation_only and c2w_new is not None:
                # Ray directions constrain orientation strongly, while depth
                # scale variation makes translation from point maps unstable.
                # Keep the network's relative translation and refine rotation.
                T_pred_for_translation = torch.linalg.inv(c2w_old.float()) @ c2w_new.float()
                T_ij[:3, 3] = T_pred_for_translation[:3, 3]
            n_inliers = int(inlier_mask.sum().item())
            inlier_ratio = n_inliers / max(1, valid_new.numel())
            frame_best["inliers"] = max(frame_best["inliers"], n_inliers)

            # 检查 inlier ratio 是否足够
            min_inlier_ratio = float(os.environ.get("POINT3R_LC_MIN_INLIER_RATIO", "0.5"))
            if n_inliers < min_pairs or inlier_ratio < min_inlier_ratio:
                continue

            if c2w_new is not None:
                T_pred = torch.linalg.inv(c2w_old.float()) @ c2w_new.float()
                delta = T_ij.float() @ torch.linalg.inv(T_pred)
                cos_delta = ((torch.trace(delta[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
                correction_rot_deg = float(torch.rad2deg(torch.acos(cos_delta)).item())
                correction_trans = float(delta[:3, 3].norm().item())
                max_rot_deg = float(os.environ.get("POINT3R_LC_MAX_ROT_CORRECTION_DEG", "3.0"))
                max_trans = float(os.environ.get("POINT3R_LC_MAX_TRANS_CORRECTION", "0.20"))
                if os.environ.get("POINT3R_LC_LOG_MEASUREMENTS", "0").lower() in ("1", "true", "yes", "on"):
                    tij_text = ",".join(f"{float(v):.8g}" for v in T_ij.detach().cpu().reshape(-1))
                    pred_text = ",".join(f"{float(v):.8g}" for v in T_pred.detach().cpu().reshape(-1))
                    self._lc_emit(
                        f"[LC_MEAS] fi={frame_i} fj={fid_int} inliers={n_inliers} "
                        f"ratio={inlier_ratio:.5f} rmse={inlier_rmse:.6f} "
                        f"corr_rot_deg={correction_rot_deg:.6f} corr_trans={correction_trans:.6f} "
                        f"Tij={tij_text} Tpred={pred_text}"
                    )
                if correction_rot_deg > max_rot_deg or (not rotation_only and correction_trans > max_trans):
                    continue

            # Fisher-style information from the verified 3D correspondences.
            # An isotropic eye matrix over-constrains weak/planar directions and
            # was the main cause of large ATE regressions in early probes.
            info = self._lc_fisher_information(
                src_local[inlier_mask], T_ij, inlier_rmse, inlier_ratio
            )
            if rotation_only:
                trans_scale = float(os.environ.get("POINT3R_LC_TRANS_INFO_SCALE", "0.01"))
                info[:3, 3:] *= trans_scale
                info[3:, :3] *= trans_scale
                info[3:, 3:] *= trans_scale
                loop_rmse_deg = float(inlier_rmse * 180.0 / float(torch.pi))
                rmse_scale = max(
                    1e-3, float(os.environ.get("POINT3R_LC_LOOP_QUALITY_RMSE_DEG", "1.0"))
                )
                correction_scale = max(
                    1e-3, float(os.environ.get("POINT3R_LC_LOOP_QUALITY_CORR_DEG", "10.0"))
                )
                support_scale = max(
                    1.0, float(os.environ.get("POINT3R_LC_LOOP_QUALITY_SUPPORT", "50"))
                )
                max_loop_weight = max(
                    1e-6, float(os.environ.get("POINT3R_LC_LOOP_MAX_ROT_WEIGHT", "5.0"))
                )
                support = min(1.0, n_inliers / support_scale)
                loop_quality = (
                    inlier_ratio * inlier_ratio * support
                    * math.exp(-((loop_rmse_deg / rmse_scale) ** 2))
                    * math.exp(-((correction_rot_deg / correction_scale) ** 2))
                )
                loop_weight = max_loop_weight * loop_quality
                info[:3, :3] = torch.eye(3, device=info.device, dtype=info.dtype) * loop_weight
                self._lc_emit(
                    f"[LC_LOOP_EDGE] frame={frame_i} target={fid_int} lag={frame_i-fid_int} "
                    f"inliers={n_inliers} ratio={inlier_ratio:.4f} rmse_deg={loop_rmse_deg:.4f} "
                    f"correction_deg={correction_rot_deg:.4f} quality={loop_quality:.4f} "
                    f"graph_weight={loop_weight:.4f}"
                )
            else:
                self._lc_emit(
                    f"[LC_LOOP_EDGE_SE3] frame={frame_i} target={fid_int} lag={frame_i-fid_int} "
                    f"pairs={valid_new.numel()} inliers={n_inliers} ratio={inlier_ratio:.4f} "
                    f"rmse={inlier_rmse:.6f} correction_deg={correction_rot_deg:.4f} "
                    f"correction_trans={correction_trans:.6f}"
                )
            temporal_bonus = 1.0 + min(abs(frame_i - fid_int), 200) / 1000.0
            score = n_inliers * inlier_ratio * temporal_bonus / max(inlier_rmse, 0.01)
            scored_candidates.append((score, (frame_i, fid_int, T_ij.cpu(), info.cpu())))

        max_edges = int(os.environ.get("POINT3R_LC_MAX_EDGES_PER_FRAME", "1"))
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = [item[1] for item in scored_candidates[:max_edges]]
        self._lc_diag["verified_edges"] += len(candidates)
        if debug and (frame_i % 10 == 0 or candidates):
            self._lc_emit(
                f"[LC_AUDIT] frame={frame_i} history={int((old_fid < frame_i-delta_t).sum().item())} "
                f"best_spatial={frame_best['spatial']} best_ray={frame_best['ray']} "
                f"best_feature={frame_best['feature']} "
                f"best_mutual={frame_best['mutual']} best_inliers={frame_best['inliers']} "
                f"verified={len(candidates)} total_verified={self._lc_diag['verified_edges']}"
            )
        return candidates

    @staticmethod
    def _lc_fisher_information(src_inliers, transform, rmse, inlier_ratio):
        """SE(3) point-to-point Fisher matrix, ordered [rot, trans]."""
        device = src_inliers.device
        dtype = torch.float32
        if src_inliers.numel() == 0:
            return torch.eye(6, device=device, dtype=dtype)
        pts = (
            transform[:3, :3].float() @ src_inliers.float().T
        ).T + transform[:3, 3].float().unsqueeze(0)
        n = pts.shape[0]
        skew = torch.zeros((n, 3, 3), device=device, dtype=dtype)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        skew[:, 0, 1], skew[:, 0, 2] = -z, y
        skew[:, 1, 0], skew[:, 1, 2] = z, -x
        skew[:, 2, 0], skew[:, 2, 1] = -y, x
        eye = torch.eye(3, device=device, dtype=dtype).expand(n, -1, -1)
        jac = torch.cat((-skew, eye), dim=2)
        fisher = torch.einsum("nki,nkj->ij", jac, jac) / max(1, n)
        # Preserve anisotropy while keeping a bounded total edge strength.
        eigvals, eigvecs = torch.linalg.eigh(fisher)
        eig_floor = float(os.environ.get("POINT3R_LC_FISHER_EIG_FLOOR", "0.02"))
        eigvals = eigvals.clamp_min(max(eig_floor, 1e-6))
        fisher = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
        target_trace = min(
            float(os.environ.get("POINT3R_LC_INFO_MAX_TRACE", "30.0")),
            6.0 * float(inlier_ratio) / max(float(rmse), 0.02),
        )
        fisher = fisher * (target_trace / fisher.trace().clamp_min(1e-6))
        return fisher

    def _lc_ransac_se3(self, src, dst, threshold, iterations, seed):
        """Deterministic 3D-3D RANSAC followed by an inlier-only SE(3) fit."""
        n = int(src.shape[0])
        if n < 4:
            return None
        try:
            generator = torch.Generator(device=src.device)
            generator.manual_seed(int(seed))
        except Exception:
            generator = None
        best_mask = None
        best_count = 0
        best_rmse = float("inf")
        for _ in range(max(1, iterations)):
            if generator is None:
                sample = torch.randperm(n, device=src.device)[:4]
            else:
                sample = torch.randperm(n, generator=generator, device=src.device)[:4]
            T = self._umeyama_se3(src[sample], dst[sample])
            if T is None:
                continue
            residual = ((T[:3, :3] @ src.T).T + T[:3, 3] - dst).norm(dim=-1)
            mask = residual < threshold
            count = int(mask.sum().item())
            if count < 4:
                continue
            rmse = float(torch.sqrt(torch.mean(residual[mask] ** 2)).item())
            if count > best_count or (count == best_count and rmse < best_rmse):
                best_mask, best_count, best_rmse = mask, count, rmse
        if best_mask is None:
            return None
        refined = self._umeyama_se3(src[best_mask], dst[best_mask])
        if refined is None:
            return None
        residual = ((refined[:3, :3] @ src.T).T + refined[:3, 3] - dst).norm(dim=-1)
        final_mask = residual < threshold
        if int(final_mask.sum().item()) < 4:
            return None
        refined = self._umeyama_se3(src[final_mask], dst[final_mask])
        residual = ((refined[:3, :3] @ src[final_mask].T).T + refined[:3, 3] - dst[final_mask]).norm(dim=-1)
        rmse = float(torch.sqrt(torch.mean(residual ** 2)).item())
        return refined, final_mask, rmse

    def _pose_static_memory_refine(
        self,
        frame_i,
        current_world,
        current_local,
        current_feat,
        current_conf,
        predicted_c2w,
        memory_world,
        memory_feat,
        memory_conf,
    ):
        """Refine the current pose against persistent static memory.

        This is deliberately causal: only memory from earlier frames is used,
        and only the current camera pose is corrected.  Persistent memory is
        not warped by the correction, avoiding the online bank oscillation
        observed in v33/v34.
        """
        enabled = os.environ.get(
            "POINT3R_POSE_STATIC_REFINE", "0"
        ).lower() in ("1", "true", "yes", "on")
        if not enabled or frame_i <= 0 or predicted_c2w is None:
            return None
        if (
            current_local is None
            or current_world is None
            or current_feat is None
            or memory_world is None
            or memory_feat is None
            or current_local.shape[0] < 12
            or memory_world.shape[0] < 12
        ):
            return None

        device = current_world.device
        max_current = max(32, int(os.environ.get(
            "POINT3R_POSE_STATIC_MAX_CURRENT", "256"
        )))
        max_memory = max(128, int(os.environ.get(
            "POINT3R_POSE_STATIC_MAX_MEMORY", "4096"
        )))
        if current_conf is not None and current_conf.numel() == current_world.shape[0]:
            current_conf = torch.nan_to_num(
                current_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            current_ids = torch.topk(
                current_conf, k=min(max_current, current_conf.numel()), largest=True
            ).indices
        else:
            current_ids = torch.linspace(
                0, current_world.shape[0] - 1,
                min(max_current, current_world.shape[0]), device=device,
            ).round().long()
        if memory_conf is not None and memory_conf.numel() == memory_world.shape[0]:
            memory_conf = torch.nan_to_num(
                memory_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            memory_ids = torch.topk(
                memory_conf, k=min(max_memory, memory_conf.numel()), largest=True
            ).indices
        elif memory_world.shape[0] > max_memory:
            memory_ids = torch.linspace(
                0, memory_world.shape[0] - 1, max_memory, device=device,
            ).round().long()
        else:
            memory_ids = torch.arange(memory_world.shape[0], device=device)

        query_world = current_world[current_ids].float()
        query_local = current_local[current_ids].float()
        query_feat = F.normalize(current_feat[current_ids].float(), dim=-1)
        key_world = memory_world[memory_ids].float()
        key_feat = F.normalize(memory_feat[memory_ids].float(), dim=-1)
        radius = max(1e-4, float(os.environ.get(
            "POINT3R_POSE_STATIC_RADIUS", "0.15"
        )))
        spatial = torch.cdist(query_world, key_world)
        neighbour_k = min(
            max(1, int(os.environ.get("POINT3R_POSE_STATIC_NEIGHBORS", "8"))),
            key_world.shape[0],
        )
        near_dist, near_local_ids = torch.topk(
            spatial, k=neighbour_k, largest=False, dim=1
        )
        near_feat = key_feat[near_local_ids]
        feat_sim = torch.einsum("nd,nkd->nk", query_feat, near_feat)
        feat_min = float(os.environ.get("POINT3R_POSE_STATIC_FEAT_SIM", "0.60"))
        admissible = (near_dist < radius) & (feat_sim > feat_min)
        score = near_dist / radius + 0.25 * (1.0 - feat_sim)
        score = score.masked_fill(~admissible, float("inf"))
        best_score, best_near = score.min(dim=1)
        proposed = torch.nonzero(torch.isfinite(best_score), as_tuple=False).flatten()
        min_pairs = max(8, int(os.environ.get(
            "POINT3R_POSE_STATIC_MIN_PAIRS", "20"
        )))
        if proposed.numel() < min_pairs:
            if frame_i % 10 == 0:
                self._lc_emit(
                    f"[POSE_STATIC_AUDIT] frame={frame_i} stage=proposal "
                    f"current={current_ids.numel()} memory={memory_ids.numel()} "
                    f"pairs={proposed.numel()} min_pairs={min_pairs}"
                )
            return None
        proposed = proposed[torch.argsort(best_score[proposed])]
        proposed_old = near_local_ids[proposed, best_near[proposed]]
        used_old = set()
        keep_query = []
        keep_old = []
        for query_id, old_id in zip(proposed.tolist(), proposed_old.tolist()):
            if old_id in used_old:
                continue
            used_old.add(old_id)
            keep_query.append(query_id)
            keep_old.append(old_id)
        if len(keep_query) < min_pairs:
            if frame_i % 10 == 0:
                self._lc_emit(
                    f"[POSE_STATIC_AUDIT] frame={frame_i} stage=unique "
                    f"proposals={proposed.numel()} unique={len(keep_query)} "
                    f"min_pairs={min_pairs}"
                )
            return None
        keep_query = torch.tensor(keep_query, device=device, dtype=torch.long)
        keep_old = torch.tensor(keep_old, device=device, dtype=torch.long)
        src = query_local[keep_query]
        dst = key_world[keep_old]
        estimate = self._lc_ransac_se3(
            src,
            dst,
            float(os.environ.get("POINT3R_POSE_STATIC_RANSAC_THRESH", "0.06")),
            int(os.environ.get("POINT3R_POSE_STATIC_RANSAC_ITERS", "96")),
            seed=(frame_i + 1) * 7919,
        )
        if estimate is None:
            if frame_i % 10 == 0:
                self._lc_emit(
                    f"[POSE_STATIC_AUDIT] frame={frame_i} stage=ransac_none "
                    f"pairs={src.shape[0]}"
                )
            return None
        measured_c2w, inliers, rmse = estimate
        inlier_count = int(inliers.sum().item())
        inlier_ratio = inlier_count / max(1, int(src.shape[0]))
        if (
            inlier_count < min_pairs
            or inlier_ratio < float(os.environ.get(
                "POINT3R_POSE_STATIC_MIN_INLIER_RATIO", "0.70"
            ))
        ):
            if frame_i % 10 == 0:
                self._lc_emit(
                    f"[POSE_STATIC_AUDIT] frame={frame_i} stage=inlier_gate "
                    f"pairs={src.shape[0]} inliers={inlier_count} "
                    f"ratio={inlier_ratio:.4f} rmse={rmse:.6f}"
                )
            return None

        predicted = predicted_c2w.to(device=device, dtype=torch.float32)
        delta = measured_c2w.float() @ torch.linalg.inv(predicted)
        cosine = ((torch.trace(delta[:3, :3]) - 1.0) * 0.5).clamp(-1.0, 1.0)
        correction_deg = float(torch.rad2deg(torch.acos(cosine)).item())
        correction_trans = float(delta[:3, 3].norm().item())
        if (
            correction_deg > float(os.environ.get(
                "POINT3R_POSE_STATIC_MAX_ROT_DEG", "4.0"
            ))
            or correction_trans > float(os.environ.get(
                "POINT3R_POSE_STATIC_MAX_TRANS", "0.12"
            ))
        ):
            self._lc_emit(
                f"[POSE_STATIC_REJECT] frame={frame_i} pairs={src.shape[0]} "
                f"inliers={inlier_count} ratio={inlier_ratio:.4f} rmse={rmse:.6f} "
                f"corr_deg={correction_deg:.4f} corr_trans={correction_trans:.6f}"
            )
            return None

        alpha = max(0.0, min(1.0, float(os.environ.get(
            "POINT3R_POSE_STATIC_ALPHA", "0.35"
        ))))
        mixed = (1.0 - alpha) * torch.eye(3, device=device) + alpha * delta[:3, :3]
        u, _, vh = torch.linalg.svd(mixed)
        delta_rot = u @ vh
        if torch.det(delta_rot) < 0:
            u[:, -1] *= -1
            delta_rot = u @ vh
        delta_blend = torch.eye(4, device=device, dtype=torch.float32)
        delta_blend[:3, :3] = delta_rot
        delta_blend[:3, 3] = alpha * delta[:3, 3]
        corrected = delta_blend @ predicted
        self._lc_emit(
            f"[POSE_STATIC_ACCEPT] frame={frame_i} pairs={src.shape[0]} "
            f"inliers={inlier_count} ratio={inlier_ratio:.4f} rmse={rmse:.6f} "
            f"corr_deg={correction_deg:.4f} corr_trans={correction_trans:.6f} "
            f"alpha={alpha:.3f}"
        )
        return corrected

    def _lc_ransac_so3(self, src, dst, threshold_deg, iterations, seed):
        """Robust Wahba alignment of camera-frame observation rays."""
        src_ray = F.normalize(src.float(), dim=-1)
        dst_ray = F.normalize(dst.float(), dim=-1)
        n = int(src_ray.shape[0])
        if n < 3:
            return None
        generator = torch.Generator(device=src.device)
        generator.manual_seed(int(seed))
        threshold = max(float(threshold_deg), 0.1) * torch.pi / 180.0
        best_mask, best_count, best_rmse = None, 0, float("inf")
        for _ in range(max(1, iterations)):
            sample = torch.randperm(n, generator=generator, device=src.device)[:3]
            R = self._lc_wahba_so3(src_ray[sample], dst_ray[sample])
            if R is None:
                continue
            aligned = (R @ src_ray.T).T
            angular = torch.acos((aligned * dst_ray).sum(-1).clamp(-1.0, 1.0))
            mask = angular < threshold
            count = int(mask.sum().item())
            if count < 3:
                continue
            rmse = float(torch.sqrt(torch.mean(angular[mask] ** 2)).item())
            if count > best_count or (count == best_count and rmse < best_rmse):
                best_mask, best_count, best_rmse = mask, count, rmse
        if best_mask is None:
            return None
        R = self._lc_wahba_so3(src_ray[best_mask], dst_ray[best_mask])
        if R is None:
            return None
        aligned = (R @ src_ray.T).T
        angular = torch.acos((aligned * dst_ray).sum(-1).clamp(-1.0, 1.0))
        final_mask = angular < threshold
        if int(final_mask.sum().item()) < 3:
            return None
        R = self._lc_wahba_so3(src_ray[final_mask], dst_ray[final_mask])
        angular = torch.acos(
            (((R @ src_ray[final_mask].T).T) * dst_ray[final_mask]).sum(-1).clamp(-1.0, 1.0)
        )
        T = torch.eye(4, device=src.device, dtype=torch.float32)
        T[:3, :3] = R
        return T, final_mask, float(torch.sqrt(torch.mean(angular ** 2)).item())

    @staticmethod
    def _lc_essential_so3(src, dst, threshold, probability=0.999):
        """Recover current->target rotation from calibrated bearing matches.

        The essential constraint permits camera translation, unlike pure
        Wahba bearing alignment, so parallax is not absorbed as rotation.
        """
        try:
            import cv2
            import numpy as np

            src_np = src.detach().float().cpu().numpy()
            dst_np = dst.detach().float().cpu().numpy()
            valid = (
                np.isfinite(src_np).all(axis=1)
                & np.isfinite(dst_np).all(axis=1)
                & (np.abs(src_np[:, 2]) > 1e-5)
                & (np.abs(dst_np[:, 2]) > 1e-5)
            )
            valid_ids = np.flatnonzero(valid)
            if valid_ids.size < 8:
                return None
            pts_src = src_np[valid_ids, :2] / src_np[valid_ids, 2:3]
            pts_dst = dst_np[valid_ids, :2] / dst_np[valid_ids, 2:3]
            camera = np.eye(3, dtype=np.float64)
            essential, ransac_mask = cv2.findEssentialMat(
                pts_src.astype(np.float64),
                pts_dst.astype(np.float64),
                cameraMatrix=camera,
                method=cv2.RANSAC,
                prob=float(probability),
                threshold=max(1e-6, float(threshold)),
            )
            if essential is None:
                return None
            essential = np.asarray(essential, dtype=np.float64)
            if essential.shape == (3, 3):
                candidates = [essential]
            elif essential.ndim == 2 and essential.shape[1] == 3:
                candidates = [
                    essential[row:row + 3]
                    for row in range(0, essential.shape[0] - 2, 3)
                ]
            elif essential.ndim == 2 and essential.shape[0] == 3:
                candidates = [
                    essential[:, col:col + 3]
                    for col in range(0, essential.shape[1] - 2, 3)
                ]
            else:
                return None
            best = None
            x1 = np.concatenate(
                (pts_src, np.ones((pts_src.shape[0], 1))), axis=1
            )
            x2 = np.concatenate(
                (pts_dst, np.ones((pts_dst.shape[0], 1))), axis=1
            )
            for candidate in candidates:
                mask_in = None if ransac_mask is None else ransac_mask.copy()
                count, rotation, _, pose_mask = cv2.recoverPose(
                    candidate,
                    pts_src.astype(np.float64),
                    pts_dst.astype(np.float64),
                    cameraMatrix=camera,
                    mask=mask_in,
                )
                if rotation is None or pose_mask is None or int(count) < 8:
                    continue
                local_mask = pose_mask.reshape(-1) > 0
                ex1 = (candidate @ x1.T).T
                etx2 = (candidate.T @ x2.T).T
                numerator = np.sum(x2 * ex1, axis=1)
                denominator = np.sqrt(
                    ex1[:, 0] ** 2 + ex1[:, 1] ** 2
                    + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
                )
                sampson = np.abs(numerator) / np.maximum(
                    denominator, 1e-12
                )
                rmse = float(np.sqrt(np.mean(sampson[local_mask] ** 2)))
                score = (int(local_mask.sum()), -rmse)
                if best is None or score > best[0]:
                    best = (score, rotation, local_mask, rmse)
            if best is None:
                return None
            _, rotation, local_mask, rmse = best
            full_mask = torch.zeros(
                src.shape[0], dtype=torch.bool, device=src.device
            )
            selected_ids = torch.as_tensor(
                valid_ids[local_mask], dtype=torch.long, device=src.device
            )
            full_mask[selected_ids] = True
            transform = torch.eye(
                4, dtype=torch.float32, device=src.device
            )
            transform[:3, :3] = torch.from_numpy(rotation).to(
                device=src.device, dtype=torch.float32
            )
            return transform, full_mask, rmse
        except Exception:
            return None

    @staticmethod
    def _lc_wahba_so3(src_ray, dst_ray):
        try:
            H = src_ray.float().T @ dst_ray.float()
            U, _, Vt = torch.linalg.svd(H)
            correction = torch.eye(3, device=H.device, dtype=H.dtype)
            correction[-1, -1] = torch.det(Vt.T @ U.T)
            return Vt.T @ correction @ U.T
        except Exception:
            return None

    @staticmethod
    def _umeyama_se3(src, dst):
        """
        最小二乘 SE3 估计（无尺度），src/dst: [N,3]。
        返回 [4,4] 变换矩阵，将 src 变换到 dst 坐标系。
        失败返回 None。
        """
        try:
            n = src.shape[0]
            if n < 4:
                return None
            mu_s = src.mean(dim=0)
            mu_d = dst.mean(dim=0)
            src_c = src - mu_s
            dst_c = dst - mu_d
            H = src_c.T @ dst_c  # [3,3]
            U, S, Vt = torch.linalg.svd(H)
            # 修正行列式保证 det(R)=+1
            d = torch.det(Vt.T @ U.T)
            D = torch.diag(torch.tensor([1.0, 1.0, d], device=src.device, dtype=src.dtype))
            R = Vt.T @ D @ U.T
            t = mu_d - R @ mu_s
            T = torch.eye(4, device=src.device, dtype=src.dtype)
            T[:3, :3] = R
            T[:3, 3] = t
            return T
        except Exception:
            return None

    def _lc_run_pgo(self, n_frames):
        """
        用 Open3D PoseGraph 做全局优化。
        loop edge 的 T_ij 来自 Umeyama 独立估计，不来自预测 pose。
        返回 pose_corrections: dict {frame_id -> corrected_c2w [4,4]}
        """
        try:
            import open3d as o3d
            import numpy as np
        except ImportError:
            self._lc_emit("[LC_PGO] open3d unavailable; skipped")
            return {}

        if not self._lc_candidates and not self._lc_odometry_edges:
            return {}

        self._lc_emit(
            f"[LC_PGO] start frames={n_frames} loop_edges={len(self._lc_candidates)} "
            f"ray_odom_edges={len(self._lc_odometry_edges)}"
        )
        pose_graph = o3d.pipelines.registration.PoseGraph()
        original_poses = [
            pose.detach().cpu().clone() if pose is not None else torch.eye(4)
            for pose in self._pose_trajectory
        ]

        for fi in range(n_frames):
            c2w = self._pose_trajectory[fi]
            node_pose = c2w.cpu().numpy() if c2w is not None else np.eye(4)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(node_pose))

        # Sequential network odometry is the backbone prior.  Earlier v29
        # replaced these edges with independently fitted SE(3) measurements;
        # sub-degree per-edge bias then accumulated into 5--10 degree drift.
        # Keep the original chain and add verified measurements as auxiliary
        # uncertain constraints below.
        info_odo = np.eye(6) * float(os.environ.get("POINT3R_LC_ODO_WEIGHT", "1.0"))
        for fi in range(n_frames - 1):
            c2w_i = self._pose_trajectory[fi]
            c2w_j = self._pose_trajectory[fi + 1]
            if c2w_i is None or c2w_j is None:
                continue
            # Open3D edge 语义：T_rel 将 source(i) 对齐到 target(j)
            # 即 T_rel @ p_i = p_j，对应 c2w: T_rel = inv(c2w_j) @ c2w_i
            T_rel = np.linalg.inv(c2w_j.cpu().numpy()) @ c2w_i.cpu().numpy()
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    fi, fi + 1, T_rel, info_odo, uncertain=False))

        # Verified local SE(3) edges supplement, rather than replace, the
        # network trajectory.  _lc_rotation_edges stores measurements as
        # current -> target, whereas an edge target -> current needs the
        # inverse transform.
        full_se3 = os.environ.get(
            "POINT3R_LC_FULL_SE3", "0"
        ).lower() in ("1", "true", "yes", "on")
        if full_se3:
            min_trace = float(os.environ.get(
                "POINT3R_LC_SE3_MIN_INFO_TRACE", "0.10"
            ))
            loop_pairs = {
                (int(loop_current), int(loop_target))
                for loop_current, loop_target, _, _ in self._lc_candidates
            }
            for (current_frame, target_frame), (measured_cur_to_target, measured_info) in self._lc_rotation_edges.items():
                if not (0 <= target_frame < current_frame < n_frames):
                    continue
                if (current_frame, target_frame) in loop_pairs:
                    continue
                if float(measured_info.trace().item()) < min_trace:
                    continue
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        target_frame,
                        current_frame,
                        np.linalg.inv(measured_cur_to_target.numpy()),
                        measured_info.numpy(),
                        uncertain=True,
                    )
                )

        # Loop closure 边（T_ij 来自 Umeyama，有独立几何约束）
        for (fi, fj, T_ij, info_mat) in self._lc_candidates:
            if fi >= n_frames or fj >= n_frames:
                continue
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    fi, fj,
                    T_ij.numpy(),
                    info_mat.numpy(),
                    uncertain=True))

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=float(os.environ.get("POINT3R_LC_PGO_CORR", "0.3")),
            edge_prune_threshold=float(os.environ.get("POINT3R_LC_PGO_PRUNE", "0.25")),
            preference_loop_closure=float(os.environ.get("POINT3R_LC_PGO_LC_PREF", "0.1")),
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

        corrections = {}
        rot_corrections = []
        trans_corrections = []
        for fi in range(n_frames):
            optimized = torch.tensor(
                pose_graph.nodes[fi].pose, dtype=torch.float32)
            corrections[fi] = optimized
            before = original_poses[fi]
            delta_r = optimized[:3, :3] @ before[:3, :3].T
            cos_angle = ((torch.trace(delta_r) - 1.0) * 0.5).clamp(-1.0, 1.0)
            rot_corrections.append(float(torch.rad2deg(torch.acos(cos_angle)).item()))
            trans_corrections.append(float((optimized[:3, 3] - before[:3, 3]).norm().item()))
            if fi < len(self._pose_trajectory):
                self._pose_trajectory[fi] = optimized
        self._lc_pgo_executed = True
        self._lc_emit(
            f"[LC_PGO] done rot_mean_deg={sum(rot_corrections)/max(1,len(rot_corrections)):.6f} "
            f"rot_max_deg={max(rot_corrections, default=0.0):.6f} "
            f"trans_mean={sum(trans_corrections)/max(1,len(trans_corrections)):.6f} "
            f"trans_max={max(trans_corrections, default=0.0):.6f}"
        )
        return corrections

    def _lc_direct_ray_odometry(self, n_frames):
        """Integrate verified adjacent ray rotations; preserve network translations."""
        corrections = {}
        if n_frames <= 0 or not self._pose_trajectory:
            return corrections
        first = self._pose_trajectory[0]
        if first is None:
            return corrections
        corrections[0] = first.detach().cpu().float().clone()
        for fi in range(1, n_frames):
            original = self._pose_trajectory[fi]
            previous_original = self._pose_trajectory[fi - 1]
            if original is None or previous_original is None:
                continue
            pose = original.detach().cpu().float().clone()
            if fi in self._lc_odometry_edges and (fi - 1) in corrections:
                current_to_previous = self._lc_odometry_edges[fi][0].float()
                pose[:3, :3] = corrections[fi - 1][:3, :3] @ current_to_previous[:3, :3]
            corrections[fi] = pose
        for fi, pose in corrections.items():
            if fi < len(self._pose_trajectory):
                self._pose_trajectory[fi] = pose
        self._lc_emit(
            f"[LC_DIRECT_ODOM] frames={n_frames} ray_odom_edges={len(self._lc_odometry_edges)}"
        )
        return corrections

    def _lc_rotation_graph_with_priors(self, n_frames):
        """Chordal SO(3) averaging of ray odometry with network-pose priors."""
        originals = []
        for fi in range(n_frames):
            pose = self._pose_trajectory[fi]
            originals.append(None if pose is None else pose.detach().cpu().float().clone())
        rotations = [None if p is None else p[:3, :3].clone() for p in originals]
        prior_weight = max(1e-6, float(os.environ.get("POINT3R_LC_ROT_GRAPH_PRIOR_WEIGHT", "1.0")))
        edge_cap = max(1e-6, float(os.environ.get("POINT3R_LC_ROT_GRAPH_EDGE_WEIGHT", "5.0")))
        iterations = max(1, int(os.environ.get("POINT3R_LC_ROT_GRAPH_ITERS", "20")))
        robust_enabled = os.environ.get(
            "POINT3R_LC_ROT_GRAPH_IRLS", "0"
        ).lower() in ("1", "true", "yes", "on")
        robust_delta_deg = max(
            1e-3, float(os.environ.get("POINT3R_LC_ROT_GRAPH_IRLS_DELTA_DEG", "2.0"))
        )
        robust_warmup = max(
            0, int(os.environ.get("POINT3R_LC_ROT_GRAPH_IRLS_WARMUP", "3"))
        )
        rotation_edges = self._lc_rotation_edges if self._lc_rotation_edges else {
            (fi, fi - 1): edge for fi, edge in self._lc_odometry_edges.items() if fi > 0
        }
        anisotropic_graph = os.environ.get(
            "POINT3R_LC_ANISOTROPIC_ROT_GRAPH", "0"
        ).lower() in ("1", "true", "yes", "on")

        def project(mat):
            U, _, Vt = torch.linalg.svd(mat)
            correction = torch.eye(3, dtype=mat.dtype)
            correction[-1, -1] = torch.det(U @ Vt)
            return U @ correction @ Vt

        def so3_log(rotation):
            cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
            angle = torch.acos(cosine)
            vee = 0.5 * torch.stack((
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ))
            if float(angle) < 1e-6:
                return vee
            return vee * (angle / torch.sin(angle).clamp_min(1e-6))

        def so3_exp(vector):
            angle = vector.norm()
            if float(angle) < 1e-8:
                return torch.eye(3, dtype=vector.dtype)
            axis = vector / angle
            x, y, z = axis
            skew = torch.stack((
                torch.stack((x * 0.0, -z, y)),
                torch.stack((z, y * 0.0, -x)),
                torch.stack((-y, x, z * 0.0)),
            ))
            eye = torch.eye(3, dtype=vector.dtype)
            return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)

        def robustify(weight, current, target, iteration):
            if not robust_enabled or iteration < robust_warmup:
                return weight
            delta = current.T @ target
            cosine = ((torch.trace(delta) - 1.0) * 0.5).clamp(-1.0, 1.0)
            residual_deg = float(torch.rad2deg(torch.acos(cosine)).item())
            # Cauchy IRLS: preserve graph-consistent constraints while
            # suppressing locally plausible edges that conflict globally.
            scale = residual_deg / robust_delta_deg
            return weight / (1.0 + scale * scale)

        for iteration in range(iterations):
            updated = list(rotations)
            for fi in range(n_frames):
                if originals[fi] is None:
                    continue
                if anisotropic_graph:
                    current = rotations[fi]
                    normal_lie = torch.eye(3, dtype=current.dtype) * prior_weight
                    prior_residual = so3_log(
                        originals[fi][:3, :3] @ current.T
                    )
                    rhs_lie = prior_weight * prior_residual
                    for (current_frame, target_frame), (measured, info) in rotation_edges.items():
                        target = None
                        if fi == current_frame and rotations[target_frame] is not None:
                            target = rotations[target_frame] @ measured[:3, :3].float()
                        elif fi == target_frame and rotations[current_frame] is not None:
                            target = rotations[current_frame] @ measured[:3, :3].float().T
                        if target is None:
                            continue
                        residual_world = so3_log(target @ current.T)
                        residual_deg = float(torch.rad2deg(residual_world.norm()).item())
                        robust_scale = 1.0
                        if robust_enabled and iteration >= robust_warmup:
                            ratio_robust = residual_deg / robust_delta_deg
                            robust_scale = 1.0 / (1.0 + ratio_robust * ratio_robust)
                        info_body = info[:3, :3].float()
                        mean_info = info_body.diagonal().mean().clamp_min(1e-6)
                        capped_mean = min(edge_cap, float(mean_info))
                        info_body = info_body * (capped_mean / float(mean_info))
                        # The ray Fisher matrix is expressed in the current
                        # camera tangent.  Transport it before solving a
                        # left/world-frame rotation increment.
                        info_world = current @ info_body @ current.T
                        info_world = info_world * robust_scale
                        normal_lie += info_world
                        rhs_lie += info_world @ residual_world
                    delta = torch.linalg.solve(
                        normal_lie + 1e-6 * torch.eye(3), rhs_lie
                    )
                    max_step_deg = max(0.05, float(os.environ.get(
                        "POINT3R_LC_ROT_GRAPH_MAX_STEP_DEG", "0.75"
                    )))
                    max_step = max_step_deg * torch.pi / 180.0
                    delta_norm = delta.norm()
                    if float(delta_norm) > max_step:
                        delta = delta * (max_step / delta_norm)
                    updated[fi] = project(so3_exp(delta) @ current)
                    continue
                total = prior_weight * originals[fi][:3, :3]
                weight_sum = prior_weight
                for (current_frame, target_frame), (measured, info) in rotation_edges.items():
                    target = None
                    if fi == current_frame and rotations[target_frame] is not None:
                        target = rotations[target_frame] @ measured[:3, :3].float()
                    elif fi == target_frame and rotations[current_frame] is not None:
                        target = rotations[current_frame] @ measured[:3, :3].float().T
                    if target is None:
                        continue
                    info_rot = info[:3, :3].float()
                    w = min(edge_cap, max(0.0, float(info_rot.diagonal().mean())))
                    w = robustify(w, rotations[fi], target, iteration)
                    total = total + w * target
                    weight_sum += w
                updated[fi] = project(total / max(weight_sum, 1e-6))
            rotations = updated

        corrections = {}
        for fi, original in enumerate(originals):
            if original is None:
                continue
            pose = original.clone()
            pose[:3, :3] = rotations[fi]
            corrections[fi] = pose
            self._pose_trajectory[fi] = pose
        self._lc_emit(
            f"[LC_ROT_GRAPH] frames={n_frames} ray_odom_edges={len(self._lc_odometry_edges)} "
            f"rotation_edges={len(rotation_edges)} "
            f"prior_weight={prior_weight:.4f} edge_cap={edge_cap:.4f} iterations={iterations} "
            f"irls={int(robust_enabled)} anisotropic={int(anisotropic_graph)} "
            f"delta_deg={robust_delta_deg:.4f} warmup={robust_warmup}"
        )
        return corrections

    def _lc_translation_graph_with_priors(self, n_frames, corrections):
        """Optimize camera centers with rotations fixed by the SO(3) graph."""
        if not corrections or not self._lc_translation_edges:
            return corrections
        valid_frames = [fi for fi in range(n_frames) if fi in corrections]
        if not valid_frames:
            return corrections
        frame_to_col = {fi: col for col, fi in enumerate(valid_frames)}
        prior_weight = max(1e-6, float(os.environ.get(
            "POINT3R_LC_TRANS_GRAPH_PRIOR_WEIGHT", "5.0"
        )))
        edge_cap = max(1e-6, float(os.environ.get(
            "POINT3R_LC_TRANS_GRAPH_EDGE_WEIGHT", "2.0"
        )))
        max_correction = max(0.0, float(os.environ.get(
            "POINT3R_LC_TRANS_GRAPH_MAX_CORRECTION", "0.05"
        )))
        n_vars = len(valid_frames)
        anisotropic_graph = os.environ.get(
            "POINT3R_LC_ANISOTROPIC_TRANS_GRAPH", "0"
        ).lower() in ("1", "true", "yes", "on")
        if anisotropic_graph:
            hessian = torch.zeros((3 * n_vars, 3 * n_vars), dtype=torch.float32)
            gradient = torch.zeros((3 * n_vars,), dtype=torch.float32)
        else:
            rows = []
            targets = []
        for fi in valid_frames:
            weight = prior_weight * (100.0 if fi == valid_frames[0] else 1.0)
            col = frame_to_col[fi]
            if anisotropic_graph:
                sl = slice(3 * col, 3 * col + 3)
                weight_matrix = torch.eye(3) * weight
                hessian[sl, sl] += weight_matrix
                gradient[sl] += weight_matrix @ corrections[fi][:3, 3].float()
            else:
                row = torch.zeros(n_vars, dtype=torch.float32)
                root_weight = math.sqrt(weight)
                row[col] = root_weight
                rows.append(row)
                targets.append(corrections[fi][:3, 3].float() * root_weight)
        accepted_edges = 0
        for (current_frame, target_frame), (measured, info) in self._lc_translation_edges.items():
            if current_frame not in frame_to_col or target_frame not in frame_to_col:
                continue
            weight = min(
                edge_cap,
                max(0.0, float(info[3:, 3:].diagonal().mean().item())),
            )
            if weight <= 1e-6:
                continue
            # measured maps current-camera coordinates to target-camera
            # coordinates: t_rel = R_target^T (c_current-c_target).
            target_rotation = corrections[target_frame][:3, :3].float()
            center_delta = target_rotation @ measured[:3, 3].float()
            if anisotropic_graph:
                raw_info = info[3:, 3:].float()
                raw_mean = raw_info.diagonal().mean().clamp_min(1e-6)
                directional = raw_info / raw_mean
                weight_matrix = directional * weight
                ci = frame_to_col[current_frame]
                ti = frame_to_col[target_frame]
                cs = slice(3 * ci, 3 * ci + 3)
                ts = slice(3 * ti, 3 * ti + 3)
                hessian[cs, cs] += weight_matrix
                hessian[ts, ts] += weight_matrix
                hessian[cs, ts] -= weight_matrix
                hessian[ts, cs] -= weight_matrix
                weighted_delta = weight_matrix @ center_delta
                gradient[cs] += weighted_delta
                gradient[ts] -= weighted_delta
            else:
                root_weight = math.sqrt(weight)
                row = torch.zeros(n_vars, dtype=torch.float32)
                row[frame_to_col[current_frame]] = root_weight
                row[frame_to_col[target_frame]] = -root_weight
                rows.append(row)
                targets.append(center_delta * root_weight)
            accepted_edges += 1
        if accepted_edges == 0:
            return corrections
        if anisotropic_graph:
            hessian += torch.eye(3 * n_vars) * 1e-6
            solution = torch.linalg.solve(hessian, gradient).reshape(n_vars, 3)
        else:
            design = torch.stack(rows, dim=0)
            rhs = torch.stack(targets, dim=0)
            solution = torch.linalg.lstsq(design, rhs).solution
        correction_norms = []
        for fi in valid_frames:
            pose = corrections[fi].clone()
            original_t = pose[:3, 3].clone()
            proposed_t = solution[frame_to_col[fi]]
            delta = proposed_t - original_t
            delta_norm = float(delta.norm().item())
            if max_correction > 0.0 and delta_norm > max_correction:
                delta = delta * (max_correction / max(delta_norm, 1e-8))
            pose[:3, 3] = original_t + delta
            corrections[fi] = pose
            self._pose_trajectory[fi] = pose
            correction_norms.append(float(delta.norm().item()))
        self._lc_emit(
            f"[LC_TRANS_GRAPH] frames={len(valid_frames)} edges={accepted_edges} "
            f"prior_weight={prior_weight:.4f} edge_cap={edge_cap:.4f} "
            f"anisotropic={int(anisotropic_graph)} "
            f"mean={sum(correction_norms)/max(1,len(correction_norms)):.6f} "
            f"max={max(correction_norms, default=0.0):.6f}"
        )
        return corrections

    def _lc_translation_chain_with_anchors(self, n_frames, corrections):
        """Correct lag-1 translation increments without global over-smoothing.

        The global camera-center least-squares graph can improve ATE while
        worsening RPE translation because lag-2/3 constraints alter local
        velocity.  This path consumes only verified lag-1 local-coordinate
        SE(3) measurements, blends them by their information, and removes a
        configurable fraction of accumulated segment drift at each anchor.
        """
        if not corrections or not self._lc_translation_edges:
            return corrections
        originals = [
            None if pose is None else pose.detach().cpu().float().clone()
            for pose in self._pose_trajectory
        ]
        segment = max(2, int(os.environ.get(
            "POINT3R_LC_TRANS_CHAIN_SEGMENT", "10"
        )))
        max_beta = max(0.0, min(1.0, float(os.environ.get(
            "POINT3R_LC_TRANS_CHAIN_MAX_BETA", "0.25"
        ))))
        quality_scale = max(1e-6, float(os.environ.get(
            "POINT3R_LC_TRANS_CHAIN_QUALITY_SCALE", "1.0"
        )))
        anchor_strength = max(0.0, min(1.0, float(os.environ.get(
            "POINT3R_LC_TRANS_CHAIN_ANCHOR_STRENGTH", "0.75"
        ))))
        accepted = 0
        beta_values = []
        correction_norms = []
        valid = [fi for fi in range(n_frames) if fi in corrections and originals[fi] is not None]
        if len(valid) < 2:
            return corrections
        for segment_start in range(valid[0], valid[-1] + 1, segment):
            segment_end = min(valid[-1], segment_start + segment)
            if segment_start not in corrections or originals[segment_start] is None:
                continue
            chain_centers = {segment_start: corrections[segment_start][:3, 3].clone()}
            for fi in range(segment_start + 1, segment_end + 1):
                if fi not in corrections or originals[fi] is None or originals[fi - 1] is None:
                    continue
                predicted_delta = originals[fi][:3, 3] - originals[fi - 1][:3, 3]
                corrected_delta = predicted_delta
                edge = self._lc_translation_edges.get((fi, fi - 1))
                if edge is not None:
                    measured, info = edge
                    measured_world = (
                        corrections[fi - 1][:3, :3].float()
                        @ measured[:3, 3].float()
                    )
                    quality = max(0.0, float(info[3:, 3:].diagonal().mean().item()))
                    beta = max_beta * quality / (quality + quality_scale)
                    corrected_delta = (1.0 - beta) * predicted_delta + beta * measured_world
                    accepted += 1
                    beta_values.append(beta)
                previous_center = chain_centers.get(
                    fi - 1, corrections[fi - 1][:3, 3].clone()
                )
                chain_centers[fi] = previous_center + corrected_delta
            if segment_end not in chain_centers or originals[segment_end] is None:
                continue
            endpoint_drift = (
                originals[segment_end][:3, 3] - chain_centers[segment_end]
            ) * anchor_strength
            length = max(1, segment_end - segment_start)
            for fi, proposed in chain_centers.items():
                fraction = float(fi - segment_start) / float(length)
                proposed = proposed + fraction * endpoint_drift
                pose = corrections[fi].clone()
                before = pose[:3, 3].clone()
                pose[:3, 3] = proposed
                corrections[fi] = pose
                self._pose_trajectory[fi] = pose
                correction_norms.append(float((proposed - before).norm().item()))
        self._lc_emit(
            f"[LC_TRANS_CHAIN] frames={len(valid)} edges={accepted} segment={segment} "
            f"mean_beta={sum(beta_values)/max(1,len(beta_values)):.6f} "
            f"anchor={anchor_strength:.3f} "
            f"mean={sum(correction_norms)/max(1,len(correction_norms)):.6f} "
            f"max={max(correction_norms, default=0.0):.6f}"
        )
        return corrections

    def _lc_state_aware_trajectory_regularization(self, corrections):
        """Robustly regularize trajectory acceleration without using GT.

        Ray-aware correspondences and the rotation graph remove part of the
        drift, but their per-frame estimates still contain high-frequency
        pose noise.  This final graph term keeps every predicted pose as a
        unary anchor and penalizes only likely-jitter second differences.
        Large observed accelerations receive a Cauchy-like downweight, so
        genuine abrupt motion is preserved rather than blindly averaged.
        """
        enabled = os.environ.get(
            "POINT3R_LC_STATE_REGULARIZATION", "0"
        ).lower() in ("1", "true", "yes", "on")
        if not enabled or not corrections:
            return corrections
        valid_frames = sorted(
            fi for fi, pose in corrections.items() if pose is not None
        )
        if len(valid_frames) < 4:
            return corrections
        # The temporal second-difference operator assumes a contiguous stream.
        if valid_frames != list(range(valid_frames[0], valid_frames[-1] + 1)):
            self._lc_emit(
                f"[LC_STATE_REG] skipped noncontiguous frames={len(valid_frames)}"
            )
            return corrections
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
        except Exception as error:
            self._lc_emit(f"[LC_STATE_REG] scipy unavailable: {error}")
            return corrections

        def _smooth(values, strength):
            values = np.asarray(values, dtype=np.float64)
            n = values.shape[0]
            if strength <= 0.0 or n < 4:
                return values.copy(), 0.0, 0.0
            d2 = np.zeros((n - 2, n), dtype=np.float64)
            rows = np.arange(n - 2)
            d2[rows, rows] = 1.0
            d2[rows, rows + 1] = -2.0
            d2[rows, rows + 2] = 1.0
            acceleration = d2 @ values
            magnitude = np.linalg.norm(acceleration, axis=1)
            median = float(np.median(magnitude))
            mad = float(np.median(np.abs(magnitude - median)))
            scale = max(median + 2.5 * 1.4826 * mad, 1e-6)
            weight = 1.0 / (1.0 + (magnitude / scale) ** 4)
            normal = np.eye(n, dtype=np.float64) + float(strength) * (
                d2.T @ (weight[:, None] * d2)
            )
            refined = np.linalg.solve(normal, values)
            before = float(np.sqrt(np.mean(np.sum(acceleration ** 2, axis=1))))
            after_acceleration = d2 @ refined
            after = float(np.sqrt(np.mean(np.sum(after_acceleration ** 2, axis=1))))
            return refined, before, after

        pose_cpu = [
            corrections[fi].detach().cpu().float().clone()
            for fi in valid_frames
        ]
        positions = np.stack([pose[:3, 3].numpy() for pose in pose_cpu], axis=0)
        matrices = np.stack([pose[:3, :3].numpy() for pose in pose_cpu], axis=0)
        translation_strength = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_TRANS_STRENGTH", "30.0"
        )))
        rotation_strength = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_ROT_STRENGTH", "1.0"
        )))
        lie_increment = os.environ.get(
            "POINT3R_LC_STATE_LIE_INCREMENT", "0"
        ).lower() in ("1", "true", "yes", "on")
        ray_uncertainty_power = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_UNCERTAINTY_POWER", "8.0"
        )))
        auto_rotation_mode = os.environ.get(
            "POINT3R_LC_STATE_AUTO_ROT_MODE", "0"
        ).lower() in ("1", "true", "yes", "on")
        high_jerk_rad = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_HIGH_JERK_RAD", "0.035"
        )))
        low_rotation_strength = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_LOW_ROT_STRENGTH", "0.1"
        )))
        local_rotation_mode = os.environ.get(
            "POINT3R_LC_STATE_LOCAL_ROT_MODE", "0"
        ).lower() in ("1", "true", "yes", "on")
        low_motion_ray_fusion = os.environ.get(
            "POINT3R_LC_STATE_LOW_RAY_FUSION", "0"
        ).lower() in ("1", "true", "yes", "on")
        ray_fusion_transport_only = os.environ.get(
            "POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY", "0"
        ).lower() in ("1", "true", "yes", "on")
        ray_fusion_prior = max(1e-6, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_FUSION_PRIOR", "1.0"
        )))
        ray_fusion_delta_deg = max(1e-3, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG", "2.0"
        )))
        ray_fusion_max_step_deg = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG", "2.0"
        )))
        cycle_observable_fusion = os.environ.get(
            "POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION", "0"
        ).lower() in ("1", "true", "yes", "on")
        ray_cycle_delta_deg = max(1e-3, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_CYCLE_DELTA_DEG", "1.0"
        )))
        ray_cycle_power = max(0.0, float(os.environ.get(
            "POINT3R_LC_STATE_RAY_CYCLE_POWER", "2.0"
        )))
        rotation_mode = "legacy"
        translation_rotation = None
        angular_accel_rms = 0.0
        if lie_increment:
            def _smooth_velocity(
                values, strength, external_weight=None, robust_weight=True
            ):
                values = np.asarray(values, dtype=np.float64)
                count = values.shape[0]
                if strength <= 0.0 or count < 3:
                    return values.copy(), 0.0, 0.0
                d1 = np.zeros((count - 1, count), dtype=np.float64)
                rows = np.arange(count - 1)
                d1[rows, rows] = -1.0
                d1[rows, rows + 1] = 1.0
                acceleration = d1 @ values
                magnitude = np.linalg.norm(acceleration, axis=1)
                median = float(np.median(magnitude))
                mad = float(np.median(np.abs(magnitude - median)))
                scale = max(median + 2.5 * 1.4826 * mad, 1e-6)
                if robust_weight:
                    weight = 1.0 / (1.0 + (magnitude / scale) ** 4)
                else:
                    weight = np.ones_like(magnitude)
                if external_weight is not None:
                    external_weight = np.asarray(external_weight, dtype=np.float64)
                    if external_weight.shape == weight.shape:
                        weight = weight * np.clip(external_weight, 0.0, 1.0)
                normal = np.eye(count, dtype=np.float64) + float(strength) * (
                    d1.T @ (weight[:, None] * d1)
                )
                refined = np.linalg.solve(normal, values)
                before = float(np.sqrt(np.mean(np.sum(acceleration ** 2, axis=1))))
                after_acceleration = d1 @ refined
                after = float(np.sqrt(np.mean(np.sum(after_acceleration ** 2, axis=1))))
                return refined, before, after

            original_rotation = Rotation.from_matrix(matrices)
            relative_rotation = original_rotation[:-1].inv() * original_rotation[1:]
            angular_velocity = relative_rotation.as_rotvec()
            if angular_velocity.shape[0] >= 2:
                raw_angular_acceleration = np.diff(angular_velocity, axis=0)
                angular_accel_rms = float(np.sqrt(np.mean(np.sum(
                    raw_angular_acceleration ** 2, axis=1
                ))))

            # The adjacent ray edge already contains the reliability used by
            # the robust rotation graph.  Convert it to [0,1] and treat its
            # complement as motion-prior confidence: unreliable geometry gets
            # a strong smooth-motion factor, while reliable geometry rapidly
            # turns that factor off and preserves genuine camera motion.
            edge_quality = np.zeros(len(valid_frames) - 1, dtype=np.float64)
            max_rot_weight = max(1e-6, float(os.environ.get(
                "POINT3R_LC_ODOM_MAX_ROT_WEIGHT", "5.0"
            )))
            for velocity_i in range(len(valid_frames) - 1):
                frame_i = valid_frames[velocity_i + 1]
                target_i = valid_frames[velocity_i]
                edge = self._lc_rotation_edges.get((frame_i, target_i))
                if edge is None or len(edge) < 2 or edge[1] is None:
                    continue
                information = edge[1]
                if torch.is_tensor(information):
                    information = information.detach().cpu().float()
                    quality = float(
                        torch.trace(information[:3, :3]).item()
                        / (3.0 * max_rot_weight)
                    )
                    edge_quality[velocity_i] = max(0.0, min(1.0, quality))
            acceleration_quality = np.minimum(edge_quality[:-1], edge_quality[1:])
            rotation_prior_weight = np.power(
                1.0 - np.clip(acceleration_quality, 0.0, 1.0),
                ray_uncertainty_power,
            )
            if local_rotation_mode:
                # Begin from the conservative v45 anchored prior.  Then apply
                # a strong Lie-increment factor only to local angular-jerk
                # outliers that are also poorly constrained by ray geometry.
                # Requiring both an absolute and robust within-sequence outlier
                # threshold avoids suppressing sustained, genuine camera turns.
                reference = original_rotation[0]
                tangent = (reference.inv() * original_rotation).as_rotvec()
                tangent, weak_before, weak_after = _smooth(
                    tangent, low_rotation_strength
                )
                base_rotation = reference * Rotation.from_rotvec(tangent)
                base_velocity = (
                    base_rotation[:-1].inv() * base_rotation[1:]
                ).as_rotvec()
                base_acceleration = np.diff(base_velocity, axis=0)
                base_magnitude = np.linalg.norm(base_acceleration, axis=1)
                local_median = float(np.median(base_magnitude))
                local_mad = float(np.median(np.abs(
                    base_magnitude - local_median
                )))
                robust_threshold = (
                    local_median + 2.5 * 1.4826 * local_mad
                )
                local_threshold = max(high_jerk_rad, robust_threshold, 1e-6)
                local_gate = np.clip(
                    (base_magnitude - local_threshold)
                    / max(0.5 * local_threshold, 1e-6),
                    0.0,
                    1.0,
                )
                local_weight = rotation_prior_weight * local_gate
                base_velocity, strong_before, strong_after = _smooth_velocity(
                    base_velocity,
                    rotation_strength,
                    local_weight,
                    robust_weight=False,
                )
                refined_rotations = [base_rotation[0]]
                for increment in base_velocity:
                    refined_rotations.append(
                        refined_rotations[-1] * Rotation.from_rotvec(increment)
                    )
                refined_rotation = Rotation.concatenate(refined_rotations)
                rot_before = weak_before
                rot_after = strong_after
                rotation_mode = "local_hybrid"
                local_gate_mean = float(local_gate.mean()) if local_gate.size else 0.0
                local_gate_max = float(local_gate.max()) if local_gate.size else 0.0
            elif (
                (not auto_rotation_mode)
                or angular_accel_rms > high_jerk_rad
            ):
                angular_velocity, rot_before, rot_after = _smooth_velocity(
                    angular_velocity, rotation_strength, rotation_prior_weight
                )
                refined_rotations = [original_rotation[0]]
                for increment in angular_velocity:
                    refined_rotations.append(
                        refined_rotations[-1] * Rotation.from_rotvec(increment)
                    )
                refined_rotation = Rotation.concatenate(refined_rotations)
                rotation_mode = "strong_lie"
                local_gate_mean = 0.0
                local_gate_max = 0.0
            else:
                # Smooth, low-angular-acceleration trajectories (notably real
                # handheld motion) are already coherent.  Applying the strong
                # increment prior here suppresses genuine rotation and hurts
                # RPE_rot.  Retain the v45 weak anchored tangent prior.  The
                # optional v49 path first fuses each reliable adjacent ray
                # measurement directly into its SO(3) increment.  This targets
                # the quantity evaluated by RPE without diffusing a local edge
                # through the entire absolute-pose graph.
                low_motion_rotation = original_rotation
                fused_edges = 0
                fused_corrections_deg = []
                if low_motion_ray_fusion:
                    fused_velocity = angular_velocity.copy()
                    observable_velocity = angular_velocity.copy()
                    edge_cap = max(1e-6, float(os.environ.get(
                        "POINT3R_LC_ROT_GRAPH_EDGE_WEIGHT", "5.0"
                    )))

                    def _edge_rotation_matrix(edge):
                        if edge is None or len(edge) < 1 or edge[0] is None:
                            return None
                        matrix = edge[0][:3, :3]
                        if torch.is_tensor(matrix):
                            matrix = matrix.detach().cpu().numpy()
                        matrix = np.asarray(matrix, dtype=np.float64)
                        return matrix if matrix.shape == (3, 3) else None

                    # A high inlier/Fisher score can still carry systematic
                    # orientation bias.  Evaluate every lag-2/3 measurement
                    # against the product of its lag-1 edges and distribute
                    # that cycle score to the involved adjacent increments.
                    cycle_score_sum = np.zeros(
                        len(valid_frames) - 1, dtype=np.float64
                    )
                    cycle_score_count = np.zeros(
                        len(valid_frames) - 1, dtype=np.float64
                    )
                    frame_order = {
                        frame_i: order_i
                        for order_i, frame_i in enumerate(valid_frames)
                    }
                    if cycle_observable_fusion:
                        for (cycle_current, cycle_target), direct_edge in (
                            self._lc_rotation_edges.items()
                        ):
                            lag = int(cycle_current) - int(cycle_target)
                            if lag < 2 or lag > 3:
                                continue
                            if (
                                cycle_current not in frame_order
                                or cycle_target not in frame_order
                            ):
                                continue
                            direct_matrix = _edge_rotation_matrix(direct_edge)
                            if direct_matrix is None:
                                continue
                            composed = np.eye(3, dtype=np.float64)
                            chain_indices = []
                            valid_cycle = True
                            for chain_frame in range(
                                int(cycle_target) + 1,
                                int(cycle_current) + 1,
                            ):
                                adjacent = self._lc_rotation_edges.get(
                                    (chain_frame, chain_frame - 1)
                                )
                                adjacent_matrix = _edge_rotation_matrix(adjacent)
                                if adjacent_matrix is None:
                                    valid_cycle = False
                                    break
                                composed = composed @ adjacent_matrix
                                order_i = frame_order.get(chain_frame)
                                if order_i is None or order_i <= 0:
                                    valid_cycle = False
                                    break
                                chain_indices.append(order_i - 1)
                            if not valid_cycle or not chain_indices:
                                continue
                            residual_matrix = direct_matrix.T @ composed
                            residual_deg = float(
                                Rotation.from_matrix(
                                    residual_matrix
                                ).magnitude() * 180.0 / np.pi
                            )
                            if not np.isfinite(residual_deg):
                                continue
                            cycle_score = 1.0 / (
                                1.0
                                + (residual_deg / ray_cycle_delta_deg) ** 2
                            )
                            for chain_index in chain_indices:
                                cycle_score_sum[chain_index] += cycle_score
                                cycle_score_count[chain_index] += 1.0
                    cycle_weight = np.divide(
                        cycle_score_sum,
                        np.maximum(cycle_score_count, 1.0),
                    )
                    cycle_weight = np.where(
                        cycle_score_count > 0.0,
                        np.power(
                            np.clip(cycle_weight, 0.0, 1.0),
                            ray_cycle_power,
                        ),
                        0.0,
                    )

                    for velocity_i in range(len(valid_frames) - 1):
                        frame_i = valid_frames[velocity_i + 1]
                        target_i = valid_frames[velocity_i]
                        edge = self._lc_rotation_edges.get((frame_i, target_i))
                        if edge is None or len(edge) < 2 or edge[1] is None:
                            continue
                        measured, information = edge
                        if measured is None:
                            continue
                        measured_matrix = measured[:3, :3]
                        if torch.is_tensor(measured_matrix):
                            measured_matrix = measured_matrix.detach().cpu().numpy()
                        information_body = information[:3, :3]
                        if torch.is_tensor(information_body):
                            information_body = information_body.detach().cpu().numpy()
                        measured_rotation = Rotation.from_matrix(
                            np.asarray(measured_matrix, dtype=np.float64)
                        )
                        base_increment = Rotation.from_rotvec(
                            fused_velocity[velocity_i]
                        )
                        innovation = (
                            base_increment.inv() * measured_rotation
                        ).as_rotvec()
                        innovation_deg = float(
                            np.linalg.norm(innovation) * 180.0 / np.pi
                        )
                        if not np.isfinite(innovation_deg):
                            continue
                        info_body = np.asarray(
                            information_body, dtype=np.float64
                        )
                        mean_info = max(
                            1e-9, float(np.trace(info_body) / 3.0)
                        )
                        info_body = info_body * (
                            min(edge_cap, mean_info) / mean_info
                        )
                        robust_scale = 1.0 / (
                            1.0
                            + (innovation_deg / ray_fusion_delta_deg) ** 2
                        )
                        info_body = info_body * robust_scale
                        normal = (
                            ray_fusion_prior * np.eye(3, dtype=np.float64)
                            + info_body
                        )
                        try:
                            delta = np.linalg.solve(normal, info_body @ innovation)
                        except np.linalg.LinAlgError:
                            continue
                        delta_norm = float(np.linalg.norm(delta))
                        max_step = ray_fusion_max_step_deg * np.pi / 180.0
                        if max_step > 0.0 and delta_norm > max_step:
                            delta = delta * (max_step / max(delta_norm, 1e-12))
                            delta_norm = max_step
                        if delta_norm <= 1e-8:
                            continue
                        fused_velocity[velocity_i] = (
                            base_increment * Rotation.from_rotvec(delta)
                        ).as_rotvec()
                        if cycle_observable_fusion:
                            observable_delta = delta * cycle_weight[velocity_i]
                            observable_velocity[velocity_i] = (
                                base_increment
                                * Rotation.from_rotvec(observable_delta)
                            ).as_rotvec()
                        fused_edges += 1
                        fused_corrections_deg.append(
                            delta_norm * 180.0 / np.pi
                        )
                    integrated = [original_rotation[0]]
                    for increment in fused_velocity:
                        integrated.append(
                            integrated[-1] * Rotation.from_rotvec(increment)
                        )
                    low_motion_rotation = Rotation.concatenate(integrated)
                    observable_rotation = None
                    if cycle_observable_fusion:
                        observable_integrated = [original_rotation[0]]
                        for increment in observable_velocity:
                            observable_integrated.append(
                                observable_integrated[-1]
                                * Rotation.from_rotvec(increment)
                            )
                        observable_rotation = Rotation.concatenate(
                            observable_integrated
                        )

                fused_reference = low_motion_rotation[0]
                fused_tangent = (
                    fused_reference.inv() * low_motion_rotation
                ).as_rotvec()
                fused_tangent, rot_before, rot_after = _smooth(
                    fused_tangent, low_rotation_strength
                )
                fused_refined_rotation = (
                    fused_reference * Rotation.from_rotvec(fused_tangent)
                )
                refined_rotation = fused_refined_rotation
                if low_motion_ray_fusion and ray_fusion_transport_only:
                    # The ray increment is useful for transporting the
                    # body-frame translation chain, but v49 shows that writing
                    # it out as the final camera orientation adds systematic
                    # RPE rotation bias.  Keep it latent and restore v47's weak
                    # network-anchored rotation for the observable pose.
                    translation_rotation = fused_refined_rotation
                    reference = original_rotation[0]
                    tangent = (
                        reference.inv() * original_rotation
                    ).as_rotvec()
                    tangent, rot_before, rot_after = _smooth(
                        tangent, low_rotation_strength
                    )
                    refined_rotation = (
                        reference * Rotation.from_rotvec(tangent)
                    )
                    if cycle_observable_fusion and observable_rotation is not None:
                        observable_reference = observable_rotation[0]
                        observable_tangent = (
                            observable_reference.inv() * observable_rotation
                        ).as_rotvec()
                        observable_tangent, rot_before, rot_after = _smooth(
                            observable_tangent, low_rotation_strength
                        )
                        refined_rotation = (
                            observable_reference
                            * Rotation.from_rotvec(observable_tangent)
                        )
                rotation_mode = (
                    "weak_cycle_fused"
                    if (
                        low_motion_ray_fusion
                        and ray_fusion_transport_only
                        and cycle_observable_fusion
                    )
                    else
                    "weak_ray_transport"
                    if low_motion_ray_fusion and ray_fusion_transport_only
                    else "weak_ray_fused"
                    if low_motion_ray_fusion
                    else "weak_tangent"
                )
                local_gate_mean = 0.0
                local_gate_max = 0.0
                if low_motion_ray_fusion:
                    self._lc_emit(
                        f"[LC_RAY_INCREMENT] frames={len(valid_frames)} "
                        f"accepted={fused_edges} "
                        f"mean_deg={np.mean(fused_corrections_deg) if fused_corrections_deg else 0.0:.6f} "
                        f"max_deg={max(fused_corrections_deg, default=0.0):.6f} "
                        f"prior={ray_fusion_prior:.4f} "
                        f"delta_deg={ray_fusion_delta_deg:.4f} "
                        f"max_step_deg={ray_fusion_max_step_deg:.4f} "
                        f"transport_only={int(ray_fusion_transport_only)} "
                        f"cycle_observable={int(cycle_observable_fusion)} "
                        f"cycle_support={int(np.count_nonzero(cycle_score_count))} "
                        f"cycle_mean={float(cycle_weight.mean()) if cycle_weight.size else 0.0:.6f}"
                    )

            # Translation is regularized in the local camera frame, then
            # integrated using the refined orientation.  This is invariant to
            # the arbitrary world-axis choice and couples the SE(3) chain
            # without trusting noisy point-map translation edges.
            world_velocity = np.diff(positions, axis=0)
            body_velocity = original_rotation[:-1].inv().apply(world_velocity)
            body_velocity, trans_before, trans_after = _smooth_velocity(
                body_velocity, translation_strength
            )
            if translation_rotation is None:
                translation_rotation = refined_rotation
            refined_world_velocity = translation_rotation[:-1].apply(
                body_velocity
            )
            positions = np.vstack((
                positions[0],
                positions[0] + np.cumsum(refined_world_velocity, axis=0),
            ))
            matrices = refined_rotation.as_matrix()
            ray_quality_mean = float(edge_quality.mean()) if edge_quality.size else 0.0
        else:
            positions, trans_before, trans_after = _smooth(
                positions, translation_strength
            )
            rotation = Rotation.from_matrix(matrices)
            reference = rotation[0]
            tangent = (reference.inv() * rotation).as_rotvec()
            tangent, rot_before, rot_after = _smooth(
                tangent, rotation_strength
            )
            matrices = (reference * Rotation.from_rotvec(tangent)).as_matrix()
            ray_quality_mean = 0.0
            local_gate_mean = 0.0
            local_gate_max = 0.0
        for index, frame_i in enumerate(valid_frames):
            template = corrections[frame_i]
            refined = template.detach().cpu().float().clone()
            refined[:3, :3] = torch.from_numpy(matrices[index]).float()
            refined[:3, 3] = torch.from_numpy(positions[index]).float()
            refined = refined.to(device=template.device, dtype=template.dtype)
            corrections[frame_i] = refined
            if frame_i < len(self._pose_trajectory):
                self._pose_trajectory[frame_i] = refined
        self._lc_emit(
            f"[LC_STATE_REG] frames={len(valid_frames)} "
            f"trans_strength={translation_strength:.4f} "
            f"rot_strength={rotation_strength:.4f} "
            f"lie_increment={int(lie_increment)} "
            f"rot_mode={rotation_mode} "
            f"raw_rot_accel={angular_accel_rms:.6f} "
            f"rot_switch={high_jerk_rad:.6f} "
            f"local_gate={local_gate_mean:.6f}/{local_gate_max:.6f} "
            f"ray_power={ray_uncertainty_power:.4f} "
            f"ray_quality={ray_quality_mean:.6f} "
            f"trans_accel={trans_before:.6f}->{trans_after:.6f} "
            f"rot_accel={rot_before:.6f}->{rot_after:.6f}"
        )
        return corrections

    def _lc_transform_and_rehash_bank(self, bank, poses_before, corrections):
        """Move persistent pointers with corrected owning-frame poses."""
        if bank is None or not corrections:
            return {"tokens": 0, "moved": 0}
        moved_total = 0
        token_total = 0
        num_slots = max(1, int(self.kway_num_slots))
        num_buckets = int(self._ordered_num_buckets())
        for j in range(len(bank["feat"])):
            pos = bank["pos"][j]
            times = bank["time"][j]
            if pos is None or times is None or pos.shape[0] != times.shape[0]:
                continue
            pos = pos.clone()
            rays = bank["ray"][j]
            rays = None if rays is None else rays.clone()
            token_total += int(pos.shape[0])
            for frame_value in torch.unique(times).tolist():
                frame_id = int(frame_value)
                if (
                    frame_id < 0
                    or frame_id >= len(poses_before)
                    or frame_id not in corrections
                    or poses_before[frame_id] is None
                ):
                    continue
                mask = times == frame_id
                if not mask.any():
                    continue
                before = poses_before[frame_id].to(pos.device).float()
                after = corrections[frame_id].to(pos.device).float()
                delta = after @ torch.linalg.inv(before)
                pos[mask] = (
                    (delta[:3, :3] @ pos[mask].float().T).T
                    + delta[:3, 3].unsqueeze(0)
                ).to(pos.dtype)
                if rays is not None and rays.shape[0] == pos.shape[0]:
                    rays[mask] = F.normalize(
                        (delta[:3, :3] @ rays[mask].float().T).T,
                        dim=-1,
                    ).to(rays.dtype)
                moved_total += int(mask.sum().item())
            table = torch.full(
                (num_buckets, num_slots), -1,
                dtype=torch.long, device=pos.device,
            )
            count = torch.zeros(num_buckets, dtype=torch.long, device=pos.device)
            if pos.shape[0] > 0:
                keys = self._ordered_pack_bins(
                    self._ordered_spatial_bins(pos)
                ).long()
                sorted_keys, order = torch.sort(keys)
                unique_keys, counts = torch.unique_consecutive(
                    sorted_keys, return_counts=True
                )
                starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
                ranks = (
                    torch.arange(order.numel(), device=pos.device)
                    - torch.repeat_interleave(starts, counts)
                )
                keep = ranks < num_slots
                table[sorted_keys[keep], ranks[keep]] = order[keep]
                count[unique_keys] = torch.minimum(
                    counts, counts.new_full((), num_slots)
                )
            bank["pos"][j] = pos
            bank["ray"][j] = rays
            bank["table"][j] = table
            bank["count"][j] = count
        self._lc_emit(
            f"[LC_MEMORY_REHASH] tokens={token_total} moved={moved_total}"
        )
        return {"tokens": token_total, "moved": moved_total}

    def _forward_addmemory_ordered_kway(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        conf_i=None,
        local_pts3d=None,
        ):
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)

        img_local = None
        if local_pts3d is not None:
            img_local = local_pts3d.permute(0, 3, 1, 2)
            img_local = img_local.unfold(2, 16, 16)
            img_local = img_local.unfold(3, 16, 16)
            img_local = img_local.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)

        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key + feat_pts).float()

        conf_weight = None
        cgmc_enabled = os.environ.get("POINT3R_CGMC", "1").lower() not in ("0", "false", "no", "off", "")
        cgmc_weighted_merge = os.environ.get("POINT3R_CGMC_WEIGHTED_MERGE", "1").lower() not in ("0", "false", "no", "off", "")
        if cgmc_enabled and conf_i is not None:
            conf_map = conf_i.detach()
            if conf_map.ndim == 4 and conf_map.shape[-1] == 1:
                conf_map = conf_map[..., 0]
            elif conf_map.ndim == 4 and conf_map.shape[1] == 1:
                conf_map = conf_map[:, 0]
            if conf_map.ndim == 3:
                conf_map = torch.nan_to_num(conf_map.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                if tuple(conf_map.shape[-2:]) != (img_h, img_w):
                    conf_map = F.interpolate(conf_map[:, None], size=(img_h, img_w), mode="bilinear", align_corners=False)[:, 0]
                conf_patch = conf_map.unfold(1, 16, 16).unfold(2, 16, 16)
                conf_weight = conf_patch.reshape(bs, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).reshape(bs, -1)

        if str(self.ordered_update_impl).lower() not in ("tensor", "gpu", "tensor_gpu"):
            raise RuntimeError("point3r_ordered_kway_clean requires POINT3R_ORDERED_UPDATE_IMPL=tensor")
        budget_evict = os.environ.get("POINT3R_ORDERED_BUDGET_EVICT")
        if budget_evict and budget_evict.lower() not in ("0", "false", "no", "off", ""):
            raise RuntimeError("point3r_ordered_kway_clean is pure ordered k-way; unset POINT3R_ORDERED_BUDGET_EVICT")

        if self._ordered_slot_tables is None or len(self._ordered_slot_tables) != bs:
            self._ordered_slot_tables = [None for _ in range(bs)]
            self._ordered_slot_counts = [None for _ in range(bs)]
            self._ordered_slot_confs = [None for _ in range(bs)]
        elif self._ordered_slot_counts is None or len(self._ordered_slot_counts) != bs:
            self._ordered_slot_counts = [None for _ in range(bs)]
        if self._ordered_slot_confs is None or len(self._ordered_slot_confs) != bs:
            self._ordered_slot_confs = [None for _ in range(bs)]
        if self._ordered_slot_rays is None or len(self._ordered_slot_rays) != bs:
            self._ordered_slot_rays = [None for _ in range(bs)]
        if self._ordered_slot_times is None or len(self._ordered_slot_times) != bs:
            self._ordered_slot_times = [None for _ in range(bs)]
        if self._ordered_slot_locals is None or len(self._ordered_slot_locals) != bs:
            self._ordered_slot_locals = [None for _ in range(bs)]
        if (
            self._static_gate_dynamic_ema is None
            or len(self._static_gate_dynamic_ema) != bs
        ):
            self._static_gate_dynamic_ema = [0.0 for _ in range(bs)]
        if (
            self._static_gate_dynamic_latched is None
            or len(self._static_gate_dynamic_latched) != bs
        ):
            self._static_gate_dynamic_latched = [False for _ in range(bs)]

        dual_bank_enabled = os.environ.get(
            "POINT3R_RAY_DUAL_BANK", "0"
        ).lower() in ("1", "true", "yes", "on")
        if dual_bank_enabled and (
            self._dual_ray_bank is None
            or len(self._dual_ray_bank["feat"]) != bs
        ):
            def _new_bank():
                return {
                    "feat": [None for _ in range(bs)],
                    "pos": [None for _ in range(bs)],
                    "table": [None for _ in range(bs)],
                    "count": [None for _ in range(bs)],
                    "conf": [None for _ in range(bs)],
                    "ray": [None for _ in range(bs)],
                    "time": [None for _ in range(bs)],
                    "local": [None for _ in range(bs)],
                }
            self._dual_ray_bank = _new_bank()
            self._dual_stable_bank = _new_bank()

        memory_feat_list = []
        memory_pos_list = []
        batch_stats = []
        for j in range(bs):
            if i == 0 or memory_pos is None:
                prev_feat = memory_add[j].new_empty((0, memory_add.shape[-1]))
                prev_pos = img_pos[j].new_empty((0, 3))
                slot_table_j = None
                slot_count_j = None
                slot_conf_j = None
                slot_ray_j = prev_pos.new_empty((0, 3)).float()
                slot_time_j = prev_pos.new_empty((0,), dtype=torch.long)
                slot_local_j = prev_pos.new_empty((0, 3)).float()
            else:
                prev_feat = memory_feat[j]
                prev_pos = memory_pos[j]
                slot_table_j = self._ordered_slot_tables[j]
                slot_count_j = self._ordered_slot_counts[j]
                slot_conf_j = self._ordered_slot_confs[j]
                slot_ray_j = self._ordered_slot_rays[j]
                slot_time_j = self._ordered_slot_times[j]
                slot_local_j = self._ordered_slot_locals[j]

            write_feat_j = memory_add[j]
            write_pos_j = img_pos[j]
            write_local_j = img_local[j] if img_local is not None else None
            write_weight_j = conf_weight[j] if conf_weight is not None else None
            if (
                os.environ.get("POINT3R_POSE_STATIC_REFINE", "0").lower()
                in ("1", "true", "yes", "on")
                and i > 0
                and write_local_j is not None
                and i < len(self._pose_trajectory)
                and self._pose_trajectory[i] is not None
            ):
                pose_memory_feat = prev_feat
                pose_memory_pos = prev_pos
                pose_memory_conf = slot_conf_j
                if (
                    self._dual_stable_bank is not None
                    and j < len(self._dual_stable_bank["feat"])
                    and self._dual_stable_bank["feat"][j] is not None
                ):
                    pose_memory_feat = self._dual_stable_bank["feat"][j]
                    pose_memory_pos = self._dual_stable_bank["pos"][j]
                    pose_memory_conf = self._dual_stable_bank["conf"][j]
                refined_pose = self._pose_static_memory_refine(
                    i,
                    write_pos_j,
                    write_local_j,
                    write_feat_j,
                    write_weight_j,
                    self._pose_trajectory[i],
                    pose_memory_pos,
                    pose_memory_feat,
                    pose_memory_conf,
                )
                if refined_pose is not None:
                    refined_pose = refined_pose.to(write_local_j.device)
                    self._pose_trajectory[i] = refined_pose.detach()
                    corrected_dense = (
                        torch.einsum(
                            "ij,nj->ni",
                            refined_pose[:3, :3].to(img_local[j].dtype),
                            img_local[j],
                        )
                        + refined_pose[:3, 3].to(img_local[j].dtype).unsqueeze(0)
                    )
                    img_pos[j] = corrected_dense.to(img_pos[j].dtype)
                    write_pos_j = img_pos[j]
                    self._online_pose_refinements[i] = refined_pose.detach()
            static_memory_keep = None
            scene_dynamic_mode = False
            cgmc_seen = int(write_pos_j.shape[0])
            cgmc_kept = cgmc_seen
            cgmc_thresh = -1.0
            cgmc_conf_mean = -1.0
            cgmc_conf_min = -1.0
            cgmc_conf_max = -1.0
            if write_weight_j is not None and write_weight_j.numel() == write_pos_j.shape[0]:
                drop_q = float(os.environ.get("POINT3R_CGMC_DROP_QUANTILE", "0.35"))
                min_conf = float(os.environ.get("POINT3R_CGMC_MIN_CONF", "0.0"))
                drop_q = max(0.0, min(0.95, drop_q))
                conf_vals = torch.nan_to_num(write_weight_j.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                if drop_q > 0.0 and conf_vals.numel() > 1:
                    threshold = torch.maximum(torch.quantile(conf_vals, drop_q), conf_vals.new_tensor(min_conf))
                else:
                    threshold = conf_vals.new_tensor(min_conf)
                keep = conf_vals >= threshold
                write_feat_j = write_feat_j[keep]
                write_pos_j = write_pos_j[keep]
                if write_local_j is not None:
                    write_local_j = write_local_j[keep]
                write_weight_j = conf_vals[keep]
                cgmc_kept = int(write_pos_j.shape[0])
                # Fast timing path: scalar diagnostics stay disabled to avoid
                # synchronizing CUDA with the host on every frame.

            # Training-free temporal static-consistency gate.  A mutual
            # feature match to the immediately previous frame that moves too
            # far in world coordinates is likely dynamic foreground.  Audit
            # and filtering are independently switchable so the threshold can
            # be validated without changing predictions.
            static_gate = os.environ.get(
                "POINT3R_STATIC_GATE", "0"
            ).lower() in ("1", "true", "yes", "on")
            static_audit = os.environ.get(
                "POINT3R_STATIC_GATE_AUDIT", "0"
            ).lower() in ("1", "true", "yes", "on")
            ray_scene_adaptive = os.environ.get(
                "POINT3R_RAY_SCENE_ADAPTIVE", "0"
            ).lower() in ("1", "true", "yes", "on")
            if (
                (static_gate or static_audit or ray_scene_adaptive)
                and i > 0
                and self._lc_feat is not None
                and j < len(self._lc_feat)
                and self._lc_feat[j] is not None
                and write_pos_j.shape[0] > 1
            ):
                previous_mask = self._lc_fid[j] == (i - 1)
                if previous_mask.any():
                    previous_feat = F.normalize(
                        self._lc_feat[j][previous_mask].float(), dim=-1
                    )
                    previous_pos = self._lc_pos[j][previous_mask].float()
                    current_feat = F.normalize(write_feat_j.float(), dim=-1)
                    feature_sim = current_feat @ previous_feat.T
                    best_sim_static, current_to_previous = feature_sim.max(dim=1)
                    previous_to_current = feature_sim.argmax(dim=0)
                    current_ids = torch.arange(
                        write_pos_j.shape[0], device=write_pos_j.device
                    )
                    mutual_static = (
                        previous_to_current[current_to_previous] == current_ids
                    )
                    matched_previous_pos = previous_pos[current_to_previous]
                    current_world_pos = write_pos_j.float()
                    world_motion = torch.norm(
                        current_world_pos - matched_previous_pos, dim=-1
                    )

                    max_sample = min(256, int(write_pos_j.shape[0]))
                    sample_ids = torch.linspace(
                        0, write_pos_j.shape[0] - 1, max_sample,
                        device=write_pos_j.device,
                    ).round().long()
                    sample_pos = write_pos_j[sample_ids].float()
                    spacing = torch.cdist(sample_pos, sample_pos)
                    spacing.fill_diagonal_(float("inf"))
                    median_spacing = spacing.min(dim=1).values.median()
                    motion_factor = float(os.environ.get(
                        "POINT3R_STATIC_GATE_NN_MULT", "0.75"
                    ))
                    motion_min = float(os.environ.get(
                        "POINT3R_STATIC_GATE_MIN_DIST", "0.05"
                    ))
                    motion_threshold = torch.maximum(
                        median_spacing * motion_factor,
                        median_spacing.new_tensor(motion_min),
                    )
                    feature_threshold = float(os.environ.get(
                        "POINT3R_STATIC_GATE_FEAT_SIM", "0.70"
                    ))
                    # Remove the dominant inter-frame rigid motion before
                    # classifying dynamic tokens.  Raw world-coordinate
                    # differences also contain coherent reconstruction/pose
                    # drift and otherwise reject many static points.  Fit a
                    # robust current->previous SE(3) transform on mutual,
                    # feature-consistent matches, trim the largest residuals,
                    # and refit once.
                    rigid_fit_mask = mutual_static & (
                        best_sim_static >= feature_threshold
                    )
                    rigid_residual = world_motion
                    if int(rigid_fit_mask.sum().item()) >= 12:
                        fit_ids = torch.where(rigid_fit_mask)[0]
                        for _ in range(2):
                            src_fit = current_world_pos[fit_ids]
                            dst_fit = matched_previous_pos[fit_ids]
                            src_center = src_fit.mean(dim=0)
                            dst_center = dst_fit.mean(dim=0)
                            covariance = (
                                (src_fit - src_center).T
                                @ (dst_fit - dst_center)
                            )
                            u_fit, _, vh_fit = torch.linalg.svd(covariance)
                            rotation_fit = vh_fit.T @ u_fit.T
                            if torch.det(rotation_fit) < 0:
                                vh_fit = vh_fit.clone()
                                vh_fit[-1] *= -1
                                rotation_fit = vh_fit.T @ u_fit.T
                            aligned_current = (
                                (current_world_pos - src_center)
                                @ rotation_fit.T
                                + dst_center
                            )
                            rigid_residual = torch.norm(
                                aligned_current - matched_previous_pos, dim=-1
                            )
                            if fit_ids.numel() >= 20:
                                trim_threshold = torch.quantile(
                                    rigid_residual[fit_ids], 0.70
                                )
                                trimmed = fit_ids[
                                    rigid_residual[fit_ids] <= trim_threshold
                                ]
                                if trimmed.numel() >= 12:
                                    fit_ids = trimmed
                    dynamic_mask = rigid_fit_mask & (
                        rigid_residual > motion_threshold
                    )
                    dynamic_ratio = float(dynamic_mask.sum().item()) / float(
                        max(1, int(rigid_fit_mask.sum().item()))
                    )
                    ema_alpha = max(
                        0.0,
                        min(
                            1.0,
                            float(os.environ.get(
                                "POINT3R_STATIC_GATE_EMA_ALPHA", "0.20"
                            )),
                        ),
                    )
                    if i <= 1:
                        dynamic_ema = dynamic_ratio
                    else:
                        dynamic_ema = (
                            (1.0 - ema_alpha)
                            * self._static_gate_dynamic_ema[j]
                            + ema_alpha * dynamic_ratio
                        )
                    self._static_gate_dynamic_ema[j] = dynamic_ema
                    scene_adaptive = os.environ.get(
                        "POINT3R_STATIC_GATE_SCENE_ADAPTIVE", "0"
                    ).lower() in ("1", "true", "yes", "on")
                    adaptive_warmup = int(os.environ.get(
                        "POINT3R_STATIC_GATE_WARMUP", "10"
                    ))
                    adaptive_threshold = float(os.environ.get(
                        "POINT3R_STATIC_GATE_SCENE_THRESHOLD", "0.45"
                    ))
                    adaptive_active = (
                        (not scene_adaptive)
                        or (
                            i >= adaptive_warmup
                            and dynamic_ema >= adaptive_threshold
                        )
                    )
                    sticky_policy = os.environ.get(
                        "POINT3R_RAY_SCENE_STICKY", "0"
                    ).lower() in ("1", "true", "yes", "on")
                    if sticky_policy and adaptive_active:
                        self._static_gate_dynamic_latched[j] = True
                    if sticky_policy:
                        adaptive_active = self._static_gate_dynamic_latched[j]
                    scene_dynamic_mode = ray_scene_adaptive and adaptive_active
                    if static_audit and (i % 10 == 0):
                        raw_dynamic = rigid_fit_mask & (
                            world_motion > motion_threshold
                        )
                        self._lc_emit(
                            f"[STATIC_GATE] frame={i} tokens={write_pos_j.shape[0]} "
                            f"mutual={int(mutual_static.sum().item())} "
                            f"raw_dynamic={int(raw_dynamic.sum().item())} "
                            f"dynamic={int(dynamic_mask.sum().item())} "
                            f"ratio={dynamic_ratio:.6f} ema={dynamic_ema:.6f} "
                            f"active={int(adaptive_active)} "
                            f"spacing={float(median_spacing.item()):.6f} "
                            f"threshold={float(motion_threshold.item()):.6f} "
                            f"residual_p50={float(rigid_residual[rigid_fit_mask].median().item()):.6f} "
                            f"residual_p90={float(torch.quantile(rigid_residual[rigid_fit_mask], 0.90).item()):.6f}"
                        )
                    if static_gate and adaptive_active and dynamic_mask.any():
                        # Do not remove these observations from short-range
                        # odometry: even a noisy/dynamic frame can still carry
                        # useful static correspondences for the robust SO(3)
                        # estimator.  Apply the veto only at the persistent
                        # memory-write boundary below.
                        static_memory_keep = ~dynamic_mask

            # Ray metadata is used by both ray-aware memory updates and optional loop closure.
            lc_enabled = os.environ.get("POINT3R_LC_ENABLED", "0").lower() in ("1","true","yes","on")
            rayaware_enabled = os.environ.get("POINT3R_RAYAWARE_UPDATE", "0").lower() in ("1","true","yes","on")
            write_ray_j = None
            c2w_i = self._pose_trajectory[i] if i < len(self._pose_trajectory) else None
            ray_adaptive_strength = None
            motion_gate = rayaware_enabled and os.environ.get(
                "POINT3R_RAY_MOTION_GATE", "0"
            ).lower() in ("1", "true", "yes", "on")
            if motion_gate and c2w_i is not None and i > 0 and self._pose_trajectory[i - 1] is not None:
                prev_c2w = self._pose_trajectory[i - 1].to(c2w_i.device)
                trans_delta = torch.norm(c2w_i[:3, 3] - prev_c2w[:3, 3])
                motion_sigma = max(float(os.environ.get("POINT3R_RAY_MOTION_SIGMA", "0.025")), 1e-6)
                # strength=1 applies the ray gate; strength=0 recovers the
                # original confidence-weighted averaging rule.
                ray_adaptive_strength = torch.exp(-0.5 * (trans_delta / motion_sigma) ** 2)
                if j == 0 and os.environ.get("POINT3R_RAY_MOTION_AUDIT", "0").lower() in ("1", "true", "yes", "on"):
                    self._lc_emit(
                        f"[RAY_MOTION] frame={i} trans={float(trans_delta.detach().cpu()):.8f} "
                        f"strength={float(ray_adaptive_strength.detach().cpu()):.6f}"
                    )
            if (lc_enabled or rayaware_enabled) and write_pos_j.shape[0] > 0:
                if i == 0 and j == 0:
                    self._lc_emit(
                        f"[RAY_CONFIG] local_geometry={'pts3d_in_self_view' if write_local_j is not None else 'pose_inverse_fallback'} "
                        f"rayaware_update={int(rayaware_enabled)} loop_closure={int(lc_enabled)}"
                    )
                if write_local_j is None:
                    if c2w_i is not None:
                        R_i = c2w_i[:3, :3].to(write_pos_j.device)
                        t_i = c2w_i[:3, 3].to(write_pos_j.device)
                        write_local_j = (R_i.T @ (write_pos_j - t_i.unsqueeze(0)).T).T
                    else:
                        write_local_j = write_pos_j
                ray_local_j = F.normalize(write_local_j, dim=-1)
                if c2w_i is not None:
                    R_i = c2w_i[:3, :3].to(write_pos_j.device)
                    write_ray_j = F.normalize((R_i @ ray_local_j.T).T, dim=-1)
                else:
                    write_ray_j = ray_local_j
            if lc_enabled and write_pos_j.shape[0] > 0:
                max_lag = max(1, int(os.environ.get("POINT3R_LC_ODOM_MAX_LAG", "1")))
                for lag in range(1, min(max_lag, i) + 1):
                    target_frame = i - lag
                    decoupled_se3 = os.environ.get(
                        "POINT3R_LC_DECOUPLED_SE3", "0"
                    ).lower() in ("1", "true", "yes", "on")
                    if decoupled_se3:
                        translation_edge = self._lc_estimate_adjacent_rotation(
                            j, write_pos_j, write_local_j, write_feat_j, i,
                            write_pos_j.device, target_frame=target_frame,
                            force_rotation=False,
                        )
                        if translation_edge is not None:
                            self._lc_translation_edges[(i, target_frame)] = translation_edge
                        odom_edge = self._lc_estimate_adjacent_rotation(
                            j, write_pos_j, write_local_j, write_feat_j, i,
                            write_pos_j.device, target_frame=target_frame,
                            force_rotation=True,
                        )
                    else:
                        odom_edge = self._lc_estimate_adjacent_rotation(
                            j, write_pos_j, write_local_j, write_feat_j, i,
                            write_pos_j.device, target_frame=target_frame,
                        )
                    if odom_edge is not None:
                        cycle_enabled = os.environ.get(
                            "POINT3R_LC_ODOM_CYCLE_ENABLED", "0"
                        ).lower() in ("1", "true", "yes", "on")
                        if lag > 1 and cycle_enabled:
                            composed = torch.eye(3, dtype=torch.float32)
                            chain_complete = True
                            for chain_frame in range(target_frame + 1, i + 1):
                                chain_edge = self._lc_rotation_edges.get(
                                    (chain_frame, chain_frame - 1)
                                )
                                if chain_edge is None:
                                    chain_complete = False
                                    break
                                composed = composed @ chain_edge[0][:3, :3].float()
                            if chain_complete:
                                measured_rot = odom_edge[0][:3, :3].float()
                                cycle_delta = measured_rot @ composed.T
                                cosine = ((torch.trace(cycle_delta) - 1.0) * 0.5).clamp(-1.0, 1.0)
                                cycle_deg = float(torch.rad2deg(torch.acos(cosine)).item())
                                cycle_scale = max(
                                    1e-3, float(os.environ.get("POINT3R_LC_ODOM_CYCLE_DELTA_DEG", "2.0"))
                                )
                                cycle_weight = 1.0 / (1.0 + (cycle_deg / cycle_scale) ** 2)
                                cycle_max = float(os.environ.get("POINT3R_LC_ODOM_CYCLE_MAX_DEG", "5.0"))
                                if cycle_deg > cycle_max:
                                    cycle_weight = 0.0
                                odom_edge[1][:3, :3] *= cycle_weight
                                self._lc_emit(
                                    f"[LC_CYCLE] frame={i} target={target_frame} lag={lag} "
                                    f"residual_deg={cycle_deg:.4f} weight={cycle_weight:.4f}"
                                )
                        self._lc_rotation_edges[(i, target_frame)] = odom_edge
                        if lag == 1:
                            self._lc_odometry_edges[i] = odom_edge
                # detect 在 update 前执行，保证来源帧信息未被 merge 破坏
                loop_detect = os.environ.get(
                    "POINT3R_LC_LOOP_DETECTION", "1"
                ).lower() in ("1", "true", "yes", "on")
                new_cands = self._lc_detect_and_estimate(
                    j, write_pos_j, write_local_j, write_ray_j, write_feat_j,
                    i, c2w_i, write_pos_j.device) if loop_detect else []
                if new_cands:
                    self._lc_candidates.extend(new_cands)
                    min_loop_weight = float(os.environ.get("POINT3R_LC_LOOP_MIN_ROT_WEIGHT", "0.10"))
                    decoupled_se3 = os.environ.get(
                        "POINT3R_LC_DECOUPLED_SE3", "0"
                    ).lower() in ("1", "true", "yes", "on")
                    for loop_current, loop_target, loop_transform, loop_info in new_cands:
                        loop_weight = float(loop_info[:3, :3].diagonal().mean())
                        if loop_weight >= min_loop_weight:
                            self._lc_rotation_edges[(loop_current, loop_target)] = (
                                loop_transform, loop_info
                            )
                            if decoupled_se3:
                                self._lc_translation_edges[(loop_current, loop_target)] = (
                                    loop_transform, loop_info
                                )
                    self._lc_emit(f"[LC] frame={i} batch={j}: +{len(new_cands)} loop edges "
                                  f"total={len(self._lc_candidates)} rotation_total={len(self._lc_rotation_edges)}")
                self._lc_update_index(
                    j, write_pos_j, write_local_j, write_ray_j, write_feat_j,
                    i, c2w_i, write_pos_j.device)

            if static_memory_keep is not None:
                write_feat_j = write_feat_j[static_memory_keep]
                write_pos_j = write_pos_j[static_memory_keep]
                if write_local_j is not None:
                    write_local_j = write_local_j[static_memory_keep]
                if write_ray_j is not None:
                    write_ray_j = write_ray_j[static_memory_keep]
                if write_weight_j is not None:
                    write_weight_j = write_weight_j[static_memory_keep]
                cgmc_kept = int(write_pos_j.shape[0])

            if dual_bank_enabled:
                def _bank_previous(bank):
                    if bank["feat"][j] is None:
                        return (
                            memory_add[j].new_empty((0, memory_add.shape[-1])),
                            img_pos[j].new_empty((0, 3)),
                            None, None, None,
                            img_pos[j].new_empty((0, 3)).float(),
                            img_pos[j].new_empty((0,), dtype=torch.long),
                            img_pos[j].new_empty((0, 3)).float(),
                        )
                    return (
                        bank["feat"][j], bank["pos"][j], bank["table"][j],
                        bank["count"][j], bank["conf"][j], bank["ray"][j],
                        bank["time"][j], bank["local"][j],
                    )

                ray_previous = _bank_previous(self._dual_ray_bank)
                # The stable geometry stream must continue from the actual
                # recurrent memory handed in by Point3R.  Starting a second
                # empty "stable" bank silently drops the initialized memory on
                # frame zero and is not equivalent to the original model.
                stable_previous = (
                    prev_feat,
                    prev_pos,
                    slot_table_j,
                    slot_count_j,
                    slot_conf_j,
                    slot_ray_j,
                    slot_time_j,
                    slot_local_j,
                )
                # The auxiliary RayAway-style retain/replace policy samples
                # random coins.  Keep those samples inside a forked RNG scope
                # so enabling the pose-only bank cannot perturb either the
                # stable geometry stream or downstream metric sampling.
                ray_rng_devices = (
                    [write_pos_j.device.index]
                    if write_pos_j.is_cuda
                    else []
                )
                ray_bank_update_every = max(1, int(os.environ.get(
                    "POINT3R_RAY_BANK_UPDATE_EVERY", "1"
                )))
                update_ray_bank = (
                    ray_previous[0].shape[0] == 0
                    or (i % ray_bank_update_every) == 0
                )
                if update_ray_bank:
                    with torch.random.fork_rng(devices=ray_rng_devices, enabled=True):
                        ray_result = self._ordered_update_single_tensor(
                            ray_previous[0], ray_previous[1], write_feat_j, write_pos_j,
                            ray_previous[2], ray_previous[3], ray_previous[4],
                            new_conf_j=write_weight_j if cgmc_weighted_merge else None,
                            memory_ray_j=ray_previous[5], new_ray_j=write_ray_j,
                            memory_time_j=ray_previous[6], new_time_j=i,
                            memory_local_j=ray_previous[7], new_local_j=write_local_j,
                            ray_adaptive_strength=ray_adaptive_strength,
                            rayaware_override=True,
                        )
                else:
                    ray_result = (*ray_previous, {
                        "_pointer_loop_payload": None,
                        "ray_bank_update_skipped": 1,
                    })
                stable_result = self._ordered_update_single_tensor(
                    stable_previous[0], stable_previous[1], write_feat_j, write_pos_j,
                    stable_previous[2], stable_previous[3], stable_previous[4],
                    new_conf_j=write_weight_j if cgmc_weighted_merge else None,
                    memory_ray_j=stable_previous[5], new_ray_j=write_ray_j,
                    memory_time_j=stable_previous[6], new_time_j=i,
                    memory_local_j=stable_previous[7], new_local_j=write_local_j,
                    ray_adaptive_strength=None,
                    rayaware_override=False,
                )
                pointer_payload = ray_result[8].pop(
                    "_pointer_loop_payload", None
                )
                pointer_edges = self._lc_pointer_loop_edges_from_payload(
                    i, pointer_payload, write_pos_j.device
                )
                if pointer_edges:
                    self._lc_candidates.extend(pointer_edges)
                    for current_frame, target_frame, transform, info in pointer_edges:
                        self._lc_rotation_edges[(current_frame, target_frame)] = (
                            transform, info
                        )
                    if self._lc_diag is None:
                        self._lc_diag = {
                            "frames": 0,
                            "history_frames": 0,
                            "spatial_pairs": 0,
                            "ray_pairs": 0,
                            "mutual_pairs": 0,
                            "verified_edges": 0,
                        }
                    self._lc_diag["verified_edges"] += len(pointer_edges)
                for bank, result in (
                    (self._dual_ray_bank, ray_result),
                    (self._dual_stable_bank, stable_result),
                ):
                    for key, value in zip(
                        ("feat", "pos", "table", "count", "conf", "ray", "time", "local"),
                        result[:8],
                    ):
                        bank[key][j] = value
                pose_only_ensemble = os.environ.get(
                    "POINT3R_RAY_POSE_ONLY_ENSEMBLE", "0"
                ).lower() in ("1", "true", "yes", "on")
                pointer_loop_graph = os.environ.get(
                    "POINT3R_RAY_POINTER_LOOP_GRAPH", "0"
                ).lower() in ("1", "true", "yes", "on")
                pose_input_only = os.environ.get(
                    "POINT3R_RAY_POSE_INPUT_ONLY", "0"
                ).lower() in ("1", "true", "yes", "on")
                post_decoder_only = os.environ.get(
                    "POINT3R_RAY_POSE_POST_DECODER_ONLY", "0"
                ).lower() in ("1", "true", "yes", "on")
                # In pose-only mode the main recurrent stream must remain
                # bit-for-bit on the original stable K-way/ConfSelect policy.
                # The ray bank is consumed only by the pose path and can never
                # become the geometry decoder's recurrent memory.
                active_result = (
                    stable_result
                    if (
                        pose_only_ensemble
                        or pointer_loop_graph
                        or pose_input_only
                        or post_decoder_only
                        or scene_dynamic_mode
                    )
                    else ray_result
                )
                (
                    memory_feat_j, memory_pos_j, slot_table_j, slot_count_j,
                    slot_conf_j, slot_ray_j, slot_time_j, slot_local_j, stats_j,
                ) = active_result
                stats_j["dual_bank"] = 1
                stats_j["active_bank"] = (
                    "stable"
                    if (
                        pose_only_ensemble
                        or pointer_loop_graph
                        or pose_input_only
                        or post_decoder_only
                        or scene_dynamic_mode
                    )
                    else "ray"
                )
                stats_j["ray_bank_memory"] = int(ray_result[1].shape[0])
                stats_j["stable_bank_memory"] = int(stable_result[1].shape[0])
                stats_j["ray_bank_update_every"] = ray_bank_update_every
                stats_j["ray_bank_updated"] = int(update_ray_bank)
            else:
                memory_feat_j, memory_pos_j, slot_table_j, slot_count_j, slot_conf_j, slot_ray_j, slot_time_j, slot_local_j, stats_j = self._ordered_update_single_tensor(
                    prev_feat,
                    prev_pos,
                    write_feat_j,
                    write_pos_j,
                    slot_table_j,
                    slot_count_j,
                    slot_conf_j,
                    new_conf_j=write_weight_j if cgmc_weighted_merge else None,
                    memory_ray_j=slot_ray_j,
                    new_ray_j=write_ray_j,
                    memory_time_j=slot_time_j,
                    new_time_j=i,
                    memory_local_j=slot_local_j,
                    new_local_j=write_local_j,
                    ray_adaptive_strength=ray_adaptive_strength,
                    rayaware_override=(False if scene_dynamic_mode else None),
                )
            stats_j["confwrite_seen"] = cgmc_seen
            stats_j["confwrite_kept"] = cgmc_kept
            stats_j["confwrite_drop"] = cgmc_seen - cgmc_kept
            stats_j["confwrite_conf_mean"] = cgmc_conf_mean
            stats_j["confwrite_conf_min"] = cgmc_conf_min
            stats_j["confwrite_conf_max"] = cgmc_conf_max
            stats_j["confwrite_thresh"] = cgmc_thresh
            self._ordered_slot_tables[j] = slot_table_j
            self._ordered_slot_counts[j] = slot_count_j
            self._ordered_slot_confs[j] = slot_conf_j
            self._ordered_slot_rays[j] = slot_ray_j
            self._ordered_slot_times[j] = slot_time_j
            self._ordered_slot_locals[j] = slot_local_j
            stats_j["frame"] = int(i)
            stats_j["batch"] = int(j)
            stats_j["theta_bins"] = int(self.ordered_theta_bins)
            stats_j["phi_bins"] = int(self.ordered_phi_bins)
            stats_j["rho_bins"] = int(self.ordered_rho_bins)
            if self._last_sparse_readout_stats is not None and j < len(self._last_sparse_readout_stats):
                stats_j.update(self._last_sparse_readout_stats[j])
            batch_stats.append(stats_j)
            memory_feat_list.append(memory_feat_j)
            memory_pos_list.append(memory_pos_j)

        self.memory_update_stats.append(batch_stats)
        init_memory_feat_list = [memory_feat_index.clone().detach() for memory_feat_index in memory_feat_list]
        if i == 0:
            return torch.stack(memory_feat_list, dim=0), torch.stack(memory_pos_list, dim=0), torch.stack(init_memory_feat_list, dim=0), img_pos
        return memory_feat_list, memory_pos_list, init_memory_feat_list, img_pos


    def _forward_addmemory_merge(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        conf_i=None,
        local_pts3d=None,
        ):
        if self.memory_update_mode not in ("ordered_kway", "ordered"):
            raise RuntimeError("point3r_ordered_kway_clean only supports POINT3R_MEMORY_UPDATE_MODE=ordered_kway")
        return self._forward_addmemory_ordered_kway(
            i,
            pts3d=pts3d,
            init_memory_feat=init_memory_feat,
            memory_feat=memory_feat,
            memory_pos=memory_pos,
            feat_i=feat_i,
            dec_i=dec_i,
            shape_i=shape_i,
            conf_i=conf_i,
            local_pts3d=local_pts3d,
        )

    def _forward_merge(self, views, point3r_tag=False):
        self.memory_update_stats = []
        self.sparse_readout_stats = []
        self._last_sparse_readout_stats = None
        self._ordered_slot_tables = None
        self._ordered_slot_counts = None
        self._ordered_slot_confs = None
        self._ordered_slot_rays = None
        self._ordered_slot_times = None
        self._ordered_slot_locals = None
        self._static_gate_dynamic_ema = None
        self._static_gate_dynamic_latched = None
        self._dual_ray_bank = None
        self._dual_stable_bank = None
        self._last_pose_only_dec = None
        self._lc_pos = None
        self._lc_fid = None
        self._lc_local = None
        self._lc_ray = None
        self._lc_feat = None
        self._pose_trajectory = []
        self._lc_candidates = []
        self._lc_odometry_edges = {}
        self._lc_rotation_edges = {}
        self._lc_translation_edges = {}
        self._lc_diag = None
        self._lc_pgo_executed = False
        self._online_pose_refinements = {}
        self._v70_prev_stable_c2w = None
        self._v70_prev_token_c2w = None
        self._v71_base_score_ema = None
        self._v71_token_score_ema = None
        self._v71_use_token = None
        self._v74_relative_gain_ema = None
        self._v74_effect_size_ema = None
        self._v74_observability_ema = None
        self._v74_use_token = None
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        pose_only_ensemble = os.environ.get(
            "POINT3R_RAY_POSE_ONLY_ENSEMBLE", "0"
        ).lower() in ("1", "true", "yes", "on")
        # Keep a genuinely independent recurrent pose state for the ray-aware
        # branch.  v54 reused ``mem`` for both decoders, so the alternative
        # readout was only frame-local and could not accumulate a distinct
        # ray-conditioned motion history.
        ray_mem = mem.clone() if pose_only_ensemble else None
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        merge_tag = False
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if i >= 2:
                merge_tag = True
            if merge_tag:
                memory_len_max = max(f_memory_j.shape[0] for f_memory_j in memory_feat)
                f_memory_list_padded = []
                pos_memory_list_padded = []
                mask_memory_list_padded = []
                for j in range(len(memory_feat)):
                    f_memory_j = memory_feat[j]
                    pos_memory_j = pos_decode_memory[j]
                    padding_size = memory_len_max - f_memory_j.shape[0]
                    padding = torch.zeros(padding_size, f_memory_j.shape[1]).to(f_memory_j.device)
                    padding_pos = torch.zeros(padding_size, pos_memory_j.shape[1]).to(pos_memory_j.device)
                    mask_valid = torch.ones(f_memory_j.shape[0]).to(f_memory_j.device)
                    mask_invalid = torch.zeros(padding_size).to(f_memory_j.device)
                    padded_memory_j = torch.cat((f_memory_j, padding), dim=0)
                    padded_pos_memory_j = torch.cat((pos_memory_j, padding_pos), dim=0)
                    padded_mask_j = torch.cat((mask_valid, mask_invalid), dim=0)
                    f_memory_list_padded.append(padded_memory_j)
                    pos_memory_list_padded.append(padded_pos_memory_j)
                    mask_memory_list_padded.append(padded_mask_j)
                memory_feat = torch.stack(f_memory_list_padded, dim=0)
                pos_decode_memory = torch.stack(pos_memory_list_padded, dim=0)
                mask_memory = torch.stack(mask_memory_list_padded, dim=0)
            else:
                mask_memory = None
            
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                    ray_pose_feat_i = (
                        self.pose_token.expand(feat_i.shape[0], -1, -1)
                        if pose_only_ensemble else None
                    )
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                    ray_pose_feat_i = (
                        self.pose_retriever.inquire(global_img_feat_i, ray_mem)
                        if pose_only_ensemble else None
                    )
                pose_pos_i = None
            else:
                pose_feat_i = None
                ray_pose_feat_i = None
                pose_pos_i = None

            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                pose_pos_i,
                pose_only_feat=ray_pose_feat_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            if pose_only_ensemble:
                if self._last_pose_only_dec is not None:
                    ray_out_pose_feat_i = self._last_pose_only_dec[-1][:, 0:1]
                    new_ray_mem = self.pose_retriever.update_mem(
                        ray_mem, global_img_feat_i, ray_out_pose_feat_i
                    )
                else:
                    # The ray bank is not populated for the first frame.  Use
                    # the identical bootstrap only once; subsequent updates
                    # are driven exclusively by the ray decoder output.
                    new_ray_mem = new_mem.clone()
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]

            token_residual = os.environ.get(
                "POINT3R_RAY_POSE_TOKEN_RESIDUAL", "0"
            ).lower() in ("1", "true", "yes", "on")
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            res = self._apply_geometry_safe_pose_readout(
                res, dec[-1][:, 0:1], pos_decode_img, i
            )
            token_c2w = None
            if self._last_pose_only_dec is not None and 'camera_pose' in res:
                ray_dec = self._last_pose_only_dec
                ray_head_input = [
                    ray_dec[0].float(),
                    ray_dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    ray_dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    ray_dec[self.dec_depth].float(),
                ]
                if token_residual:
                    # Combine only the two pose-bearing decoder tokens. Dense
                    # image/point tokens stay on the verified stable stream.
                    base_tokens = (head_input[0][:, :1], head_input[3][:, :1])
                    ray_tokens = (ray_head_input[0][:, :1], ray_head_input[3][:, :1])
                    cosines = []
                    aligned_ray_tokens = []
                    for base_token, ray_token in zip(base_tokens, ray_tokens):
                        base_f = base_token.float()
                        ray_f = ray_token.float()
                        cosine = F.cosine_similarity(base_f, ray_f, dim=-1).mean(dim=-1)
                        cosines.append(cosine)
                        aligned_ray_tokens.append(
                            F.normalize(ray_f, dim=-1)
                            * torch.linalg.norm(base_f, dim=-1, keepdim=True).clamp_min(1e-6)
                        )
                    latent_cosine = torch.stack(cosines, dim=0).mean(dim=0).clamp(-1.0, 1.0)
                    max_weight = max(0.0, min(0.5, float(os.environ.get(
                        "POINT3R_RAY_POSE_TOKEN_MAX_WEIGHT", "0.25"
                    ))))
                    agreement_power = max(1.0, float(os.environ.get(
                        "POINT3R_RAY_POSE_TOKEN_AGREEMENT_POWER", "4.0"
                    )))
                    token_weight = max_weight * ((latent_cosine + 1.0) * 0.5).pow(
                        agreement_power
                    )
                    mixed_head_input = [value.clone() for value in head_input]
                    for layer_index, aligned_ray in ((0, aligned_ray_tokens[0]), (3, aligned_ray_tokens[1])):
                        base_token = mixed_head_input[layer_index][:, :1]
                        weight_view = token_weight[:, None, None].to(base_token.dtype)
                        mixed_head_input[layer_index][:, :1] = (
                            (1.0 - weight_view) * base_token
                            + weight_view * aligned_ray.to(base_token.dtype)
                        )
                    token_res = self._downstream_head(
                        mixed_head_input, shape[i], pos=pos_i
                    )
                    if 'camera_pose' in token_res:
                        try:
                            from dust3r.utils.camera import pose_encoding_to_camera
                            token_c2w = pose_encoding_to_camera(
                                token_res['camera_pose']
                            )
                        except Exception as error:
                            self._lc_emit(
                                f"[POSE_TOKEN_INCREMENT] frame={i} "
                                f"decode_failed={error}"
                            )
                    if i % 10 == 0:
                        self._lc_emit(
                            f"[POSE_TOKEN_RAY] frame={i} "
                            f"weight={float(token_weight.mean().detach().cpu()):.6f} "
                            f"cosine={float(latent_cosine.mean().detach().cpu()):.6f}"
                        )
                # Always retain v56's independently decoded ray observation
                # and its verified SE(3) consensus.  v68 skipped this path
                # whenever token residuals were enabled.
                ray_res = self._downstream_head(
                    ray_head_input, shape[i], pos=pos_i
                )
                ray_pose = ray_res.get('camera_pose', None)
                if ray_pose is not None:
                    try:
                        from dust3r.utils.camera import (
                            camera_to_pose_encoding,
                            pose_encoding_to_camera,
                        )
                        base_c2w = pose_encoding_to_camera(res['camera_pose'])
                        ray_c2w = pose_encoding_to_camera(ray_pose)
                        fused_c2w, fusion_weight, disagreement_deg, disagreement_trans = (
                            self._pose_only_consensus_fuse(base_c2w, ray_c2w)
                        )
                        res['camera_pose'] = camera_to_pose_encoding(fused_c2w)
                        if i % 10 == 0:
                            self._lc_emit(
                                f"[POSE_ONLY_RAY] frame={i} "
                                f"weight={float(fusion_weight.mean().detach().cpu()):.6f} "
                                f"rot_deg={float(disagreement_deg.mean().detach().cpu()):.6f} "
                                f"trans={float(disagreement_trans.mean().detach().cpu()):.6f}"
                            )
                    except Exception as error:
                        self._lc_emit(
                            f"[POSE_ONLY_RAY] frame={i} fusion_failed={error}"
                        )
            # RayAware-style retain-or-replace at the pose-observation level.
            # Compare each branch against its own pixel-aligned local/world
            # pointmaps, smooth only the GT-free score, and use hysteresis to
            # avoid framewise branch flicker.  The main K-way memory and v56
            # recurrent pose states are unchanged regardless of this choice.
            if token_residual and token_c2w is not None and 'camera_pose' in res:
                try:
                    from dust3r.utils.camera import (
                        camera_to_pose_encoding,
                        pose_encoding_to_camera,
                    )
                    stable_c2w = pose_encoding_to_camera(res['camera_pose'])
                    stable_stats = self._pose_pointmap_consistency(
                        res, stable_c2w
                    )
                    token_stats = self._pose_pointmap_consistency(
                        token_res, token_c2w
                    )
                    if stable_stats is not None and token_stats is not None:
                        stable_score, stable_disp, stable_obs = stable_stats
                        token_score, token_disp, token_obs = token_stats
                        score_delta = stable_score - token_score
                        relative_gain = (
                            score_delta / stable_score.clamp_min(1e-8)
                        )
                        pooled_dispersion = 0.5 * (
                            stable_disp + token_disp
                        )
                        effect_size = (
                            score_delta / pooled_dispersion.clamp_min(1e-8)
                        )
                        observability = torch.minimum(
                            stable_obs, token_obs
                        )
                        momentum = min(0.99, max(0.0, float(os.environ.get(
                            "POINT3R_RAY_POSE_GEOM_EMA", "0.80"
                        ))))
                        if self._v74_relative_gain_ema is None:
                            self._v74_relative_gain_ema = relative_gain.detach()
                            self._v74_effect_size_ema = effect_size.detach()
                            self._v74_observability_ema = observability.detach()
                            self._v74_use_token = torch.zeros_like(
                                stable_score, dtype=torch.bool
                            )
                        else:
                            self._v74_relative_gain_ema = (
                                momentum * self._v74_relative_gain_ema
                                + (1.0 - momentum) * relative_gain.detach()
                            )
                            self._v74_effect_size_ema = (
                                momentum * self._v74_effect_size_ema
                                + (1.0 - momentum) * effect_size.detach()
                            )
                            self._v74_observability_ema = (
                                momentum * self._v74_observability_ema
                                + (1.0 - momentum) * observability.detach()
                            )
                        relative_gain_min = max(0.0, float(os.environ.get(
                            "POINT3R_RAY_POSE_GEOM_REL_GAIN_MIN", "0.01"
                        )))
                        effect_size_min = max(0.0, float(os.environ.get(
                            "POINT3R_RAY_POSE_GEOM_EFFECT_MIN", "0.10"
                        )))
                        observability_min = max(0.0, float(os.environ.get(
                            "POINT3R_RAY_POSE_GEOM_OBS_MIN", "0.005"
                        )))
                        finite = (
                            torch.isfinite(self._v74_relative_gain_ema)
                            & torch.isfinite(self._v74_effect_size_ema)
                            & torch.isfinite(self._v74_observability_ema)
                        )
                        enter = (
                            finite
                            & (self._v74_relative_gain_ema > relative_gain_min)
                            & (self._v74_effect_size_ema > effect_size_min)
                            & (self._v74_observability_ema > observability_min)
                        )
                        stay = (
                            finite
                            & (self._v74_relative_gain_ema > 0.5 * relative_gain_min)
                            & (self._v74_effect_size_ema > 0.5 * effect_size_min)
                            & (self._v74_observability_ema > observability_min)
                        )
                        self._v74_use_token = torch.where(
                            self._v74_use_token, stay, enter
                        )
                        procrustes_c2w = self._pose_pointmap_procrustes(token_res)
                        geometric_token_c2w = token_c2w
                        geometric_weight = torch.zeros_like(relative_gain)
                        if procrustes_c2w is not None:
                            (
                                geometric_token_c2w,
                                geometric_weight,
                                _,
                                _,
                            ) = self._pose_only_consensus_fuse(
                                token_c2w,
                                procrustes_c2w,
                                max_weight_env="POINT3R_RAY_POSE_PROCRUSTES_MAX_WEIGHT",
                                default_max_weight="0.50",
                            )
                        use_token = self._v74_use_token[:, None, None]
                        selected_c2w = torch.where(
                            use_token, geometric_token_c2w, stable_c2w
                        )
                        res['camera_pose'] = camera_to_pose_encoding(
                            selected_c2w
                        )
                        if i % 10 == 0:
                            self._lc_emit(
                                f"[POSE_GEOM_OBSERVABLE] frame={i} "
                                f"relative_gain={float(self._v74_relative_gain_ema.mean().detach().cpu()):.6f} "
                                f"effect={float(self._v74_effect_size_ema.mean().detach().cpu()):.6f} "
                                f"observability={float(self._v74_observability_ema.mean().detach().cpu()):.6f} "
                                f"procrustes_weight={float(geometric_weight.mean().detach().cpu()):.6f} "
                                f"selected={int(self._v74_use_token.sum().detach().cpu())}"
                            )
                except Exception as error:
                    self._lc_emit(
                        f"[POSE_GEOM_OBSERVABLE] frame={i} selection_failed={error}"
                    )
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)
            if pose_only_ensemble:
                ray_mem = (
                    new_ray_mem * update_mask_mem
                    + ray_mem * (1 - update_mask_mem)
                )
                if i % 10 == 0:
                    state_delta = torch.linalg.norm(
                        (ray_mem - mem).float(), dim=-1
                    ).mean()
                    self._lc_emit(
                        f"[RAY_POSE_STREAM] frame={i} "
                        f"state_delta={float(state_delta.detach().cpu()):.6f} "
                        f"independent={int(i > 0 and self._last_pose_only_dec is not None)}"
                    )
            if mask_memory is not None:
                memory_feat_new_list = []
                pos_decode_memory_new_list = []
                for j in range(mask_memory.shape[0]):
                    j_mask_memory = mask_memory[j]
                    j_mask_memory = j_mask_memory.bool()
                    j_memory_feat = memory_feat[j]
                    j_pos_decode_memory = pos_decode_memory[j]
                    j_memory_feat = j_memory_feat[j_mask_memory]
                    j_pos_decode_memory = j_pos_decode_memory[j_mask_memory]
                    memory_feat_new_list.append(j_memory_feat)
                    pos_decode_memory_new_list.append(j_pos_decode_memory)
                memory_feat = memory_feat_new_list
                pos_decode_memory = pos_decode_memory_new_list
                init_memory_feat = [memory_feat_in.clone().detach() for memory_feat_in in memory_feat]

            lc_enabled = os.environ.get("POINT3R_LC_ENABLED", "0").lower() in ("1","true","yes","on")
            rayaware_enabled = os.environ.get("POINT3R_RAYAWARE_UPDATE", "0").lower() in ("1","true","yes","on")

            # Save c2w for world-ray construction and optional loop closure.
            if lc_enabled or rayaware_enabled:
                cam_pose = res.get('camera_pose', None)
                if cam_pose is not None:
                    try:
                        from dust3r.utils.camera import pose_encoding_to_camera
                        bs_pose = cam_pose.shape[0]
                        # 保证 trajectory 是 list of list，外层 batch，内层帧
                        # 这里简化处理：trajectory 只存 batch 0（pose eval 也只用 batch 0）
                        # 如果需要支持 batch > 1，需要把 trajectory 改成二维 list
                        for b in range(min(bs_pose, 1)):  # 当前只支持 batch 0
                            c2w_b = pose_encoding_to_camera(cam_pose[b:b+1])
                            if c2w_b.dim() == 3:
                                c2w_b = c2w_b[0]
                            c2w_b = c2w_b.detach()
                            while len(self._pose_trajectory) <= i:
                                self._pose_trajectory.append(None)
                            self._pose_trajectory[i] = c2w_b
                    except Exception:
                        while len(self._pose_trajectory) <= i:
                            self._pose_trajectory.append(None)
                else:
                    while len(self._pose_trajectory) <= i:
                        self._pose_trajectory.append(None)

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                this_pts3d_self = res.get('pts3d_in_self_view', None)
                if this_pts3d_self is not None:
                    this_pts3d_self = this_pts3d_self.clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img= self._forward_addmemory_merge(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                    conf_i=res.get('conf', None),
                    local_pts3d=this_pts3d_self,
                )
                if i in self._online_pose_refinements:
                    try:
                        from dust3r.utils.camera import camera_to_pose_encoding
                        corrected_pose = self._online_pose_refinements[i]
                        res['camera_pose'] = camera_to_pose_encoding(
                            corrected_pose.to(res['camera_pose'].device).unsqueeze(0)
                        )
                    except Exception as error:
                        self._lc_emit(
                            f"[POSE_STATIC] frame={i} pose writeback failed: {error}"
                        )

                # v33: close the loop during streaming, not only after the
                # final frame.  Correct both banks and feed their rehashed
                # coordinates into the next decoder readout.
                online_correction = os.environ.get(
                    "POINT3R_LC_ONLINE_CORRECTION", "0"
                ).lower() in ("1", "true", "yes", "on")
                online_interval = max(1, int(os.environ.get(
                    "POINT3R_LC_ONLINE_INTERVAL", "15"
                )))
                if (
                    online_correction
                    and lc_enabled
                    and i > 0
                    and (i + 1) % online_interval == 0
                    and self._lc_rotation_edges
                ):
                    online_before = [
                        None if pose is None else pose.detach().cpu().float().clone()
                        for pose in self._pose_trajectory
                    ]
                    online_corrections = self._lc_rotation_graph_with_priors(i + 1)
                    if os.environ.get(
                        "POINT3R_LC_DECOUPLED_SE3", "0"
                    ).lower() in ("1", "true", "yes", "on"):
                        online_corrections = self._lc_translation_graph_with_priors(
                            i + 1, online_corrections
                        )
                    self._lc_transform_and_rehash_bank(
                        self._dual_ray_bank, online_before, online_corrections
                    )
                    self._lc_transform_and_rehash_bank(
                        self._dual_stable_bank, online_before, online_corrections
                    )
                    if pos_decode_img is not None and i in online_corrections:
                        before_i = online_before[i].to(pos_decode_img.device)
                        after_i = online_corrections[i].to(pos_decode_img.device)
                        delta_i = after_i @ torch.linalg.inv(before_i)
                        pos_decode_img = (
                            torch.einsum(
                                'ij,bnj->bni',
                                delta_i[:3, :3].to(pos_decode_img.dtype),
                                pos_decode_img,
                            )
                            + delta_i[:3, 3].to(pos_decode_img.dtype).view(1, 1, 3)
                        )
                    dynamic_threshold = float(os.environ.get(
                        "POINT3R_STATIC_GATE_SCENE_THRESHOLD", "0.45"
                    ))
                    dynamic_warmup = int(os.environ.get(
                        "POINT3R_STATIC_GATE_WARMUP", "10"
                    ))
                    use_stable = (
                        self._static_gate_dynamic_latched is not None
                        and any(self._static_gate_dynamic_latched)
                    )
                    active_bank = self._dual_stable_bank if use_stable else self._dual_ray_bank
                    if active_bank is not None:
                        memory_feat = list(active_bank["feat"])
                        pos_decode_memory = list(active_bank["pos"])
                        init_memory_feat = [value.clone().detach() for value in memory_feat]
                        self._ordered_slot_tables = list(active_bank["table"])
                        self._ordered_slot_counts = list(active_bank["count"])
                        self._ordered_slot_confs = list(active_bank["conf"])
                        self._ordered_slot_rays = list(active_bank["ray"])
                        self._ordered_slot_times = list(active_bank["time"])
                        self._ordered_slot_locals = list(active_bank["local"])
                    try:
                        from dust3r.utils.camera import camera_to_pose_encoding
                        for corrected_frame, corrected_pose in online_corrections.items():
                            if corrected_frame >= len(ress):
                                continue
                            pose_device = ress[corrected_frame]['camera_pose'].device
                            ress[corrected_frame]['camera_pose'] = camera_to_pose_encoding(
                                corrected_pose.to(pose_device).unsqueeze(0)
                            )
                    except Exception as error:
                        self._lc_emit(f"[LC_ONLINE] pose writeback failed: {error}")
                    self._lc_emit(
                        f"[LC_ONLINE] frame={i} corrections={len(online_corrections)} "
                        f"active_bank={'stable' if use_stable else 'ray'}"
                    )

        # ---- PGO 后写回 ress[i]["camera_pose"] ----
        if lc_enabled and (self._lc_candidates or self._lc_odometry_edges or self._lc_rotation_edges):
            n_frames = len(self._pose_trajectory)
            poses_before_correction = [
                None if pose is None else pose.detach().cpu().float().clone()
                for pose in self._pose_trajectory
            ]
            direct_odom = os.environ.get(
                "POINT3R_LC_DIRECT_RAY_ODOM", "0"
            ).lower() in ("1", "true", "yes", "on")
            rotation_graph = os.environ.get(
                "POINT3R_LC_ROTATION_GRAPH", "0"
            ).lower() in ("1", "true", "yes", "on")
            decoupled_se3 = os.environ.get(
                "POINT3R_LC_DECOUPLED_SE3", "0"
            ).lower() in ("1", "true", "yes", "on")
            if decoupled_se3:
                corrections = self._lc_rotation_graph_with_priors(n_frames)
                translation_mode = os.environ.get(
                    "POINT3R_LC_TRANSLATION_MODE", "graph"
                ).lower()
                if translation_mode == "chain":
                    corrections = self._lc_translation_chain_with_anchors(
                        n_frames, corrections
                    )
                else:
                    corrections = self._lc_translation_graph_with_priors(
                        n_frames, corrections
                    )
            elif rotation_graph:
                pointer_loop_pgo = os.environ.get(
                    "POINT3R_RAY_POINTER_LOOP_PGO", "0"
                ).lower() in ("1", "true", "yes", "on")
                if pointer_loop_pgo and self._lc_candidates:
                    # First distribute verified long-time SE(3) closure over
                    # the trajectory, then retain v50's adjacent rotation
                    # graph and state regularizer on the corrected backbone.
                    self._lc_run_pgo(n_frames)
                corrections = self._lc_rotation_graph_with_priors(n_frames)
            elif direct_odom:
                corrections = self._lc_direct_ray_odometry(n_frames)
                run_loop_after = os.environ.get(
                    "POINT3R_LC_DIRECT_RUN_LOOP_PGO", "0"
                ).lower() in ("1", "true", "yes", "on")
                if run_loop_after and self._lc_candidates:
                    corrections = self._lc_run_pgo(n_frames)
            else:
                corrections = self._lc_run_pgo(n_frames)
            corrections = self._lc_state_aware_trajectory_regularization(
                corrections
            )
            if corrections:
                sync_memory = os.environ.get(
                    "POINT3R_LC_SYNC_MEMORY", "0"
                ).lower() in ("1", "true", "yes", "on")
                if sync_memory:
                    self._lc_transform_and_rehash_bank(
                        self._dual_ray_bank,
                        poses_before_correction,
                        corrections,
                    )
                    self._lc_transform_and_rehash_bank(
                        self._dual_stable_bank,
                        poses_before_correction,
                        corrections,
                    )
                # 把修正后的 c2w 转换回 camera_pose encoding，写回 ress
                try:
                    from dust3r.utils.camera import camera_to_pose_encoding
                    for fi, c2w_corrected in corrections.items():
                        if fi < len(ress) and ress[fi] is not None:
                            c2w_dev = c2w_corrected.to(
                                ress[fi]['camera_pose'].device
                                if 'camera_pose' in ress[fi] else torch.device('cpu'))
                            new_pose = camera_to_pose_encoding(c2w_dev.unsqueeze(0))  # [1,7]
                            if 'camera_pose' in ress[fi]:
                                ress[fi]['camera_pose'] = new_pose
                            sync_geometry = os.environ.get(
                                "POINT3R_LC_SYNC_GEOMETRY", "0"
                            ).lower() in ("1", "true", "yes", "on")
                            if (
                                sync_geometry
                                and fi < len(poses_before_correction)
                                and poses_before_correction[fi] is not None
                                and 'pts3d_in_other_view' in ress[fi]
                            ):
                                before = poses_before_correction[fi].to(c2w_dev.device)
                                delta = c2w_dev @ torch.linalg.inv(before)
                                points = ress[fi]['pts3d_in_other_view']
                                ress[fi]['pts3d_in_other_view'] = (
                                    torch.einsum(
                                        'ij,bhwj->bhwi',
                                        delta[:3, :3].to(points.device, points.dtype),
                                        points,
                                    )
                                    + delta[:3, 3].to(points.device, points.dtype).view(1, 1, 1, 3)
                                )
                except Exception as e:
                    print(f"[LC] Warning: could not write back pose corrections: {e}", flush=True)

        if lc_enabled:
            diag = self._lc_diag or {}
            self._lc_emit(
                "[LC_SUMMARY] "
                f"frames={diag.get('frames', 0)} history_frames={diag.get('history_frames', 0)} "
                f"spatial_pairs={diag.get('spatial_pairs', 0)} ray_pairs={diag.get('ray_pairs', 0)} "
                f"mutual_pairs={diag.get('mutual_pairs', 0)} "
                f"verified_edges={diag.get('verified_edges', 0)} "
                f"pgo_executed={int(self._lc_pgo_executed)}"
            )
        return ress, views

    def _forward(self, views, point3r_tag=False):
        self.memory_update_stats = []
        self.sparse_readout_stats = []
        self._last_sparse_readout_stats = None
        self._ordered_slot_tables = None
        self._ordered_slot_counts = None
        self._ordered_slot_confs = None
        self._ordered_slot_rays = None
        self._ordered_slot_times = None
        self._ordered_slot_locals = None
        self._static_gate_dynamic_ema = None
        self._static_gate_dynamic_latched = None
        self._dual_ray_bank = None
        self._dual_stable_bank = None
        self._last_pose_only_dec = None
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            
            mask_memory = None

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = None
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                pose_pos_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            res = self._apply_geometry_safe_pose_readout(
                res, dec[-1][:, 0:1], pos_decode_img, i
            )
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)  

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img = self._forward_addmemory(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                )

        return ress, views

    def forward(self, views, point3r_tag=False):
        ress, views= self._forward_merge(views, point3r_tag=point3r_tag)
        return ARCroco3DStereoOutput(ress=ress, views=views)
        # stage1
        # ress, views = self._forward(views, point3r_tag=point3r_tag)
        # return ARCroco3DStereoOutput(ress=ress, views=views)

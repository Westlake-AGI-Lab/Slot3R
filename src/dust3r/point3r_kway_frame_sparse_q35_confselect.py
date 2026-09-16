import sys
import os

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
        self.confselect_merge_threshold = float(os.environ.get("POINT3R_CONFSELECT_MERGE_THRESHOLD", "0.90"))
        self.confselect_default_conf = float(os.environ.get("POINT3R_CONFSELECT_DEFAULT_CONF", "1.0"))
        self.confselect_decay_gamma = float(os.environ.get("POINT3R_CONFSELECT_DECAY_GAMMA", "1.0"))
        self.confselect_conf_margin = float(os.environ.get("POINT3R_CONFSELECT_CONF_MARGIN", "0.0"))
        self.confselect_collect_stats = os.environ.get("POINT3R_CONFSELECT_STATS", "0").lower() in ("1", "true", "yes", "on")
        print("[Q35_CONFSELECT_SPARSE512] drop_quantile=0.35 merge_threshold=" f"{self.confselect_merge_threshold:.3f}", flush=True)

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

    def _sparse_readout_select_indices(self, memory_pos_j, query_pos_j, valid_mask_j=None):
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
        max_tokens = int(self.sparse_readout_max_tokens)
        if max_tokens > 0 and selected.numel() > max_tokens:
            if anchor_idx.numel() >= max_tokens:
                selected = torch.unique(anchor_idx)[:max_tokens]
            else:
                is_anchor = torch.zeros(memory_pos_j.shape[0], dtype=torch.bool, device=device)
                is_anchor[anchor_idx] = True
                rest = selected[~is_anchor[selected]]
                selected = torch.cat((torch.unique(anchor_idx), rest[: max_tokens - torch.unique(anchor_idx).numel()]), dim=0)

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
            selected, stats = self._sparse_readout_select_indices(memory_pos[j], query_pos[j], valid_mask_j)
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
        point3r_tag=False,
    ):
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
        return new_memory_feat, dec

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
    ):
        num_slots = max(1, int(self.kway_num_slots))
        num_new = int(img_pos_j.shape[0])
        num_buckets = int(self._ordered_num_buckets())
        device = img_pos_j.device
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
            return memory_feat_j, memory_pos_j, slot_table_j, slot_count_j, slot_conf_j, stats

        if new_conf_j is None:
            new_conf_j = torch.full((num_new,), float(self.confselect_default_conf), dtype=torch.float32, device=device)
        else:
            new_conf_j = new_conf_j.reshape(-1).to(device=device, dtype=torch.float32)
            if new_conf_j.shape[0] != num_new:
                new_conf_j = torch.full((num_new,), float(self.confselect_default_conf), dtype=torch.float32, device=device)
        new_conf_j = torch.nan_to_num(new_conf_j, nan=0.0, posinf=0.0, neginf=0.0)

        img_keys = self._ordered_pack_bins(self._ordered_spatial_bins(img_pos_j)).long()
        sorted_keys, sort_idx = torch.sort(img_keys)
        unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

        old_counts_for_token = slot_count_j[img_keys].clamp_max(num_slots)
        cand_idx = slot_table_j[img_keys].clamp_min(0)
        local_ids = torch.arange(num_slots, device=device, dtype=torch.long).unsqueeze(0)
        valid_slots = local_ids < old_counts_for_token.unsqueeze(1)
        has_existing = old_counts_for_token > 0
        best_sim = torch.full((num_new,), -float("inf"), dtype=torch.float32, device=device)
        best_local = torch.zeros((num_new,), dtype=torch.long, device=device)
        best_conf = torch.full((num_new,), -float("inf"), dtype=torch.float32, device=device)
        min_conf = torch.full((num_new,), float("inf"), dtype=torch.float32, device=device)

        if memory_feat_j.shape[0] > 0:
            cand_feat = memory_feat_j[cand_idx]
            query = F.normalize(new_feat_j.float(), dim=-1)
            keys = F.normalize(cand_feat.float(), dim=-1)
            sims = (query.unsqueeze(1) * keys).sum(dim=-1).masked_fill(~valid_slots, -float("inf"))
            best_sim, best_local = sims.max(dim=1)
            cand_conf = slot_conf_j[cand_idx].masked_fill(~valid_slots, float("inf"))
            min_conf, _ = cand_conf.min(dim=1)
            best_conf = slot_conf_j[cand_idx[
                torch.arange(num_new, device=device), best_local
            ]].float()

        similar_mask = has_existing & (best_sim >= float(self.confselect_merge_threshold))
        conf_margin = float(self.confselect_conf_margin)
        merge_mask = similar_mask & (new_conf_j >= best_conf + conf_margin)
        distinct_candidate_mask = (
            has_existing
            & (best_sim < float(self.confselect_merge_threshold))
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
            beta = (conf_mean / (old_conf + conf_mean + 1e-6)).to(dtype=memory_feat_all.dtype).unsqueeze(-1)
            memory_feat_all[unique_targets] = (1.0 - beta) * memory_feat_all[unique_targets] + beta * feat_mean
            beta_pos = beta.to(dtype=memory_pos_all.dtype)
            memory_pos_all[unique_targets] = (1.0 - beta_pos) * memory_pos_all[unique_targets] + beta_pos * pos_mean
            slot_conf_all[unique_targets] = ((1.0 - beta.squeeze(-1).float()) * old_conf + beta.squeeze(-1).float() * conf_mean).float()

        # Full buckets need a bounded replacement path; otherwise decay only
        # lowers the threshold but cannot admit distinct observations.
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
            replace_keys = img_keys[replace_indices]
            replace_sorted_keys, replace_sort_idx = torch.sort(replace_keys)
            replace_unique_keys, replace_counts = torch.unique_consecutive(replace_sorted_keys, return_counts=True)
            replace_starts = torch.cat([replace_counts.new_zeros(1), replace_counts.cumsum(0)[:-1]])
            replace_rank_sorted = torch.arange(replace_indices.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(replace_starts, replace_counts)
            replace_rank = torch.empty_like(replace_rank_sorted)
            replace_rank[replace_sort_idx] = replace_rank_sorted
            replace_indices = replace_indices[replace_rank == 0]
            replace_accept_mask[replace_indices] = True
            replace_keys = img_keys[replace_indices]
            replace_cand_idx = slot_table_j[replace_keys].clamp_min(0)
            replace_conf = slot_conf_all[replace_cand_idx]
            replace_local = replace_conf.argmin(dim=1)
            replace_targets = replace_cand_idx[torch.arange(replace_indices.shape[0], device=device), replace_local]
            memory_feat_all[replace_targets] = new_feat_j[replace_indices]
            memory_pos_all[replace_targets] = img_pos_j[replace_indices]
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
        return memory_feat_all, memory_pos_all, slot_table_j, slot_count_j, slot_conf_all, stats

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
            else:
                prev_feat = memory_feat[j]
                prev_pos = memory_pos[j]
                slot_table_j = self._ordered_slot_tables[j]
                slot_count_j = self._ordered_slot_counts[j]
                slot_conf_j = self._ordered_slot_confs[j]

            write_feat_j = memory_add[j]
            write_pos_j = img_pos[j]
            write_weight_j = conf_weight[j] if conf_weight is not None else None
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
                write_weight_j = conf_vals[keep]
                cgmc_kept = int(write_pos_j.shape[0])
                # Fast timing path: scalar diagnostics stay disabled to avoid
                # synchronizing CUDA with the host on every frame.

            memory_feat_j, memory_pos_j, slot_table_j, slot_count_j, slot_conf_j, stats_j = self._ordered_update_single_tensor(
                prev_feat,
                prev_pos,
                write_feat_j,
                write_pos_j,
                slot_table_j,
                slot_count_j,
                slot_conf_j,
                new_conf_j=write_weight_j if cgmc_weighted_merge else None,
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
        )

    def _forward_merge(self, views, point3r_tag=False):
        self.memory_update_stats = []
        self.sparse_readout_stats = []
        self._last_sparse_readout_stats = None
        self._ordered_slot_tables = None
        self._ordered_slot_counts = None
        self._ordered_slot_confs = None
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
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
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)
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

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
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
                )
                
        return ress, views

    def _forward(self, views, point3r_tag=False):
        self.memory_update_stats = []
        self.sparse_readout_stats = []
        self._last_sparse_readout_stats = None
        self._ordered_slot_tables = None
        self._ordered_slot_counts = None
        self._ordered_slot_confs = None
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

    

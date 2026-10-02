"""Model names and paper configurations shared by evaluation tasks."""
import os

MODEL_MODULES = {
    "core": "dust3r.point3r_kway_frame_sparse_q35_confselect",
    "vpc_m": "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v82e_balanced_predecoder_pose",
    "vpc_a": "dust3r.point3r_kway_frame_sparse_q35_confselect_rayaware_v106_fresh_bank_pose",
    "point3r": "dust3r.point3r",
}
MODEL_ALIASES = {"ours": "core", "ours_ray": "vpc_m", "ours_rayma": "vpc_a"}
MODELS = tuple(MODEL_MODULES) + ("ghost", "cut3r", "ttt3r")
RAY_SETTINGS = {
    "POINT3R_RAYAWARE_UPDATE": "1",
    "POINT3R_RAY_DUAL_BANK": "1",
    "POINT3R_RAY_PAPER_UPDATE": "1",
    "POINT3R_RAY_KWAY_DIVERSE_UPDATE": "0",
    "POINT3R_RAY_HYBRID_READOUT": "1",
    "POINT3R_RAY_POSE_INPUT_ONLY": "1",
    "POINT3R_RAY_POSE_POST_DECODER_ONLY": "0",
    "POINT3R_RAY_POSE_ONLY_ENSEMBLE": "0",
    "POINT3R_RAY_POSE_INPUT_TOKENS": "128",
    "POINT3R_RAY_POSE_INPUT_TEMPERATURE": "0.10",
}
VARIANT_SETTINGS = {
    "core": {"POINT3R_RAYAWARE_UPDATE": "0"},
    "vpc_m": {**RAY_SETTINGS, "POINT3R_RAY_BANK_UPDATE_EVERY": "4",
              "POINT3R_RAY_POSE_INPUT_MAX_WEIGHT": "0.025"},
    "vpc_a": {**RAY_SETTINGS, "POINT3R_RAY_BANK_UPDATE_EVERY": "1",
              "POINT3R_V106_RAY_BANK_UPDATE_EVERY": "1",
              "POINT3R_V106_POSE_INPUT_WEIGHT": "0.15"},
}

def canonical_model(name):
    return MODEL_ALIASES.get(name, name)

def configure_model(args):
    """One configuration path for both direct Python and shell entrypoints."""
    clear_point3r_env()
    for settings in VARIANT_SETTINGS.values():
        for key in settings:
            os.environ.pop(key, None)
    if args.model not in VARIANT_SETTINGS:
        return
    os.environ.update({
        "POINT3R_MEMORY_UPDATE_MODE": "ordered_kway",
        "POINT3R_MEMORY_IMPL": "tensor",
        "POINT3R_ORDERED_UPDATE_IMPL": "tensor",
        "POINT3R_KWAY_NUM_SLOTS": str(args.kway_slots),
        "POINT3R_ORDERED_WAY_POLICY": "appearance",
        "POINT3R_ORDERED_THETA_BINS": str(args.theta_bins),
        "POINT3R_ORDERED_PHI_BINS": str(args.phi_bins),
        "POINT3R_ORDERED_RHO_BINS": str(args.rho_bins),
        "POINT3R_SPARSE_READOUT": "1",
        "POINT3R_SPARSE_MODE": "max",
        "POINT3R_SPARSE_MAX_TOKENS": str(args.sparse_max_tokens),
        "POINT3R_SPARSE_GLOBAL_ANCHORS": str(args.sparse_global_anchors),
        "POINT3R_SPARSE_NEIGHBOR_RANGE": str(args.sparse_neighbor_range),
        "POINT3R_ENCODE_CHUNK_SIZE": str(args.encode_chunk_size),
        "POINT3R_CGMC_DROP_QUANTILE": str(args.drop_quantile),
        "POINT3R_CGMC_WEIGHTED_MERGE": "1",
    })
    os.environ.setdefault("POINT3R_CONFSELECT_STATS", "0")
    os.environ.update(VARIANT_SETTINGS[args.model])

def clear_point3r_env() -> None:
    for key in (
        "POINT3R_MEMORY_UPDATE_MODE",
        "POINT3R_ORDERED_UPDATE_IMPL",
        "POINT3R_KWAY_NUM_SLOTS",
        "POINT3R_ORDERED_WAY_POLICY",
        "POINT3R_ORDERED_THETA_BINS",
        "POINT3R_ORDERED_PHI_BINS",
        "POINT3R_ORDERED_RHO_BINS",
        "POINT3R_ORDERED_APP_THRESHOLD",
        "POINT3R_ORDERED_BUDGET_EVICT",
        "POINT3R_SPARSE_READOUT",
        "POINT3R_SPARSE_MODE",
        "POINT3R_SPARSE_MAX_TOKENS",
        "POINT3R_SPARSE_GLOBAL_ANCHORS",
        "POINT3R_SPARSE_NEIGHBOR_RANGE",
        "POINT3R_SPARSE_RECENT_TOKENS",
        "POINT3R_SPARSE_RECENT_FRAMES",
        "POINT3R_FIXED_MEMORY_TOKENS",
        "POINT3R_FIXED_MEMORY_MODE",
        "POINT3R_GEOANCHOR",
        "POINT3R_GEOANCHOR_STRIDE",
        "POINT3R_GEOANCHOR_H",
        "POINT3R_GEOANCHOR_FRAMES",
        "POINT3R_GEOANCHOR_MIN_GAP",
        "POINT3R_GEOANCHOR_BUCKETS_PER_KF",
        "POINT3R_GEOANCHOR_SLOTS_PER_KF",
        "POINT3R_GEOANCHOR_MAX_KFS",
        "POINT3R_PROFILE",
    ):
        os.environ.pop(key, None)


# Pose/depth retain the original common_env.sh state-regularization protocol.
TASK_SETTINGS = {
    "POINT3R_KWAY_NUM_SLOTS_FORCE": "8",
    "POINT3R_CONFSELECT_MERGE_THRESHOLD": "0.90",
    "POINT3R_CONFSELECT_STATS": "0",
    "POINT3R_CGMC": "1", "POINT3R_CGMC_MIN_CONF": "0.0",
}
POSE_STATE_SETTINGS = {
    "POINT3R_LC_ENABLED": "1", "POINT3R_LC_LOOP_DETECTION": "0",
    "POINT3R_LC_ODOM_MAX_LAG": "3", "POINT3R_LC_FULL_SE3": "0",
    "POINT3R_LC_DECOUPLED_SE3": "0", "POINT3R_LC_ROTATION_GRAPH": "1",
    "POINT3R_LC_DIRECT_RAY_ODOM": "0", "POINT3R_LC_STATE_REGULARIZATION": "1",
    "POINT3R_LC_STATE_LIE_INCREMENT": "1", "POINT3R_LC_STATE_TRANS_STRENGTH": "30",
    "POINT3R_LC_STATE_ROT_STRENGTH": "30", "POINT3R_LC_STATE_LOCAL_ROT_MODE": "0",
    "POINT3R_LC_STATE_LOW_RAY_FUSION": "1", "POINT3R_LC_STATE_RAY_FUSION_TRANSPORT_ONLY": "1",
    "POINT3R_LC_STATE_CYCLE_OBSERVABLE_FUSION": "0", "POINT3R_LC_STATE_RAY_FUSION_PRIOR": "1",
    "POINT3R_LC_STATE_RAY_FUSION_DELTA_DEG": "2", "POINT3R_LC_STATE_RAY_FUSION_MAX_STEP_DEG": "2",
    "POINT3R_LC_ODOM_CENTERED_ROTATION": "0", "POINT3R_LC_ODOM_ESSENTIAL_ROTATION": "0",
    "POINT3R_RAY_POINTER_LOOP_GRAPH": "0", "POINT3R_RAY_POINTER_LOOP_PGO": "0",
}

def configure_task_model(args):
    configure_model(args)
    for key in (*TASK_SETTINGS, *POSE_STATE_SETTINGS):
        os.environ.pop(key, None)
    if args.model in VARIANT_SETTINGS:
        os.environ.update(TASK_SETTINGS)
        os.environ["POINT3R_KWAY_NUM_SLOTS_FORCE"] = str(args.kway_slots)
        os.environ["POINT3R_LC_ENABLED"] = "0"
        if args.model in ("vpc_m", "vpc_a"):
            os.environ.update(POSE_STATE_SETTINGS)


def add_model_arguments(parser):
    parser.add_argument("--model", type=canonical_model, choices=tuple(MODEL_MODULES), default="core")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, choices=(224, 512), default=512)
    for name, default in (("kway_slots",8),("theta_bins",16),("phi_bins",8),("rho_bins",32),
                          ("sparse_max_tokens",640),("sparse_global_anchors",128),
                          ("sparse_neighbor_range",1),("encode_chunk_size",100)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--drop_quantile", type=float, default=0.25)


def load_task_model(args):
    import importlib
    configure_task_model(args)
    module = MODEL_MODULES[args.model]
    print(f"[model] {args.model}: {module}", flush=True)
    return importlib.import_module(module).Point3R.from_pretrained(args.weights).to(args.device).eval()

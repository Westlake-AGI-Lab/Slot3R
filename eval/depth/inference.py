"""Inference for prepared indoor and KITTI views."""
import time
import torch

def move_batch_to_device(batch, device: str):
    keep_cpu = {"depthmap", "valid_mask", "idx", "instance", "true_shape"}
    for view in batch:
        for key, value in list(view.items()):
            if key in keep_cpu:
                continue
            if torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return batch


def predict(args, model, batch):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    start = time.time()
    with torch.no_grad():
        if args.dataset == "kitti":
            from dust3r.inference import inference
            preds = inference(batch, model, args.device)["pred"]
        else:
            batch = move_batch_to_device(batch, args.device)
            with torch.cuda.amp.autocast(enabled=False):
                preds = model(batch, point3r_tag=True).ress
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time()-start
    peak = torch.cuda.max_memory_allocated()/(1024**3) if torch.cuda.is_available() else 0.
    return preds, elapsed, peak

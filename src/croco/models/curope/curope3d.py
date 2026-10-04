import torch
import torch.nn as nn

# Reuse the 2D wrapper's module; do not load the same extension under two names.
from .curope2d import _kernels

class cuRoPE3D(nn.Module):
    def __init__(self, freq: float = 100.0, F0: float = 1.0):
        super().__init__()
        self.base = freq
        self.F0 = float(F0)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        len_tokens    = tokens.shape[2]
        len_positions = positions.shape[1]
        assert len_tokens == len_positions or len_tokens == len_positions + 1, (
            f"tokens: {len_tokens}, positions: {len_positions}"
        )
        if len_tokens != len_positions:
            pose_token = tokens[:, :, :1, :]
            img_tokens = tokens[:, :, 1:, :]
        else:
            pose_token = None
            img_tokens = tokens

        pos_f32  = positions.float().contiguous()
        img_cont = img_tokens.contiguous()
        encoded  = _kernels.rope_3d(img_cont, pos_f32, self.F0)

        if pose_token is not None:
            return torch.cat([pose_token, encoded], dim=2)
        return encoded

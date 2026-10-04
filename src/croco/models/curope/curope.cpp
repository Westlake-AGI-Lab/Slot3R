/* 
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/
#include <torch/extension.h>

// ---------------------------------------------------------------
// RoPE 2D forward declarations
// ---------------------------------------------------------------
void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd );

void rope_2d_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd )
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3) / 4;
    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<int64_t, 3>();
    for (int b = 0; b < B; b++) {
      for (int x = 0; x < 2; x++) {
        for (int n = 0; n < N; n++) {
            const int p = pos[b][n][x];
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    float u = tok[b][n][h][d+0+x*2*D];
                    float v = tok[b][n][h][d+D+x*2*D];
                    const float inv_freq = fwd * p / powf(base, d/float(D));
                    float c = cosf(inv_freq);
                    float s = sinf(inv_freq);
                    tok[b][n][h][d+0+x*2*D] = u*c - v*s;
                    tok[b][n][h][d+D+x*2*D] = v*c + u*s;
                }
            }
        }
      }
    }
}

void rope_2d( torch::Tensor tokens,      // B,N,H,D
              const torch::Tensor positions,  // B,N,2
              const float base,
              const float fwd )
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(positions.size(2) == 2, "positions.shape[2] must be equal to 2");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device");
    if (tokens.is_cuda())
        rope_2d_cuda( tokens, positions, base, fwd );
    else
        rope_2d_cpu( tokens, positions, base, fwd );
}

// ---------------------------------------------------------------
// RoPE 3D forward declaration
// tokens_in:  [B, H, N, D]  float32 or float16
// positions:  [B, N, 3]     float32
// tokens_out: [B, H, N, D]  pre-allocated, same dtype as tokens_in
// ---------------------------------------------------------------
void rope_3d_cuda(
    const torch::Tensor tokens_in,
    const torch::Tensor positions,
    torch::Tensor       tokens_out,
    const float F0
);

torch::Tensor rope_3d(
    torch::Tensor       tokens,    // [B, H, N, D]
    const torch::Tensor positions, // [B, N, 3]  float32
    const float F0 = 1.0f
) {
    TORCH_CHECK(tokens.dim() == 4,    "tokens must be [B, H, N, D]");
    TORCH_CHECK(positions.dim() == 3, "positions must be [B, N, 3]");
    TORCH_CHECK(tokens.size(0) == positions.size(0),
                "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(2) == positions.size(1),
                "seq_length differs between tokens & positions");
    TORCH_CHECK(positions.size(2) == 3, "positions.shape[2] must be 3");
    TORCH_CHECK(tokens.is_cuda() && positions.is_cuda(),
                "rope_3d currently requires CUDA tensors");
    TORCH_CHECK(tokens.is_contiguous(),   "tokens must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");

    auto tokens_out = torch::empty_like(tokens);
    rope_3d_cuda(tokens, positions, tokens_out, F0);
    return tokens_out;
}

// ---------------------------------------------------------------
// pybind
// ---------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_2d", &rope_2d, "RoPE 2d forward/backward");
    m.def("rope_3d", &rope_3d,
          "RoPE 3D forward (fused CUDA, returns new tensor)",
          py::arg("tokens"),
          py::arg("positions"),
          py::arg("F0") = 1.0f);
}

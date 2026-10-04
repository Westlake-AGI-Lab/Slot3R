/* 
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(tensor) {\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory"); \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous"); }
void CHECK_KERNEL() {auto error = cudaGetLastError(); TORCH_CHECK( error == cudaSuccess, cudaGetErrorString(error));}


template < typename scalar_t  >
__global__ void rope_2d_cuda_kernel( 
        //scalar_t* __restrict__ tokens, 
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> tokens,
        const int64_t* __restrict__ pos, 
        const float base, 
        const float fwd )
        // const int N, const int H, const int D )
{
    // tokens shape = (B, N, H, D)
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);
    
    // each block update a single token, for all heads
    // each thread takes care of a single output
    extern __shared__ float shared[];
    float* shared_inv_freq = shared + D;

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    const int Q = D / 4; 
    // one token = [0..Q : Q..2Q : 2Q..3Q : 3Q..D]
    //              u_Y     v_Y     u_X      v_X

    // shared memory: first, compute inv_freq
    if (threadIdx.x < Q)
        shared_inv_freq[threadIdx.x] = fwd / powf(base, threadIdx.x/float(Q));
    __syncthreads();

    // start of X or Y part
    const int X = threadIdx.x < D/2 ? 0 : 1; 
    const int m = (X*D/2) + (threadIdx.x % Q);   // index of u_Y or u_X

    // grab the cos,sin appropriate for me
    const float freq = pos[blockIdx.x*2+X] * shared_inv_freq[threadIdx.x % Q];
    const float cos = cosf(freq);
    const float sin = sinf(freq);
    /*
    float* shared_cos_sin = shared + D + D/4;
    if ((threadIdx.x % (D/2)) < Q)
        shared_cos_sin[m+0] = cosf(freq);
    else
        shared_cos_sin[m+Q] = sinf(freq);
    __syncthreads();
    const float cos = shared_cos_sin[m+0];
    const float sin = shared_cos_sin[m+Q];
    */

    for (int h = 0; h < H; h++)
    {
        // then, load all the token for this head in shared memory
        shared[threadIdx.x] = tokens[b][n][h][threadIdx.x];
        __syncthreads();

        const float u = shared[m];
        const float v = shared[m+Q];
        
        // write output
        if ((threadIdx.x % (D/2)) < Q)
            tokens[b][n][h][threadIdx.x] = u*cos - v*sin;
        else
            tokens[b][n][h][threadIdx.x] = v*cos + u*sin;
    }
}

void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd ) 
{
    const int B = tokens.size(0); // batch size
    const int N = tokens.size(1); // sequence length
    const int H = tokens.size(2); // number of heads
    const int D = tokens.size(3); // dimension per head

    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens are not contiguous");
    TORCH_CHECK(pos.is_contiguous(), "positions are not contiguous");
    TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == 2, "bad pos.shape");
    TORCH_CHECK(D % 4 == 0, "token dim must be multiple of 4");

    // one block for each layer, one thread per local-max
    const int THREADS_PER_BLOCK = D;
    const int N_BLOCKS = B * N; // each block takes care of H*D values
    const int SHARED_MEM = sizeof(float) * (D + D/4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
        rope_2d_cuda_kernel<scalar_t> <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>> (
            //tokens.data_ptr<scalar_t>(), 
            tokens.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            pos.data_ptr<int64_t>(), 
            base, fwd); //, N, H, D );
    }));
}

// ============================================================
// cuRoPE3D: fused CUDA kernel for RoPE3DContinuous
// Added to kernels.cu alongside rope_2d_cuda_kernel
//
// Mathematical spec (D_head=64, H=16):
//   tokens:    [B, H, N, D_head]   float32 or float16
//   positions: [B, N, 3]           float32  (x, y, z coords)
//
//   D_head split into 3 axes:
//     x: dims [0,      D_x)        D_x = D_head/3
//     y: dims [D_x,    2*D_x)      D_y = D_head/3
//     z: dims [2*D_x,  D_head)     D_z = D_head - 2*D_x
//
//   For each axis, 4 bases {10,100,1000,10000}:
//     freq_idx = d_local % (D_axis/2)
//     inv_freq = 1 / base^(2*freq_idx/D_axis)
//     angle    = F0 * pos * inv_freq
//     result   = cos*self +/- sin*pair   (rotate_half, chunk style)
//   output = mean over 4 bases
//
//   rotate_half (chunk(2) style):
//     front_size = D_axis - D_axis/2  (ceil)
//     front half (d_local < front_size): result = cos*self - sin*pair
//       pair_local = d_local + front_size
//     back  half (d_local >= front_size): result = cos*self + sin*pair
//       pair_local = d_local - front_size
// ============================================================

#define ROPE3D_BLOCK 256
#define ROPE3D_N_BASES 4


// ============================================================
// cuRoPE3D: fused CUDA kernel for RoPE3DContinuous  (v2, fixed)
//
// rotate_half uses chunk(2, dim=-1):
//   x1 = x[..., :D//2]        size = D//2        (floor)
//   x2 = x[..., D//2:]        size = D - D//2    (ceil when odd)
//   rotate_half = cat(-x2, x1)
//
//   So for global dim d in axis of size D_axis:
//     x2_size = D_axis - D_axis/2   (integer division, ceil)
//     x1_size = D_axis / 2          (floor)
//
//     d_local < x2_size  ("front", the -x2 part):
//       rh[d_local] = -x2[d_local] = -x[d_local + x1_size]
//       pair_local  = d_local + x1_size
//       result      = cos*self - sin*pair
//
//     d_local >= x2_size  ("back", the +x1 part):
//       rh[d_local] = x1[d_local - x2_size] = x[d_local - x2_size]
//       pair_local  = d_local - x2_size
//       result      = cos*self + sin*pair
//
// freq_idx = d_local % (D_axis/2)   [same as before, correct]
// ============================================================

#define ROPE3D_BLOCK 256
#define ROPE3D_N_BASES 4

__constant__ float c_rope3d_bases[ROPE3D_N_BASES] = {10.0f, 100.0f, 1000.0f, 10000.0f};

template <typename scalar_t>
__global__ void rope_3d_cuda_kernel(
    const scalar_t* __restrict__ tokens_in,
    const float*    __restrict__ positions,
    scalar_t*       __restrict__ tokens_out,
    const int B, const int H, const int N, const int D_head,
    const float F0,
    const int D_x
) {
    const int D_xy = D_x * 2;
    const int D_z  = D_head - D_xy;

    const int bh_idx = blockIdx.x;
    const int nd_idx = (int)blockIdx.y * ROPE3D_BLOCK + threadIdx.x;

    if (bh_idx >= B * H || nd_idx >= N * D_head) return;

    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int n = nd_idx / D_head;
    const int d = nd_idx % D_head;

    // Determine axis, local dim, axis size, coordinate
    int d_local, D_axis;
    float pos;
    if (d < D_x) {
        d_local = d;        D_axis = D_x;
        pos = positions[(b * N + n) * 3 + 0];
    } else if (d < D_xy) {
        d_local = d - D_x;  D_axis = D_x;
        pos = positions[(b * N + n) * 3 + 1];
    } else {
        d_local = d - D_xy; D_axis = D_z;
        pos = positions[(b * N + n) * 3 + 2];
    }

    // chunk(2) split sizes:
    //   x1_size = D_axis / 2       (floor, the back part of rotate_half)
    //   x2_size = D_axis - x1_size (ceil,  the front part of rotate_half = -x2)
    const int x1_size = (D_axis + 1) / 2;  // ceil(D/2), chunk前半更大
    const int x2_size = D_axis / 2;          // floor(D/2), chunk后半更小

    // rotate_half = cat(-x2, x1), so:
    //   d_local < x2_size  → front (-x2 region): pair = d_local + x1_size
    //   d_local >= x2_size → back  (+x1 region): pair = d_local - x2_size
    int pair_local;
    bool is_front;
    if (d_local < x2_size) {
        pair_local = d_local + x1_size;
        is_front   = true;
    } else {
        pair_local = d_local - x2_size;
        is_front   = false;
    }
    const int pair_global = (d - d_local) + pair_local;

    // freq_idx = d_local % (D_axis/2) = d_local % x1_size
    // (when x1_size==0 that would be div-by-zero, but D_axis>=2 always here)
    const int freq_idx = d_local % ((D_axis + 1) / 2);  // ceil(D_axis/2) = len(arange(0,D,2))

    // Read self and pair
    const long base_off = (long)bh_idx * N * D_head + (long)n * D_head;
    const float q_self = (float)tokens_in[base_off + d];
    const float q_pair = (float)tokens_in[base_off + pair_global];

    // Accumulate over 4 bases
    float acc = 0.0f;
    #pragma unroll
    for (int bi = 0; bi < ROPE3D_N_BASES; bi++) {
        const float inv_freq = __powf(c_rope3d_bases[bi],
                                      -2.0f * freq_idx / (float)D_axis);
        const float angle = F0 * pos * inv_freq;
        float s, c;
        __sincosf(angle, &s, &c);
        acc += is_front ? (c * q_self - s * q_pair)
                        : (c * q_self + s * q_pair);
    }

    tokens_out[base_off + d] = (scalar_t)(acc * 0.25f);
}

void rope_3d_cuda(
    const torch::Tensor tokens_in,
    const torch::Tensor positions,
    torch::Tensor       tokens_out,
    const float F0
) {
    const int B      = tokens_in.size(0);
    const int H      = tokens_in.size(1);
    const int N      = tokens_in.size(2);
    const int D_head = tokens_in.size(3);

    TORCH_CHECK(D_head >= 3, "D_head must be >= 3");
    TORCH_CHECK(positions.dtype() == torch::kFloat32, "positions must be float32");
    TORCH_CHECK(positions.size(0)==B && positions.size(1)==N && positions.size(2)==3,
                "positions must be [B,N,3]");
    TORCH_CHECK(tokens_in.is_contiguous(),  "tokens_in must be contiguous");
    TORCH_CHECK(positions.is_contiguous(),  "positions must be contiguous");
    TORCH_CHECK(tokens_out.is_contiguous(), "tokens_out must be contiguous");

    const int D_x   = D_head / 3;
    const int total = N * D_head;
    const dim3 grid(B * H, (total + ROPE3D_BLOCK - 1) / ROPE3D_BLOCK);
    const dim3 block(ROPE3D_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens_in.scalar_type(), "rope_3d_cuda", ([&] {
        rope_3d_cuda_kernel<scalar_t><<<grid, block>>>(
            tokens_in.data_ptr<scalar_t>(),
            positions.data_ptr<float>(),
            tokens_out.data_ptr<scalar_t>(),
            B, H, N, D_head, F0, D_x
        );
    }));
}

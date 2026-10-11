#include "../include/numeric_cuda.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define NUMERIC_CUDA_CHECK(call)                                               \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                                \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static int grid_for_size(std::size_t n, int block = 256, int cap = 65535) {
    std::size_t grid = (n + static_cast<std::size_t>(block) - 1) /
                       static_cast<std::size_t>(block);
    return static_cast<int>(grid > static_cast<std::size_t>(cap) ? cap : grid);
}

template <class T>
static T* device_alloc(std::size_t count) {
    T* ptr = nullptr;
    if (count > 0) {
        NUMERIC_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    return ptr;
}

// ============================================================================
// 1. Elementwise Arithmetic & Logic
// ============================================================================

__global__ void vector_add_kernel(const float* a, const float* b, float* out,
                                  std::size_t n) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

enum class BinaryOp {
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Power
};

__global__ void binary_f32_kernel(const float* a, const float* b, float* out,
                                  std::size_t n, BinaryOp op) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        float av = a[i];
        float bv = b[i];
        if (op == BinaryOp::Subtract) {
            out[i] = av - bv;
        } else if (op == BinaryOp::Multiply) {
            out[i] = av * bv;
        } else if (op == BinaryOp::Divide) {
            out[i] = av / bv;
        } else if (op == BinaryOp::Maximum) {
            out[i] = fmaxf(av, bv);
        } else if (op == BinaryOp::Minimum) {
            out[i] = fminf(av, bv);
        } else {
            out[i] = powf(av, bv);
        }
    }
}

enum class CompareOp {
    Greater,
    Less,
    Equal
};

__global__ void compare_f32_kernel(const float* a, const float* b, bool* out,
                                   std::size_t n, CompareOp op) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (op == CompareOp::Greater) {
            out[i] = a[i] > b[i];
        } else if (op == CompareOp::Less) {
            out[i] = a[i] < b[i];
        } else {
            out[i] = a[i] == b[i];
        }
    }
}

__global__ void logical_op_kernel(const bool* a, const bool* b, bool* out,
                                  std::size_t n, int op_type) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (op_type == 0) {
            out[i] = a[i] && b[i];
        } else if (op_type == 1) {
            out[i] = a[i] || b[i];
        } else {
            out[i] = !a[i];
        }
    }
}

__global__ void where_f32_kernel(const bool* cond, const float* a,
                                 const float* b, float* out, std::size_t n) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = cond[i] ? a[i] : b[i];
    }
}

// ============================================================================
// 2. Elementwise Math & ML Activations
// ============================================================================

enum class UnaryOp {
    Exp,
    Sqrt,
    Log,
    Log1p,
    Expm1,
    Sin,
    Cos,
    Tan,
    Tanh,
    Floor,
    Ceil,
    Abs,
    Negative,
    Square,
    Sign,
    Erf,
    Erfc,
    Gamma,
    Lgamma,
    Sigmoid,
    Relu,
    Gelu,
    Silu
};

__global__ void unary_f32_kernel(const float* x, float* out, std::size_t n,
                                 UnaryOp op) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        float v = x[i];
        if (op == UnaryOp::Exp) {
            out[i] = expf(v);
        } else if (op == UnaryOp::Sqrt) {
            out[i] = sqrtf(v);
        } else if (op == UnaryOp::Log) {
            out[i] = logf(v);
        } else if (op == UnaryOp::Log1p) {
            out[i] = log1pf(v);
        } else if (op == UnaryOp::Expm1) {
            out[i] = expm1f(v);
        } else if (op == UnaryOp::Sin) {
            out[i] = sinf(v);
        } else if (op == UnaryOp::Cos) {
            out[i] = cosf(v);
        } else if (op == UnaryOp::Tan) {
            out[i] = tanf(v);
        } else if (op == UnaryOp::Tanh) {
            out[i] = tanhf(v);
        } else if (op == UnaryOp::Floor) {
            out[i] = floorf(v);
        } else if (op == UnaryOp::Ceil) {
            out[i] = ceilf(v);
        } else if (op == UnaryOp::Abs) {
            out[i] = fabsf(v);
        } else if (op == UnaryOp::Negative) {
            out[i] = -v;
        } else if (op == UnaryOp::Square) {
            out[i] = v * v;
        } else if (op == UnaryOp::Sign) {
            out[i] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
        } else if (op == UnaryOp::Erf) {
            out[i] = erff(v);
        } else if (op == UnaryOp::Erfc) {
            out[i] = erfcf(v);
        } else if (op == UnaryOp::Gamma) {
            out[i] = tgammaf(v);
        } else if (op == UnaryOp::Lgamma) {
            out[i] = lgammaf(v);
        } else if (op == UnaryOp::Sigmoid) {
            out[i] = 1.0f / (1.0f + expf(-v));
        } else if (op == UnaryOp::Relu) {
            out[i] = fmaxf(0.0f, v);
        } else if (op == UnaryOp::Gelu) {
            out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        } else if (op == UnaryOp::Silu) {
            out[i] = v / (1.0f + expf(-v));
        }
    }
}

__global__ void leaky_relu_kernel(const float* x, float* out, std::size_t n, float alpha) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        float v = x[i];
        out[i] = (v > 0.0f) ? v : alpha * v;
    }
}

__global__ void clip_f32_kernel(const float* x, float* out, std::size_t n,
                                float lo, float hi) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = fminf(fmaxf(x[i], lo), hi);
    }
}

// ============================================================================
// 3. Row-wise Softmax, Log-Softmax, LogSumExp
// ============================================================================

__global__ void softmax_2d_kernel(const float* x, float* out, int rows, int cols, int mode) {
    // mode 0: softmax, mode 1: log_softmax, mode 2: logsumexp
    int r = blockIdx.x;
    if (r >= rows) return;

    const float* row_in = x + static_cast<std::size_t>(r) * cols;
    float* row_out = (mode == 2) ? out : (out + static_cast<std::size_t>(r) * cols);

    // Step 1: Find maximum in row
    float local_max = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, row_in[c]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    __shared__ float s_max[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_max[warp] = local_max;
    }
    __syncthreads();

    float row_max = -1e30f;
    if (warp == 0) {
        row_max = (lane < (blockDim.x >> 5)) ? s_max[lane] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffffu, row_max, offset));
        }
    }
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    row_max = shared_max;

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += expf(row_in[c] - row_max);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) {
        s_sum[warp] = local_sum;
    }
    __syncthreads();

    float row_sum = 0.0f;
    if (warp == 0) {
        row_sum = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_down_sync(0xffffffffu, row_sum, offset);
        }
    }
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum;

    // Step 3: Write outputs
    if (mode == 2) {
        if (threadIdx.x == 0) {
            row_out[r] = row_max + logf(row_sum);
        }
    } else {
        float inv_sum = 1.0f / row_sum;
        float log_sum = logf(row_sum);
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            if (mode == 0) {
                row_out[c] = expf(row_in[c] - row_max) * inv_sum;
            } else {
                row_out[c] = (row_in[c] - row_max) - log_sum;
            }
        }
    }
}

// ============================================================================
// 4. Reductions: Sum, Mean, Variance, Min, Max, BCE, MSE, Norm, Dot, Trace
// ============================================================================

__global__ void sum_kernel(const float* x, double* out, std::size_t n) {
    float local = 0.0f;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, static_cast<double>(block_sum));
        }
    }
}

__global__ void squared_diff_sum_kernel(const float* x, double mean,
                                        double* out, std::size_t n) {
    double local = 0.0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        double d = static_cast<double>(x[i]) - mean;
        local += d * d;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

__global__ void bce_loss_kernel(const float* y_pred, const float* y_true, double* out,
                               std::size_t n, float eps) {
    double local = 0.0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        double p = static_cast<double>(fminf(fmaxf(y_pred[i], eps), 1.0f - eps));
        double y = static_cast<double>(y_true[i]);
        local += -(y * log(p) + (1.0 - y) * log(1.0 - p));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

__global__ void mse_loss_kernel(const float* y_pred, const float* y_true, double* out,
                               std::size_t n) {
    double local = 0.0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        double diff = static_cast<double>(y_pred[i] - y_true[i]);
        local += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

__global__ void dot_kernel(const float* a, const float* b, double* out, std::size_t n) {
    double local = 0.0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

__global__ void count_nonzero_kernel(const float* x, unsigned long long* out, std::size_t n) {
    unsigned long long local = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += (x[i] != 0.0f);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ unsigned long long warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

__global__ void bool_reduce_kernel(const bool* x, int* out_count, std::size_t n) {
    int local = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += (x[i] ? 1 : 0);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ int warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        int block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out_count, block_sum);
        }
    }
}

// ============================================================================
// 5. Matrix Multiplications, Transpose, Outer, MatVec
// ============================================================================

#define MATMUL_TILE 16
__global__ void matmul_kernel(const float* a, const float* b, float* c, int m,
                              int n, int k) {
    __shared__ float as[MATMUL_TILE][MATMUL_TILE];
    __shared__ float bs[MATMUL_TILE][MATMUL_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * MATMUL_TILE + ty;
    int col = blockIdx.x * MATMUL_TILE + tx;
    float acc = 0.0f;

    for (int tile = 0; tile < (k + MATMUL_TILE - 1) / MATMUL_TILE; ++tile) {
        int a_col = tile * MATMUL_TILE + tx;
        int b_row = tile * MATMUL_TILE + ty;
        as[ty][tx] = (row < m && a_col < k) ? a[static_cast<std::size_t>(row) * k + a_col] : 0.0f;
        bs[ty][tx] = (b_row < k && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MATMUL_TILE; ++i) {
            acc += as[ty][i] * bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = acc;
    }
}

__global__ void transpose_kernel(const float* in, float* out, int rows,
                                 int cols) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] =
                in[static_cast<std::size_t>(y + j) * cols + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) {
        if (x < rows && y + j < cols) {
            out[static_cast<std::size_t>(y + j) * rows + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void outer_kernel(const float* a, const float* b, float* out, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        out[static_cast<std::size_t>(row) * n + col] = a[row] * b[col];
    }
}

__global__ void matvec_kernel(const float* a, const float* x, float* out, int m, int k) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row < m) {
        float sum = 0.0f;
        for (int c = threadIdx.x; c < k; c += blockDim.x) {
            sum += a[static_cast<std::size_t>(row) * k + c] * x[c];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (threadIdx.x == 0) {
            out[row] = sum;
        }
    }
}

// ============================================================================
// 6. Spatial Distance Metrics (cdist)
// ============================================================================

__global__ void cdist_kernel(const float* xa, const float* xb, float* out,
                             int m, int n, int d, int metric) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        const float* a_ptr = xa + static_cast<std::size_t>(row) * d;
        const float* b_ptr = xb + static_cast<std::size_t>(col) * d;
        float dist = 0.0f;

        if (metric == 0 || metric == 1) { // 0: euclidean, 1: sqeuclidean
            for (int k = 0; k < d; ++k) {
                float diff = a_ptr[k] - b_ptr[k];
                dist += diff * diff;
            }
            if (metric == 0) {
                dist = sqrtf(dist);
            }
        } else if (metric == 2) { // 2: cosine
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (int k = 0; k < d; ++k) {
                float va = a_ptr[k];
                float vb = b_ptr[k];
                dot += va * vb;
                norm_a += va * va;
                norm_b += vb * vb;
            }
            float denom = sqrtf(norm_a) * sqrtf(norm_b);
            dist = (denom > 1e-8f) ? fmaxf(0.0f, 1.0f - (dot / denom)) : 0.0f;
        } else if (metric == 3) { // 3: cityblock / Manhattan
            for (int k = 0; k < d; ++k) {
                dist += fabsf(a_ptr[k] - b_ptr[k]);
            }
        }
        out[static_cast<std::size_t>(row) * n + col] = dist;
    }
}

// ============================================================================
// 7. Searchsorted, Diff, Trapezoid
// ============================================================================

__global__ void searchsorted_kernel(const float* sorted_arr, int n_sorted,
                                    const float* values, int* out_indices,
                                    int n_val, bool right) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_val; i += gridDim.x * blockDim.x) {
        float val = values[i];
        int lo = 0;
        int hi = n_sorted;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            float mid_val = sorted_arr[mid];
            if (right) {
                if (val >= mid_val) lo = mid + 1;
                else hi = mid;
            } else {
                if (val <= mid_val) hi = mid;
                else lo = mid + 1;
            }
        }
        out_indices[i] = lo;
    }
}

__global__ void diff_kernel(const float* in, float* out, int n_out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_out; i += gridDim.x * blockDim.x) {
        out[i] = in[i + 1] - in[i];
    }
}

__global__ void trapezoid_kernel(const float* y, const float* x, double* out,
                                 int n_segs, float dx) {
    double local = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_segs; i += gridDim.x * blockDim.x) {
        float width = (x != nullptr) ? (x[i + 1] - x[i]) : dx;
        local += 0.5 * static_cast<double>(y[i] + y[i + 1]) * static_cast<double>(width);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    if (warp == 0) {
        double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

// ============================================================================
// 8. K-Means Clustering on GPU
// ============================================================================

__global__ void kmeans_assign_kernel(const float* x, const float* centroids,
                                     int* labels, double* out_inertia,
                                     int n_samples, int n_features, int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_samples; i += gridDim.x * blockDim.x) {
        const float* sample = x + static_cast<std::size_t>(i) * n_features;
        float min_dist = 1e30f;
        int best_c = 0;

        for (int c = 0; c < k; ++c) {
            const float* cent = centroids + static_cast<std::size_t>(c) * n_features;
            float dist = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float d = sample[f] - cent[f];
                dist += d * d;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_c = c;
            }
        }
        labels[i] = best_c;
        if (out_inertia != nullptr) {
            atomicAdd(out_inertia, static_cast<double>(min_dist));
        }
    }
}

__global__ void kmeans_accumulate_kernel(const float* x, const int* labels,
                                         float* new_centroids, int* counts,
                                         int n_samples, int n_features) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_samples; i += gridDim.x * blockDim.x) {
        int c = labels[i];
        atomicAdd(&counts[c], 1);
        const float* sample = x + static_cast<std::size_t>(i) * n_features;
        float* cent_out = new_centroids + static_cast<std::size_t>(c) * n_features;
        for (int f = 0; f < n_features; ++f) {
            atomicAdd(&cent_out[f], sample[f]);
        }
    }
}

__global__ void kmeans_normalize_kernel(float* centroids, const int* counts,
                                        int k, int n_features) {
    int c = blockIdx.x;
    if (c >= k) return;
    int count = counts[c];
    float inv_count = (count > 0) ? (1.0f / static_cast<float>(count)) : 1.0f;
    float* cent = centroids + static_cast<std::size_t>(c) * n_features;
    for (int f = threadIdx.x; f < n_features; f += blockDim.x) {
        cent[f] *= inv_count;
    }
}

// ============================================================================
// 9. Symmetric / Positive Definite Linear Solver (Gauss-Jordan)
// ============================================================================

__global__ void gauss_jordan_pivot_scale_kernel(float* a_aug, int n, int pivot) {
    float pivot_val = a_aug[static_cast<std::size_t>(pivot) * (n + 1) + pivot];
    float inv = (fabsf(pivot_val) > 1e-12f) ? (1.0f / pivot_val) : 0.0f;
    for (int c = threadIdx.x; c <= n; c += blockDim.x) {
        a_aug[static_cast<std::size_t>(pivot) * (n + 1) + c] *= inv;
    }
}

__global__ void gauss_jordan_eliminate_kernel(float* a_aug, int n, int pivot) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row < n && row != pivot) {
        float factor = a_aug[static_cast<std::size_t>(row) * (n + 1) + pivot];
        for (int c = threadIdx.x; c <= n; c += blockDim.x) {
            float pivot_row_val = a_aug[static_cast<std::size_t>(pivot) * (n + 1) + c];
            a_aug[static_cast<std::size_t>(row) * (n + 1) + c] -= factor * pivot_row_val;
        }
    }
}

// ============================================================================
// 10. Existing Bincount & Correlation
// ============================================================================

__global__ void bincount_u8_kernel(const unsigned char* x, unsigned int* out,
                                   std::size_t n) {
    __shared__ unsigned int local[256];
    if (threadIdx.x < 256) {
        local[threadIdx.x] = 0;
    }
    __syncthreads();

    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                         threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        atomicAdd(&local[x[i]], 1u);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&out[threadIdx.x], local[threadIdx.x]);
    }
}

__constant__ float c_filter_5x5[25];

__global__ void correlate2d_5x5_kernel(const float* image, float* out,
                                       int height, int width) {
    constexpr int radius = 2;
    constexpr int tile = 16;
    __shared__ float shared[tile + 2 * radius][tile + 2 * radius];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ox = blockIdx.x * tile + tx;
    int oy = blockIdx.y * tile + ty;

    for (int dy = ty; dy < tile + 2 * radius; dy += tile) {
        for (int dx = tx; dx < tile + 2 * radius; dx += tile) {
            int gy = blockIdx.y * tile + dy - radius;
            int gx = blockIdx.x * tile + dx - radius;
            shared[dy][dx] = (gy >= 0 && gy < height && gx >= 0 && gx < width)
                                 ? image[static_cast<std::size_t>(gy) * width + gx]
                                 : 0.0f;
        }
    }
    __syncthreads();

    if (ox < width && oy < height) {
        float acc = 0.0f;
#pragma unroll
        for (int fy = 0; fy < 5; ++fy) {
#pragma unroll
            for (int fx = 0; fx < 5; ++fx) {
                acc += c_filter_5x5[fy * 5 + fx] * shared[ty + fy][tx + fx];
            }
        }
        out[static_cast<std::size_t>(oy) * width + ox] = acc;
    }
}

// ============================================================================
// Host Entry Points Implementation
// ============================================================================

void cuda_vector_add_f32(const float* a, const float* b, float* out,
                         std::size_t n) {
    float* da = device_alloc<float>(n);
    float* db = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    vector_add_kernel<<<grid_for_size(n), 256>>>(da, db, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

static void cuda_binary_f32(const float* a, const float* b, float* out,
                            std::size_t n, BinaryOp op) {
    float* da = device_alloc<float>(n);
    float* db = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    binary_f32_kernel<<<grid_for_size(n), 256>>>(da, db, dout, n, op);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

void cuda_vector_subtract_f32(const float* a, const float* b, float* out,
                              std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Subtract);
}

void cuda_vector_multiply_f32(const float* a, const float* b, float* out,
                              std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Multiply);
}

void cuda_vector_divide_f32(const float* a, const float* b, float* out,
                            std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Divide);
}

void cuda_vector_maximum_f32(const float* a, const float* b, float* out,
                             std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Maximum);
}

void cuda_vector_minimum_f32(const float* a, const float* b, float* out,
                             std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Minimum);
}

void cuda_vector_power_f32(const float* a, const float* b, float* out,
                           std::size_t n) {
    cuda_binary_f32(a, b, out, n, BinaryOp::Power);
}

static void cuda_compare_f32(const float* a, const float* b, bool* out,
                             std::size_t n, CompareOp op) {
    float* da = device_alloc<float>(n);
    float* db = device_alloc<float>(n);
    bool* dout = device_alloc<bool>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    compare_f32_kernel<<<grid_for_size(n), 256>>>(da, db, dout, n, op);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

void cuda_greater_f32(const float* a, const float* b, bool* out,
                      std::size_t n) {
    cuda_compare_f32(a, b, out, n, CompareOp::Greater);
}

void cuda_less_f32(const float* a, const float* b, bool* out, std::size_t n) {
    cuda_compare_f32(a, b, out, n, CompareOp::Less);
}

void cuda_equal_f32(const float* a, const float* b, bool* out, std::size_t n) {
    cuda_compare_f32(a, b, out, n, CompareOp::Equal);
}

static void cuda_logical_op(const bool* a, const bool* b, bool* out, std::size_t n, int op) {
    bool* da = device_alloc<bool>(n);
    bool* db = (b != nullptr) ? device_alloc<bool>(n) : nullptr;
    bool* dout = device_alloc<bool>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(bool), cudaMemcpyHostToDevice));
    if (b != nullptr) {
        NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(bool), cudaMemcpyHostToDevice));
    }
    logical_op_kernel<<<grid_for_size(n), 256>>>(da, db, dout, n, op);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(da);
    if (db != nullptr) cudaFree(db);
    cudaFree(dout);
}

void cuda_logical_and_f32(const bool* a, const bool* b, bool* out, std::size_t n) {
    cuda_logical_op(a, b, out, n, 0);
}

void cuda_logical_or_f32(const bool* a, const bool* b, bool* out, std::size_t n) {
    cuda_logical_op(a, b, out, n, 1);
}

void cuda_logical_not_f32(const bool* a, bool* out, std::size_t n) {
    cuda_logical_op(a, nullptr, out, n, 2);
}

void cuda_where_f32(const bool* cond, const float* a, const float* b,
                    float* out, std::size_t n) {
    bool* dcond = device_alloc<bool>(n);
    float* da = device_alloc<float>(n);
    float* db = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dcond, cond, n * sizeof(bool), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    where_f32_kernel<<<grid_for_size(n), 256>>>(dcond, da, db, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dcond);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

static void cuda_unary_f32(const float* x, float* out, std::size_t n,
                           UnaryOp op) {
    float* dx = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    unary_f32_kernel<<<grid_for_size(n), 256>>>(dx, dout, n, op);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
}

void cuda_exp_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Exp); }
void cuda_sqrt_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Sqrt); }
void cuda_log_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Log); }
void cuda_log1p_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Log1p); }
void cuda_expm1_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Expm1); }
void cuda_sin_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Sin); }
void cuda_cos_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Cos); }
void cuda_tan_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Tan); }
void cuda_tanh_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Tanh); }
void cuda_floor_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Floor); }
void cuda_ceil_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Ceil); }
void cuda_abs_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Abs); }
void cuda_negative_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Negative); }
void cuda_square_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Square); }
void cuda_sign_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Sign); }
void cuda_erf_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Erf); }
void cuda_erfc_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Erfc); }
void cuda_gamma_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Gamma); }
void cuda_lgamma_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Lgamma); }

void cuda_sigmoid_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Sigmoid); }
void cuda_relu_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Relu); }
void cuda_gelu_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Gelu); }
void cuda_silu_f32(const float* x, float* out, std::size_t n) { cuda_unary_f32(x, out, n, UnaryOp::Silu); }

void cuda_leaky_relu_f32(const float* x, float* out, std::size_t n, float alpha) {
    float* dx = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    leaky_relu_kernel<<<grid_for_size(n), 256>>>(dx, dout, n, alpha);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
}

void cuda_clip_f32(const float* x, float* out, std::size_t n, float lo,
                   float hi) {
    float* dx = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    clip_f32_kernel<<<grid_for_size(n), 256>>>(dx, dout, n, lo, hi);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
}

static void cuda_softmax_2d_mode(const float* x, float* out, int rows, int cols, int mode) {
    std::size_t in_size = static_cast<std::size_t>(rows) * cols;
    std::size_t out_size = (mode == 2) ? static_cast<std::size_t>(rows) : in_size;
    float* dx = device_alloc<float>(in_size);
    float* dout = device_alloc<float>(out_size);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, in_size * sizeof(float), cudaMemcpyHostToDevice));
    int block_dim = (cols < 256) ? ((cols + 31) / 32 * 32) : 256;
    if (block_dim < 32) block_dim = 32;
    softmax_2d_kernel<<<rows, block_dim>>>(dx, dout, rows, cols, mode);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
}

void cuda_softmax_2d_f32(const float* x, float* out, int rows, int cols) {
    cuda_softmax_2d_mode(x, out, rows, cols, 0);
}

void cuda_log_softmax_2d_f32(const float* x, float* out, int rows, int cols) {
    cuda_softmax_2d_mode(x, out, rows, cols, 1);
}

void cuda_logsumexp_2d_f32(const float* x, float* out, int rows, int cols) {
    cuda_softmax_2d_mode(x, out, rows, cols, 2);
}

double cuda_binary_cross_entropy_f32(const float* y_pred, const float* y_true, std::size_t n, float eps) {
    float* dp = device_alloc<float>(n);
    float* dy = device_alloc<float>(n);
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dp, y_pred, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dy, y_true, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    bce_loss_kernel<<<grid_for_size(n), 256>>>(dp, dy, dout, n, eps);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double total = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dp);
    cudaFree(dy);
    cudaFree(dout);
    return total / static_cast<double>(n);
}

double cuda_mse_loss_f32(const float* y_pred, const float* y_true, std::size_t n) {
    float* dp = device_alloc<float>(n);
    float* dy = device_alloc<float>(n);
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dp, y_pred, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dy, y_true, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    mse_loss_kernel<<<grid_for_size(n), 256>>>(dp, dy, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double total = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dp);
    cudaFree(dy);
    cudaFree(dout);
    return total / static_cast<double>(n);
}

double cuda_sum_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    sum_kernel<<<grid_for_size(n), 256>>>(dx, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double total = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
    return total;
}

float cuda_min_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> dptr(dx);
    auto it = thrust::min_element(dptr, dptr + n);
    float val = *it;
    cudaFree(dx);
    return val;
}

float cuda_max_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> dptr(dx);
    auto it = thrust::max_element(dptr, dptr + n);
    float val = *it;
    cudaFree(dx);
    return val;
}

double cuda_mean_f32(const float* x, std::size_t n) {
    return cuda_sum_f32(x, n) / static_cast<double>(n);
}

double cuda_var_f32(const float* x, std::size_t n, int ddof) {
    double m = cuda_mean_f32(x, n);
    float* dx = device_alloc<float>(n);
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    squared_diff_sum_kernel<<<grid_for_size(n), 256>>>(dx, m, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double ss = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&ss, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
    return ss / static_cast<double>(static_cast<long long>(n) - ddof);
}

std::size_t cuda_argmin_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> dptr(dx);
    auto it = thrust::min_element(dptr, dptr + n);
    std::size_t idx = static_cast<std::size_t>(it - dptr);
    cudaFree(dx);
    return idx;
}

std::size_t cuda_argmax_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> dptr(dx);
    auto it = thrust::max_element(dptr, dptr + n);
    std::size_t idx = static_cast<std::size_t>(it - dptr);
    cudaFree(dx);
    return idx;
}

std::size_t cuda_count_nonzero_f32(const float* x, std::size_t n) {
    float* dx = device_alloc<float>(n);
    unsigned long long* dout = device_alloc<unsigned long long>(1);
    unsigned long long zero = 0ULL;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    count_nonzero_kernel<<<grid_for_size(n), 256>>>(dx, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    unsigned long long count = 0ULL;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&count, dout, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
    return static_cast<std::size_t>(count);
}

bool cuda_all_f32(const bool* x, std::size_t n) {
    bool* dx = device_alloc<bool>(n);
    int* dout = device_alloc<int>(1);
    int zero = 0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(bool), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(int), cudaMemcpyHostToDevice));
    bool_reduce_kernel<<<grid_for_size(n), 256>>>(dx, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    int true_count = 0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&true_count, dout, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
    return (static_cast<std::size_t>(true_count) == n);
}

bool cuda_any_f32(const bool* x, std::size_t n) {
    bool* dx = device_alloc<bool>(n);
    int* dout = device_alloc<int>(1);
    int zero = 0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(bool), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(int), cudaMemcpyHostToDevice));
    bool_reduce_kernel<<<grid_for_size(n), 256>>>(dx, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    int true_count = 0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&true_count, dout, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
    return (true_count > 0);
}

double cuda_dot_f32(const float* a, const float* b, std::size_t n) {
    float* da = device_alloc<float>(n);
    float* db = device_alloc<float>(n);
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    dot_kernel<<<grid_for_size(n), 256>>>(da, db, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double total = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return total;
}

double cuda_norm_f32(const float* x, std::size_t n, int ord) {
    if (ord == 1) {
        float* dx = device_alloc<float>(n);
        float* dabs = device_alloc<float>(n);
        NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
        unary_f32_kernel<<<grid_for_size(n), 256>>>(dx, dabs, n, UnaryOp::Abs);
        double* dout = device_alloc<double>(1);
        double zero = 0.0;
        NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
        sum_kernel<<<grid_for_size(n), 256>>>(dabs, dout, n);
        double total = 0.0;
        NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dx);
        cudaFree(dabs);
        cudaFree(dout);
        return total;
    } else if (ord == 0) { // inf norm
        float* dx = device_alloc<float>(n);
        float* dabs = device_alloc<float>(n);
        NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
        unary_f32_kernel<<<grid_for_size(n), 256>>>(dx, dabs, n, UnaryOp::Abs);
        thrust::device_ptr<float> dptr(dabs);
        auto it = thrust::max_element(dptr, dptr + n);
        float val = *it;
        cudaFree(dx);
        cudaFree(dabs);
        return static_cast<double>(val);
    } else { // default L2 / Frobenius
        return std::sqrt(cuda_dot_f32(x, x, n));
    }
}

double cuda_trace_f32(const float* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[static_cast<std::size_t>(i) * n + i]);
    }
    return sum;
}

void cuda_outer_f32(const float* a, const float* b, float* out, int m, int n) {
    float* da = device_alloc<float>(m);
    float* db = device_alloc<float>(n);
    std::size_t out_count = static_cast<std::size_t>(m) * n;
    float* dout = device_alloc<float>(out_count);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    outer_kernel<<<grid, block>>>(da, db, dout, m, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

void cuda_matvec_f32(const float* a, const float* x, float* out, int m, int k) {
    std::size_t a_count = static_cast<std::size_t>(m) * k;
    float* da = device_alloc<float>(a_count);
    float* dx = device_alloc<float>(k);
    float* dout = device_alloc<float>(m);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, a_count * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, k * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 8);
    dim3 grid((m + 7) / 8);
    matvec_kernel<<<grid, block>>>(da, dx, dout, m, k);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, m * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(dx);
    cudaFree(dout);
}

void cuda_matmul_f32(const float* a, const float* b, float* c, int m, int n,
                     int k) {
    float* da = device_alloc<float>(static_cast<std::size_t>(m) * k);
    float* db = device_alloc<float>(static_cast<std::size_t>(k) * n);
    float* dc = device_alloc<float>(static_cast<std::size_t>(m) * n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(da, a, static_cast<std::size_t>(m) * k * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(db, b, static_cast<std::size_t>(k) * n * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(MATMUL_TILE, MATMUL_TILE);
    dim3 grid((n + MATMUL_TILE - 1) / MATMUL_TILE,
              (m + MATMUL_TILE - 1) / MATMUL_TILE);
    matmul_kernel<<<grid, block>>>(da, db, dc, m, n, k);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(c, dc, static_cast<std::size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void cuda_transpose_f32(const float* in, float* out, int rows, int cols) {
    std::size_t count = static_cast<std::size_t>(rows) * cols;
    float* din = device_alloc<float>(count);
    float* dout = device_alloc<float>(count);
    NUMERIC_CUDA_CHECK(cudaMemcpy(din, in, count * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 8);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_kernel<<<grid, block>>>(din, dout, rows, cols);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, count * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(din);
    cudaFree(dout);
}

void cuda_cov_f32(const float* x, float* out, int n_samples, int n_features) {
    // 1. Compute means
    std::vector<float> means(n_features, 0.0f);
    for (int j = 0; j < n_features; ++j) {
        double col_sum = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            col_sum += x[static_cast<std::size_t>(i) * n_features + j];
        }
        means[j] = static_cast<float>(col_sum / n_samples);
    }
    // 2. Center X
    std::vector<float> x_centered(static_cast<std::size_t>(n_samples) * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            x_centered[static_cast<std::size_t>(i) * n_features + j] =
                x[static_cast<std::size_t>(i) * n_features + j] - means[j];
        }
    }
    // 3. Transpose x_centered to (n_features, n_samples)
    std::vector<float> x_t(static_cast<std::size_t>(n_features) * n_samples);
    cuda_transpose_f32(x_centered.data(), x_t.data(), n_samples, n_features);

    // 4. Multiply x_t @ x_centered -> (n_features, n_features)
    std::vector<float> cov_mat(static_cast<std::size_t>(n_features) * n_features);
    cuda_matmul_f32(x_t.data(), x_centered.data(), cov_mat.data(), n_features, n_features, n_samples);

    // 5. Scale by 1 / (n_samples - 1)
    float factor = 1.0f / static_cast<float>(n_samples - 1);
    for (std::size_t idx = 0; idx < cov_mat.size(); ++idx) {
        out[idx] = cov_mat[idx] * factor;
    }
}

void cuda_corrcoef_f32(const float* x, float* out, int n_samples, int n_features) {
    std::vector<float> cov(static_cast<std::size_t>(n_features) * n_features);
    cuda_cov_f32(x, cov.data(), n_samples, n_features);

    for (int i = 0; i < n_features; ++i) {
        float var_i = cov[static_cast<std::size_t>(i) * n_features + i];
        float std_i = std::sqrt(std::max(var_i, 1e-12f));
        for (int j = 0; j < n_features; ++j) {
            float var_j = cov[static_cast<std::size_t>(j) * n_features + j];
            float std_j = std::sqrt(std::max(var_j, 1e-12f));
            out[static_cast<std::size_t>(i) * n_features + j] =
                cov[static_cast<std::size_t>(i) * n_features + j] / (std_i * std_j);
        }
    }
}

void cuda_solve_symmetric_f32(const float* a, const float* b, float* out, int n) {
    // Augment [A | b] of size n x (n + 1)
    std::vector<float> aug(static_cast<std::size_t>(n) * (n + 1));
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            aug[static_cast<std::size_t>(r) * (n + 1) + c] = a[static_cast<std::size_t>(r) * n + c];
        }
        aug[static_cast<std::size_t>(r) * (n + 1) + n] = b[r];
    }
    float* daug = device_alloc<float>(aug.size());
    NUMERIC_CUDA_CHECK(cudaMemcpy(daug, aug.data(), aug.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int p = 0; p < n; ++p) {
        int block_threads = ((n + 1) < 256) ? (n + 1) : 256;
        gauss_jordan_pivot_scale_kernel<<<1, block_threads>>>(daug, n, p);
        NUMERIC_CUDA_CHECK(cudaGetLastError());
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        gauss_jordan_eliminate_kernel<<<grid, block>>>(daug, n, p);
        NUMERIC_CUDA_CHECK(cudaGetLastError());
    }

    NUMERIC_CUDA_CHECK(cudaMemcpy(aug.data(), daug, aug.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(daug);
    for (int r = 0; r < n; ++r) {
        out[r] = aug[static_cast<std::size_t>(r) * (n + 1) + n];
    }
}

void cuda_cdist_f32(const float* xa, const float* xb, float* out, int m, int n, int d, int metric) {
    float* dxa = device_alloc<float>(static_cast<std::size_t>(m) * d);
    float* dxb = device_alloc<float>(static_cast<std::size_t>(n) * d);
    float* dout = device_alloc<float>(static_cast<std::size_t>(m) * n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dxa, xa, static_cast<std::size_t>(m) * d * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dxb, xb, static_cast<std::size_t>(n) * d * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    cdist_kernel<<<grid, block>>>(dxa, dxb, dout, m, n, d, metric);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, static_cast<std::size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dxa);
    cudaFree(dxb);
    cudaFree(dout);
}

void cuda_searchsorted_f32(const float* sorted_arr, int n_sorted, const float* values, int* out_indices, int n_val, bool right) {
    float* ds = device_alloc<float>(n_sorted);
    float* dv = device_alloc<float>(n_val);
    int* dout = device_alloc<int>(n_val);
    NUMERIC_CUDA_CHECK(cudaMemcpy(ds, sorted_arr, n_sorted * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dv, values, n_val * sizeof(float), cudaMemcpyHostToDevice));
    searchsorted_kernel<<<grid_for_size(n_val), 256>>>(ds, n_sorted, dv, dout, n_val, right);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out_indices, dout, n_val * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(ds);
    cudaFree(dv);
    cudaFree(dout);
}

void cuda_diff_f32(const float* in, float* out, int n) {
    if (n <= 1) return;
    int n_out = n - 1;
    float* din = device_alloc<float>(n);
    float* dout = device_alloc<float>(n_out);
    NUMERIC_CUDA_CHECK(cudaMemcpy(din, in, n * sizeof(float), cudaMemcpyHostToDevice));
    diff_kernel<<<grid_for_size(n_out), 256>>>(din, dout, n_out);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(din);
    cudaFree(dout);
}

double cuda_trapezoid_f32(const float* y, const float* x, int n, float dx) {
    if (n <= 1) return 0.0;
    int n_segs = n - 1;
    float* dy = device_alloc<float>(n);
    float* dx_arr = (x != nullptr) ? device_alloc<float>(n) : nullptr;
    double* dout = device_alloc<double>(1);
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice));
    if (x != nullptr) {
        NUMERIC_CUDA_CHECK(cudaMemcpy(dx_arr, x, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    NUMERIC_CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(double), cudaMemcpyHostToDevice));
    trapezoid_kernel<<<grid_for_size(n_segs), 256>>>(dy, dx_arr, dout, n_segs, dx);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    double total = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(&total, dout, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dy);
    if (dx_arr != nullptr) cudaFree(dx_arr);
    cudaFree(dout);
    return total;
}

void cuda_kmeans_fit_f32(const float* x, float* centroids, int* labels, float* out_inertia,
                         int n_samples, int n_features, int k, int max_iters, float tol, unsigned int seed) {
    std::size_t x_count = static_cast<std::size_t>(n_samples) * n_features;
    std::size_t c_count = static_cast<std::size_t>(k) * n_features;

    // Initialize centroids on host from data samples (deterministic stride based on seed)
    for (int c = 0; c < k; ++c) {
        int sample_idx = (static_cast<int>(seed) + c * (n_samples / k)) % n_samples;
        std::memcpy(centroids + static_cast<std::size_t>(c) * n_features,
                    x + static_cast<std::size_t>(sample_idx) * n_features,
                    n_features * sizeof(float));
    }

    float* dx = device_alloc<float>(x_count);
    float* dc = device_alloc<float>(c_count);
    float* d_new_c = device_alloc<float>(c_count);
    int* d_labels = device_alloc<int>(n_samples);
    int* d_counts = device_alloc<int>(k);
    double* d_inertia = device_alloc<double>(1);

    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, x_count * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpy(dc, centroids, c_count * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> old_c(c_count);

    for (int iter = 0; iter < max_iters; ++iter) {
        NUMERIC_CUDA_CHECK(cudaMemset(d_new_c, 0, c_count * sizeof(float)));
        NUMERIC_CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));
        double zero = 0.0;
        NUMERIC_CUDA_CHECK(cudaMemcpy(d_inertia, &zero, sizeof(double), cudaMemcpyHostToDevice));

        // Assign samples to nearest centroids
        kmeans_assign_kernel<<<grid_for_size(n_samples), 256>>>(dx, dc, d_labels, d_inertia, n_samples, n_features, k);
        NUMERIC_CUDA_CHECK(cudaGetLastError());

        // Accumulate new centroid sums
        kmeans_accumulate_kernel<<<grid_for_size(n_samples), 256>>>(dx, d_labels, d_new_c, d_counts, n_samples, n_features);
        NUMERIC_CUDA_CHECK(cudaGetLastError());

        // Normalize
        int block_threads = (n_features < 256) ? n_features : 256;
        if (block_threads < 1) block_threads = 1;
        kmeans_normalize_kernel<<<k, block_threads>>>(d_new_c, d_counts, k, n_features);
        NUMERIC_CUDA_CHECK(cudaGetLastError());

        // Copy new centroids to check shift
        NUMERIC_CUDA_CHECK(cudaMemcpy(old_c.data(), dc, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        NUMERIC_CUDA_CHECK(cudaMemcpy(centroids, d_new_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        NUMERIC_CUDA_CHECK(cudaMemcpy(dc, d_new_c, c_count * sizeof(float), cudaMemcpyDeviceToDevice));

        float max_shift = 0.0f;
        for (std::size_t i = 0; i < c_count; ++i) {
            float diff = std::fabs(centroids[i] - old_c[i]);
            if (diff > max_shift) max_shift = diff;
        }
        if (max_shift < tol) {
            break;
        }
    }

    // Final labels and inertia
    double final_inertia = 0.0;
    double zero = 0.0;
    NUMERIC_CUDA_CHECK(cudaMemcpy(d_inertia, &zero, sizeof(double), cudaMemcpyHostToDevice));
    kmeans_assign_kernel<<<grid_for_size(n_samples), 256>>>(dx, dc, d_labels, d_inertia, n_samples, n_features, k);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(&final_inertia, d_inertia, sizeof(double), cudaMemcpyDeviceToHost));
    NUMERIC_CUDA_CHECK(cudaMemcpy(labels, d_labels, n_samples * sizeof(int), cudaMemcpyDeviceToHost));
    NUMERIC_CUDA_CHECK(cudaMemcpy(centroids, dc, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    if (out_inertia != nullptr) {
        *out_inertia = static_cast<float>(final_inertia);
    }

    cudaFree(dx);
    cudaFree(dc);
    cudaFree(d_new_c);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(d_inertia);
}

void cuda_pca_fit_transform_f32(const float* x, float* components, float* explained_var, float* x_trans,
                                int n_samples, int n_features, int n_components, int max_iters, float tol) {
    // 1. Center data
    std::vector<float> means(n_features, 0.0f);
    for (int j = 0; j < n_features; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            sum += x[static_cast<std::size_t>(i) * n_features + j];
        }
        means[j] = static_cast<float>(sum / n_samples);
    }
    std::vector<float> x_c(static_cast<std::size_t>(n_samples) * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            x_c[static_cast<std::size_t>(i) * n_features + j] =
                x[static_cast<std::size_t>(i) * n_features + j] - means[j];
        }
    }

    // 2. Covariance matrix on GPU
    std::vector<float> cov(static_cast<std::size_t>(n_features) * n_features);
    cuda_cov_f32(x, cov.data(), n_samples, n_features);

    // 3. Power iteration with deflation for each component
    for (int c = 0; c < n_components; ++c) {
        std::vector<float> v(n_features, 1.0f / std::sqrt(static_cast<float>(n_features)));
        float lambda = 0.0f;

        for (int it = 0; it < max_iters; ++it) {
            std::vector<float> v_next(n_features, 0.0f);
            cuda_matvec_f32(cov.data(), v.data(), v_next.data(), n_features, n_features);
            double norm = cuda_norm_f32(v_next.data(), n_features, 2);
            lambda = static_cast<float>(norm);
            if (lambda < 1e-12f) break;
            float inv_norm = 1.0f / lambda;
            float max_diff = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float v_new = v_next[f] * inv_norm;
                max_diff = std::max(max_diff, std::fabs(v_new - v[f]));
                v[f] = v_new;
            }
            if (max_diff < tol) break;
        }

        explained_var[c] = lambda;
        std::memcpy(components + static_cast<std::size_t>(c) * n_features, v.data(), n_features * sizeof(float));

        // Deflate covariance: Cov = Cov - lambda * (v @ v^T)
        for (int r = 0; r < n_features; ++r) {
            for (int col = 0; col < n_features; ++col) {
                cov[static_cast<std::size_t>(r) * n_features + col] -= lambda * v[r] * v[col];
            }
        }
    }

    // 4. Project: x_trans = x_c @ components^T -> (n_samples, n_components)
    std::vector<float> comp_t(static_cast<std::size_t>(n_features) * n_components);
    cuda_transpose_f32(components, comp_t.data(), n_components, n_features);
    cuda_matmul_f32(x_c.data(), comp_t.data(), x_trans, n_samples, n_components, n_features);
}

void cuda_logistic_regression_fit_f32(const float* x, const float* y, float* w, float* b_out,
                                      int n_samples, int n_features, float lr, int iters, float reg) {
    std::vector<float> weights(n_features, 0.0f);
    float bias = 0.0f;

    std::vector<float> z(n_samples);
    std::vector<float> p(n_samples);
    std::vector<float> err(n_samples);
    std::vector<float> grad_w(n_features);
    std::vector<float> x_t(static_cast<std::size_t>(n_features) * n_samples);

    cuda_transpose_f32(x, x_t.data(), n_samples, n_features);

    for (int iter = 0; iter < iters; ++iter) {
        // z = X @ w
        cuda_matvec_f32(x, weights.data(), z.data(), n_samples, n_features);
        for (int i = 0; i < n_samples; ++i) {
            z[i] += bias;
        }
        // p = sigmoid(z)
        cuda_sigmoid_f32(z.data(), p.data(), n_samples);
        // err = p - y
        for (int i = 0; i < n_samples; ++i) {
            err[i] = p[i] - y[i];
        }
        // grad_w = (1 / n) * X^T @ err + reg * w
        cuda_matvec_f32(x_t.data(), err.data(), grad_w.data(), n_features, n_samples);
        float inv_n = 1.0f / static_cast<float>(n_samples);
        double err_sum = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            err_sum += err[i];
        }
        float grad_b = static_cast<float>(err_sum * inv_n);

        for (int f = 0; f < n_features; ++f) {
            float g = grad_w[f] * inv_n + reg * weights[f];
            weights[f] -= lr * g;
        }
        bias -= lr * grad_b;
    }

    std::memcpy(w, weights.data(), n_features * sizeof(float));
    *b_out = bias;
}

// Custom Functor to avoid deprecated thrust::multiplies warning
struct DeviceMultiplyFloat {
    __host__ __device__ float operator()(float a, float b) const {
        return a * b;
    }
};

void cuda_cumsum_f32(const float* in, float* out, int n) {
    float* din = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(din, in, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> d_in_ptr(din);
    thrust::device_ptr<float> d_out_ptr(dout);
    thrust::inclusive_scan(d_in_ptr, d_in_ptr + n, d_out_ptr);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(din);
    cudaFree(dout);
}

void cuda_cumprod_f32(const float* in, float* out, int n) {
    float* din = device_alloc<float>(n);
    float* dout = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(din, in, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> d_in_ptr(din);
    thrust::device_ptr<float> d_out_ptr(dout);
    thrust::inclusive_scan(d_in_ptr, d_in_ptr + n, d_out_ptr, DeviceMultiplyFloat());
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(din);
    cudaFree(dout);
}

void cuda_bincount_u8_256(const unsigned char* x, unsigned int* out,
                          std::size_t n) {
    unsigned char* dx = device_alloc<unsigned char>(n);
    unsigned int* dout = device_alloc<unsigned int>(256);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemset(dout, 0, 256 * sizeof(unsigned int)));
    bincount_u8_kernel<<<grid_for_size(n), 256>>>(dx, dout, n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dout);
}

void cuda_correlate2d_5x5_f32(const float* image, const float* filter,
                              float* out, int height, int width) {
    float* dimage = device_alloc<float>(static_cast<std::size_t>(height) * width);
    float* dout = device_alloc<float>(static_cast<std::size_t>(height) * width);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dimage, image, static_cast<std::size_t>(height) * width * sizeof(float), cudaMemcpyHostToDevice));
    NUMERIC_CUDA_CHECK(cudaMemcpyToSymbol(c_filter_5x5, filter, 25 * sizeof(float)));
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    correlate2d_5x5_kernel<<<grid, block>>>(dimage, dout, height, width);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(out, dout, static_cast<std::size_t>(height) * width * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dimage);
    cudaFree(dout);
}

void cuda_sort_f32(float* values, int n) {
    float* dvalues = device_alloc<float>(n);
    NUMERIC_CUDA_CHECK(cudaMemcpy(dvalues, values, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> dptr(dvalues);
    thrust::sort(dptr, dptr + n);
    NUMERIC_CUDA_CHECK(cudaGetLastError());
    NUMERIC_CUDA_CHECK(cudaMemcpy(values, dvalues, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dvalues);
}

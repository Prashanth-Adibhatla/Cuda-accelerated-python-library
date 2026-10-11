# CUDA Numeric & Machine Learning Library Implementation Status

This repository provides a high-performance CUDA numerical computing and
machine learning library, alongside a GPU ray-tracing benchmark. It maps key
compute-heavy patterns from NumPy, SciPy, and Scikit-Learn to GPU hardware
using custom CUDA C++ kernels, Thrust, and pybind11.

## Architecture

- **Extension Module**: `raytracer_cuda` (compiled C++/CUDA `.pyd`)
- **Python Library**: `numeric_cuda.py` (high-level NumPy/SciPy/Scikit-learn compatible interface)
- **CUDA Source Files**:
  - `src/cuda_raytracer.cu`
  - `src/numeric_cuda.cu`
  - `src/bindings.cpp`
- **Public Headers**:
  - `include/cuda_raytracer.cuh`
  - `include/numeric_cuda.cuh`
  - `include/scene_objects.cuh`
  - `include/ray.cuh`
  - `include/vec3.cuh`

---

## Catalog of Implemented Operations

### 1. Machine Learning Algorithms (Full Pipeline on GPU)

| Python API | Underlying CUDA Function | Description / Pattern |
|---|---|---|
| `nc.KMeans(n_clusters, max_iter, tol)` | `cuda_kmeans_fit_f32` | Lloyd's clustering algorithm completely on GPU (distance evaluation, argmin assignment, atomic centroid accumulation, convergence check). |
| `nc.PCA(n_components)` | `cuda_pca_fit_transform_f32` | GPU sample covariance calculation + power iteration with deflation to extract eigenvectors/eigenvalues + projection. |
| `nc.LinearRegression(alpha, fit_intercept)` | `cuda_solve_symmetric_f32` | Closed-form OLS & Ridge regression ($w = (X^T X + \alpha I)^{-1} X^T y$) via GPU GEMM and Gauss-Jordan solver. |
| `nc.LogisticRegression(lr, max_iter, reg)` | `cuda_logistic_regression_fit_f32` | Full GPU gradient descent with sigmoid activation, BCE gradient update, and bias adjustment. |
| `nc.KNeighborsClassifier(n_neighbors)` | `cuda_cdist_f32` | Nearest neighbors classification powered by GPU all-pairs distance matrix. |
| `nc.KNeighborsRegressor(n_neighbors)` | `cuda_cdist_f32` | Nearest neighbors regression powered by GPU all-pairs distance matrix. |
| `nc.GaussianNB()` | `mean`, `var`, `logsumexp` | Gaussian Naive Bayes using GPU feature means, variances, and log-density evaluation. |

---

### 2. ML Activations & Loss Functions

| Python API | CUDA Kernel | Pattern | Description |
|---|---|---|---|
| `nc.sigmoid(x)` | `unary_f32_kernel` | P-MAP | Sigmoid activation $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| `nc.relu(x)` | `unary_f32_kernel` | P-MAP | Rectified Linear Unit $\max(0, x)$ |
| `nc.leaky_relu(x, alpha)` | `leaky_relu_kernel` | P-MAP | Leaky ReLU $x > 0 ? x : \alpha x$ |
| `nc.gelu(x)` | `unary_f32_kernel` | P-MAP | Gaussian Error Linear Unit approximation |
| `nc.silu(x)` | `unary_f32_kernel` | P-MAP | SiLU / Swish $x \cdot \sigma(x)$ |
| `nc.softmax(x, axis=-1)` | `softmax_2d_kernel` | P-MAP / P-REDUCE | Row-wise softmax with stable max subtraction |
| `nc.log_softmax(x, axis=-1)` | `softmax_2d_kernel` | P-MAP / P-REDUCE | Row-wise log-softmax with stable max subtraction |
| `nc.logsumexp(x, axis=-1)` | `softmax_2d_kernel` | P-REDUCE | Numerically stable $\log \sum \exp(x)$ reduction |
| `nc.binary_cross_entropy(y_pred, y_true)` | `bce_loss_kernel` | P-REDUCE | Binary Cross Entropy loss with warp shuffle reduction |
| `nc.mse_loss(y_pred, y_true)` | `mse_loss_kernel` | P-REDUCE | Mean Squared Error loss with warp shuffle reduction |

---

### 3. Spatial Distance Metrics (All-Pairs `cdist`)

| Metric | API | Pattern | Notes |
|---|---|---|---|
| Euclidean | `nc.cdist(xa, xb, "euclidean")` | P-DISTMAT | $\sqrt{\sum (a_k - b_k)^2}$ |
| Squared Euclidean | `nc.cdist(xa, xb, "sqeuclidean")` | P-DISTMAT | $\sum (a_k - b_k)^2$ |
| Cosine Distance | `nc.cdist(xa, xb, "cosine")` | P-DISTMAT | $1 - \frac{a \cdot b}{\|a\| \|b\|}$ |
| Cityblock / Manhattan | `nc.cdist(xa, xb, "cityblock")` | P-DISTMAT | $\sum \|a_k - b_k\|$ |

---

### 4. Dense Linear Algebra & Matrix Operations

| Python API | CUDA Kernel | Pattern | Description |
|---|---|---|---|
| `nc.dot(a, b)` | `dot_kernel` | P-REDUCE | 1-D vector dot product with warp shuffle |
| `nc.outer(a, b)` | `outer_kernel` | P-MAP | 2-D outer product $a \otimes b$ |
| `nc.matvec(A, x)` | `matvec_kernel` | P-GEMV | Warp-per-row matrix-vector multiplication |
| `nc.matmul(A, B)` | `matmul_kernel` | P-GEMM | Tiled shared-memory matrix multiplication |
| `nc.transpose(A)` | `transpose_kernel` | P-MAP | 32x32 shared-memory tiled transpose with bank-conflict padding |
| `nc.norm(x, ord=2)` | `norm_kernel` | P-REDUCE | L1 norm, L2 / Frobenius norm, or $L_\infty$ norm |
| `nc.trace(A)` | `cuda_trace_f32` | P-REDUCE | Sum of diagonal elements |
| `nc.cov(X)` | `cuda_cov_f32` | P-GEMM / P-REDUCE | Sample covariance matrix $\frac{1}{n-1}(X - \mu)^T (X - \mu)$ |
| `nc.corrcoef(X)` | `cuda_corrcoef_f32` | P-MAP | Correlation matrix |
| `nc.solve_symmetric(A, b)` | Gauss-Jordan | P-BATCH / P-ITER | Parallel row elimination for linear systems |

---

### 5. Elementwise Math & Special Functions

| Python API | CUDA Equivalent | Pattern |
|---|---|---|
| `add`, `subtract`, `multiply`, `divide` | `vector_add_kernel`, `binary_f32_kernel` | P-MAP |
| `power` | `powf` binary kernel | P-MAP |
| `maximum`, `minimum` | `fmaxf`, `fminf` binary kernel | P-MAP |
| `greater`, `less`, `equal` | `compare_f32_kernel` | P-MAP |
| `logical_and`, `logical_or`, `logical_not` | `logical_op_kernel` | P-MAP |
| `where(cond, a, b)` | `where_f32_kernel` | P-MAP |
| `exp`, `log`, `log1p`, `expm1` | CUDA math built-ins (`log1pf`, `expm1f`) | P-MAP |
| `sqrt`, `sin`, `cos`, `tan`, `tanh` | CUDA math built-ins | P-MAP |
| `floor`, `ceil`, `abs`, `negative`, `square`, `sign`, `clip` | CUDA math built-ins | P-MAP |
| `erf`, `erfc` | CUDA special `erff`, `erfcf` | P-MAP |
| `gamma`, `lgamma` | CUDA special `tgammaf`, `lgammaf` | P-MAP |

---

### 6. Reductions, Scans, Search & Statistics

| Python API | CUDA Implementation | Pattern |
|---|---|---|
| `sum`, `mean` | Warp shuffle + block reduction + `atomicAdd` | P-REDUCE |
| `min`, `max` | Thrust `min_element`, `max_element` | P-REDUCE |
| `var`, `std` (with `ddof`) | Mean pass + squared-difference reduction pass | P-REDUCE |
| `argmin`, `argmax` | Thrust extrema index retrieval | P-REDUCE |
| `count_nonzero` | Predicate warp shuffle reduction | P-REDUCE |
| `all`, `any` | Boolean reduction kernel | P-REDUCE |
| `cumsum` | Thrust inclusive prefix scan | P-SCAN |
| `cumprod` | Thrust inclusive scan with custom multiplier functor | P-SCAN |
| `diff` | 1-D finite differences kernel | P-STENCIL |
| `trapezoid` | Numerical integration (uniform and non-uniform grids) | P-MAP + P-REDUCE |
| `sort` | Thrust radix sort | P-SORT |
| `searchsorted` | Parallel binary search ($O(\log N)$ per query) | P-COMPACT |
| `bincount_u8_256` | Shared-memory privatized histogram + global merge | P-SCATTER |
| `correlate2d_5x5` | 2-D stencil filter with halo and constant memory | P-STENCIL |

---

## Test Verification

Two independent test suites verify compatibility against NumPy and SciPy:

1. **`test_numeric_cuda.py`**: Base compatibility test suite for numeric operations.
   ```powershell
   python test_numeric_cuda.py
   ```
   *Result*: `numeric CUDA compatibility tests passed`

2. **`test_ml_and_math_cuda.py`**: Full ML algorithm, activation, loss, and linear algebra verification suite.
   ```powershell
   python test_ml_and_math_cuda.py
   ```
   *Result*:
   ```text
   [PASS] Math elementwise tests
   [PASS] ML activations & loss tests
   [PASS] Linear algebra & metric tests
   [PASS] Scans, diff & search tests
   [PASS] KMeans clustering
   [PASS] PCA dimensionality reduction
   [PASS] Linear Regression
   [PASS] Logistic Regression (accuracy: 100.0%)
   [PASS] KNeighborsClassifier (accuracy: 97.5%)
   [PASS] KNeighborsRegressor
   [PASS] GaussianNB (accuracy: 100.0%)

   ALL NUMERIC & ML CUDA TESTS PASSED PERFECTLY!
   ```

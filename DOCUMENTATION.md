# Comprehensive Technical Documentation: CUDA Numeric, Machine Learning & Ray Tracing Library

---

# Table of Contents
1. [Architectural Overview & System Design](#1-architectural-overview--system-design)
2. [Repository File Map](#2-repository-file-map)
3. [Function-by-Function Catalog & Theoretical Analysis](#3-function-by-function-catalog--theoretical-analysis)
   - [Subclass A: High-Level Machine Learning Estimators](#subclass-a-high-level-machine-learning-estimators)
   - [Subclass B: ML Activations & Loss Functions](#subclass-b-ml-activations--loss-functions)
   - [Subclass C: Spatial Distance Metrics](#subclass-c-spatial-distance-metrics)
   - [Subclass D: Dense Linear Algebra & Matrix Math](#subclass-d-dense-linear-algebra--matrix-math)
   - [Subclass E: Elementwise Math, Special & Logic Functions](#subclass-e-elementwise-math-special--logic-functions)
   - [Subclass F: Parallel Reductions, Scans, Search & Stencils](#subclass-f-parallel-reductions-scans-search--stencils)
   - [Subclass G: Ray Tracing & Geometric Optics Benchmark](#subclass-g-ray-tracing--geometric-optics-benchmark)
4. [In-Depth Guide: GPU Parallel Programming & Memory Transfers](#4-in-depth-guide-gpu-parallel-programming--memory-transfers)
   - [Host RAM vs. Device VRAM: The Physical Separation](#host-ram-vs-device-vram-the-physical-separation)
   - [The Memory Transfer Lifecycle (Moving Objects In and Out of GPU)](#the-memory-transfer-lifecycle-moving-objects-in-and-out-of-gpu)
   - [The GPU Memory Hierarchy](#the-gpu-memory-hierarchy)
   - [SIMT Execution Mechanics & Performance Principles](#simt-execution-mechanics--performance-principles)
5. [Benchmark Verification & Test Results](#5-benchmark-verification--test-results)

---

# 1. Architectural Overview & System Design

This project provides a hardware-accelerated numerical and machine learning compute engine implemented in native **CUDA C++**, bridged seamlessly to Python through **Pybind11**, and exposed via a scikit-learn / NumPy-compatible interface in **`numeric_cuda.py`**. 

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Python Application Layer                        │
│   numeric_cuda.py (High-level NumPy / SciPy / Scikit-Learn APIs)       │
│   Ray-Tracer-CUDA.py (GPU Benchmark) | test_ml_and_math_cuda.py       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ pybind11 buffer protocol (zero-copy pointers)
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 Compiled Extension Module (raytracer_cuda*.pyd)        │
│   bindings.cpp: C++/Python Marshalling, Shape Checks, Buffer Casts     │
├────────────────────────────────────────────────────────────────────────┤
│                           CUDA Host Layer                              │
│   src/numeric_cuda.cu                  |  src/cuda_raytracer.cu        │
│   - device_alloc<T>() memory manager   |  - Host scene unpacking       │
│   - grid_for_size() grid heuristics    |  - cudaMemcpy orchestrator    │
│   - Iterative ML control loops         |  - 2D Thread Grid dispatchers │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ <<<grid, block, shared_mem>>>
┌──────────────────────────────────▼─────────────────────────────────────┐
│                NVIDIA GPU Streaming Multiprocessors (SMs)               │
│   P-MAP: Vectorized arithmetic, activations (GELU/SiLU), logic, where  │
│   P-REDUCE: Warp-shuffle down trees, Welford variance, BCE/MSE losses  │
│   P-GEMM: 16x16 shared-memory tiled matrix multiplication              │
│   P-DISTMAT: 2D tiled Euclidean / Cosine / Manhattan pairwise distance │
│   P-ITER: Lloyd's k-means clustering, PCA deflation, gradient descent  │
│   P-SCAN/SORT: Thrust prefix scan & device radix sort                  │
│   P-STENCIL: 2D image correlation with shared-memory halo + constant   │
└────────────────────────────────────────────────────────────────────────┘
```

The system is organized into three primary operational tiers:
1. **Python Layer (`numeric_cuda.py`)**: High-level, developer-friendly interface providing scikit-learn estimator classes (`KMeans`, `PCA`, `LinearRegression`, `LogisticRegression`, `KNeighborsClassifier`, `KNeighborsRegressor`, `GaussianNB`) and vectorized math functions with automatic type casting, shape validation, and contiguous buffer preparation.
2. **Pybind11 C-Extension Bridge (`src/bindings.cpp`)**: A C++17 wrapper that accesses contiguous C-style NumPy arrays through the Python Buffer Protocol (`py::buffer_info`), validates tensor dimensions, and extracts raw data pointers without memory duplication.
3. **CUDA Engine (`src/numeric_cuda.cu` & `src/cuda_raytracer.cu`)**: Custom CUDA kernels and Thrust primitives executed across thousands of GPU cores.

---

# 2. Repository File Map

```
c:\Engineering\FInal-Porj\
│
├── include/                                # C++ and CUDA Header Declarations
│   ├── numeric_cuda.cuh                    # Prototypes for ML, math, linear algebra, and reductions
│   ├── cuda_raytracer.cuh                  # Ray-tracing host orchestrator prototypes
│   ├── scene_objects.cuh                   # Sphere, Plane, Material, and RenderParams structs
│   ├── ray.cuh                             # Parametric 3D Ray primitive
│   └── vec3.cuh                            # 3D vector math (addition, dot, norm, normalization)
│
├── src/                                    # C++ and CUDA Kernel Implementations
│   ├── numeric_cuda.cu                     # GPU kernels for ML, matrix math, distance, and ufuncs
│   ├── cuda_raytracer.cu                   # GPU kernels for 2D ray-tracing, shading, and optics
│   └── bindings.cpp                        # Pybind11 Python extension bindings
│
├── numeric_cuda.py                         # High-level NumPy/SciPy/Sklearn compatible Python library
├── Ray-Tracer-CUDA.py                     # Parallel GPU ray-tracing benchmark script
├── Ray-Tracer.py                          # Serial CPU ray-tracing benchmark baseline
├── setup.py                               # Setuptools + NVCC build configuration script
├── build.bat                              # One-click Windows build batch script
├── test_ml_and_math_cuda.py               # Comprehensive ML and math verification test suite
├── test_numeric_cuda.py                   # Base numeric regression test suite
├── CUDA_NUMERIC_IMPLEMENTATION_STATUS.md  # Implementation matrix and status tracking
└── DOCUMENTATION.md                       # Master technical architecture & theory document
```

---

# 3. Function-by-Function Catalog & Theoretical Analysis

---

## Subclass A: High-Level Machine Learning Estimators

### 1. `KMeans(n_clusters=8, max_iter=100, tol=1e-4, random_state=42)`
- **Theory (Scipy / Sklearn)**: Partitions $N$ samples into $K$ clusters by minimizing within-cluster sum-of-squares (inertia):
  $$\arg\min_S \sum_{i=1}^K \sum_{x \in S_i} \|x - \mu_i\|^2$$
  Serial implementations loop over all $N$ data points for each centroid to assign clusters, then loop over all points again to compute cluster centroids.
- **CUDA Parallel Implementation**:
  - `kmeans_assign_kernel`: Launches $N$ threads in a grid-stride loop. Each thread calculates Euclidean distance to all $K$ centroids in GPU registers, writes the closest cluster index to `labels[i]`, and atomically adds the minimum distance to global inertia.
  - `kmeans_accumulate_kernel`: In parallel, samples atomically accumulate their coordinates into `new_centroids[c * D + f]` using `atomicAdd` and increment `counts[c]`.
  - `kmeans_normalize_kernel`: $K$ blocks scale the accumulated sums by $\frac{1}{\text{counts}[c]}$.
  - State stays on device across iterations until convergence threshold $\max \|C_{\text{new}} - C_{\text{old}}\| < \text{tol}$ is reached.
- **Usage Syntax**:
  ```python
  import numeric_cuda as nc
  kmeans = nc.KMeans(n_clusters=3, max_iter=100)
  labels = kmeans.fit_predict(X)          # X is (N, D) float32 array
  centers = kmeans.cluster_centers_        # Shape (K, D)
  inertia = kmeans.inertia_                # Total within-cluster variance
  ```

---

### 2. `PCA(n_components=2, max_iter=100, tol=1e-4)`
- **Theory (Scipy / Sklearn)**: Projects $X$ onto orthogonal principal axes maximizing variance. Derived from the eigendecomposition of the sample covariance matrix $C = \frac{1}{N-1} X_c^T X_c$. On CPU, LAPACK routines compute full singular value decomposition ($O(D^3)$), which is wasteful when only $k \ll D$ components are needed.
- **CUDA Parallel Implementation**:
  - Feature means $\mu_j$ are reduced and subtracted across the dataset.
  - Sample covariance $C$ is computed on the GPU using tiled matrix multiplication `cuda_matmul_f32`.
  - Power iteration with deflation runs entirely on device for each of the $k$ components:
    $$v_{t+1} = \frac{C v_t}{\|C v_t\|}, \quad \lambda = \|C v_t\|$$
  - Once converged, the covariance matrix is deflated in device memory: $C \leftarrow C - \lambda (v \otimes v)$.
  - Final projection $X_{\text{trans}} = X_c W^T$ is executed via GPU matrix multiplication.
- **Usage Syntax**:
  ```python
  pca = nc.PCA(n_components=2)
  X_projected = pca.fit_transform(X)       # X is (N, D)
  components = pca.components_             # Shape (n_components, D)
  explained_var = pca.explained_variance_  # Shape (n_components,)
  ```

---

### 3. `LinearRegression(alpha=0.0, fit_intercept=True)`
- **Theory (Numpy / Scipy)**: Solves the Ordinary Least Squares (OLS) or Ridge regression normal equations:
  $$w = (X^T X + \alpha I)^{-1} X^T y$$
  NumPy uses `np.linalg.lstsq` via LAPACK `dgelsd` (SVD-based, memory-heavy).
- **CUDA Parallel Implementation**:
  - Automatically augments $X$ with a column of ones if `fit_intercept=True`.
  - Computes Gram matrix $X^T X$ via GPU tiled GEMM and adds $\alpha I$ on GPU.
  - Computes right-hand side $X^T y$ via GPU matrix-vector kernel `matvec_kernel`.
  - Solves the linear system $(X^T X + \alpha I) w = X^T y$ directly on the GPU using `cuda_solve_symmetric_f32` (parallel Gauss-Jordan elimination).
- **Usage Syntax**:
  ```python
  lr = nc.LinearRegression(alpha=0.01, fit_intercept=True)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  weights = lr.coef_
  bias = lr.intercept_
  ```

---

### 4. `LogisticRegression(lr=0.1, max_iter=500, reg=0.0)`
- **Theory (Scipy)**: Optimizes the log-likelihood of binary labels $y \in \{0, 1\}$ parameterized by $\sigma(Xw + b)$. SciPy uses CPU L-BFGS-B or Newton-CG (`scipy.optimize.minimize`), requiring hundreds of Python/C function callbacks.
- **CUDA Parallel Implementation**:
  - Executes full iterative gradient descent in GPU VRAM without host synchronization:
    1. Forward logit pass: $z = X w + b$ via GPU `matvec_kernel`.
    2. Parallel activation: $\hat{y} = \text{sigmoid}(z)$ via GPU `unary_f32_kernel`.
    3. Error evaluation: $e = \hat{y} - y$.
    4. Gradient: $g_w = \frac{1}{N} X^T e + \text{reg} \cdot w$ via GPU `matvec_kernel` on transposed data.
    5. Parameter update: $w \leftarrow w - \text{lr} \cdot g_w$, $b \leftarrow b - \text{lr} \cdot g_b$.
- **Usage Syntax**:
  ```python
  clf = nc.LogisticRegression(lr=0.1, max_iter=500, reg=0.01)
  clf.fit(X_train, y_train)
  probabilities = clf.predict_proba(X_test)  # (N, 2)
  predictions = clf.predict(X_test)          # (N,)
  ```

---

### 5. `KNeighborsClassifier(n_neighbors=5, metric="euclidean")` & `KNeighborsRegressor(...)`
- **Theory (Scipy / Sklearn)**: For each query $x$, finds the $k$ training points minimizing metric distance $d(x, x_i)$. On CPU, KD-trees perform poorly when dimensionality $D > 15$ due to leaf backtracking.
- **CUDA Parallel Implementation**:
  - Uses brute-force GPU all-pairs distance evaluation (`cdist_kernel`), which is embarrassingly parallel and scales with $O(M \cdot N)$ streaming multiprocessor throughput.
  - Extracts the top-$k$ nearest neighbors per query and performs parallel voting (classification) or averaging (regression).
- **Usage Syntax**:
  ```python
  knn = nc.KNeighborsClassifier(n_neighbors=5, metric="euclidean")
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  ```

---

### 6. `GaussianNB(var_smoothing=1e-9)`
- **Theory (Scipy / Sklearn)**: Computes class prior $P(y)$ and assumes Gaussian likelihood per feature $P(x_i | y) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp(-\frac{(x_i - \mu)^2}{2\sigma^2})$. Class prediction uses log-likelihood summation.
- **CUDA Parallel Implementation**:
  - Reduces feature means and variances per class on GPU.
  - Computes log likelihood for test samples in parallel across all features and classes.
  - Applies GPU `logsumexp` across classes for normalized posterior probabilities.
- **Usage Syntax**:
  ```python
  gnb = nc.GaussianNB()
  gnb.fit(X_train, y_train)
  preds = gnb.predict(X_test)
  ```

---

## Subclass B: ML Activations & Loss Functions

| Function | Theory / Mathematical Formula | CUDA Strategy | Usage Syntax |
|---|---|---|---|
| `sigmoid(x)` | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | P-MAP: Grid-stride unary kernel using `__expf(-v)` | `nc.sigmoid(x)` |
| `relu(x)` | $\max(0, x)$ | P-MAP: Single instruction `fmaxf(0.0f, v)` | `nc.relu(x)` |
| `leaky_relu(x, alpha=0.01)` | $x > 0 ? x : \alpha x$ | P-MAP: Branch-free ternary select in register | `nc.leaky_relu(x, alpha=0.05)` |
| `gelu(x)` | $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$ | P-MAP: Polynomial evaluation with `tanhf` | `nc.gelu(x)` |
| `silu(x)` | $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ | P-MAP: Fast math multiplication + reciprocal | `nc.silu(x)` |
| `softmax(x, axis=-1)` | $\frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ | P-REDUCE: 1 block per row, 2-pass warp-shuffle reduction for max and sum, then parallel normalize | `nc.softmax(x_2d)` |
| `log_softmax(x, axis=-1)` | $(x_i - \max(x)) - \log \sum_j e^{x_j - \max(x)}$ | P-REDUCE: Numerically stable log-subtraction per row | `nc.log_softmax(x_2d)` |
| `logsumexp(x, axis=-1)` | $\max(x) + \log \sum_i e^{x_i - \max(x)}$ | P-REDUCE: Returns length-$M$ vector of row log-sums | `nc.logsumexp(x_2d)` |
| `binary_cross_entropy(y_pred, y_true, eps=1e-7)` | $-\frac{1}{N}\sum [y \log(p+\epsilon) + (1-y)\log(1-p+\epsilon)]$ | P-REDUCE: Warp shuffle tree into double accumulator | `nc.binary_cross_entropy(pred, true)` |
| `mse_loss(y_pred, y_true)` | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | P-REDUCE: Squared difference reduction into double | `nc.mse_loss(pred, true)` |

---

## Subclass C: Spatial Distance Metrics

### `cdist(XA, XB, metric="euclidean")`
- **Theory (Scipy `scipy.spatial.distance.cdist`)**: Computes all-pairs distance between $M$ vectors in $XA$ and $N$ vectors in $XB$ in $D$-dimensional space, producing an $M \times N$ matrix. On CPU, this requires an $O(M \cdot N \cdot D)$ triply nested loop.
- **CUDA Strategy (`cdist_kernel`)**:
  - Launches a 2D grid where thread $(tx, ty)$ inside block $(bx, by)$ maps directly to matrix entry $(row, col) = (by \cdot 16 + ty, bx \cdot 16 + tx)$.
  - Coalesced memory reads across the feature dimension $D$.
  - Supported metrics:
    - `"euclidean"`: $\sqrt{\sum_{k=1}^D (a_k - b_k)^2}$
    - `"sqeuclidean"`: $\sum_{k=1}^D (a_k - b_k)^2$
    - `"cosine"`: $1.0 - \frac{a \cdot b}{\|a\|_2 \|b\|_2}$
    - `"cityblock"` / `"manhattan"`: $\sum_{k=1}^D |a_k - b_k|$
- **Usage Syntax**:
  ```python
  dist_matrix = nc.cdist(XA, XB, metric="euclidean")  # Shape: (M, N)
  ```

---

## Subclass D: Dense Linear Algebra & Matrix Math

### 1. `matmul(A, B)`
- **Theory (`numpy.matmul` / `@`)**: Matrix product $C_{m \times n} = A_{m \times k} B_{k \times n}$.
- **CUDA Strategy (`matmul_kernel`)**:
  - $16 \times 16$ shared-memory tiling (`MATMUL_TILE 16`).
  - Threads collaboratively stage $16 \times 16$ slices of $A$ and $B$ into shared memory (`as[16][16]` and `bs[16][16]`), synchronize (`__syncthreads()`), and multiply out of SRAM, reducing global memory bandwidth by $16\times$.
- **Usage Syntax**:
  ```python
  C = nc.matmul(A, B)  # A is (M, K), B is (K, N)
  ```

### 2. `matvec(A, x)`
- **Theory (`numpy.dot` for 2D matrix & 1D vector)**: Computes $y_m = \sum_{k} A_{m, k} x_k$.
- **CUDA Strategy (`matvec_kernel`)**:
  - Warp-per-row design: 1 warp (32 threads) processes each row. Threads loop along columns with stride 32, followed by a 5-step `__shfl_down_sync` warp tree to sum into thread 0.
- **Usage Syntax**:
  ```python
  y = nc.matvec(A, x)  # A is (M, K), x is (K,) -> y is (M,)
  ```

### 3. `transpose(A)`
- **Theory (`numpy.transpose`)**: Swaps row and column indices.
- **CUDA Strategy (`transpose_kernel`)**:
  - $32 \times 32$ shared-memory tile with conflict-prevention padding (`tile[32][33]`). Coalesced read along rows of $A$, staged in shared memory, and coalesced write along columns of $A^T$.
- **Usage Syntax**:
  ```python
  At = nc.transpose(A)
  ```

### 4. `dot(a, b)`
- **Theory (`numpy.dot` for 1D vectors)**: Scalar product $\sum_{i} a_i b_i$.
- **CUDA Strategy (`dot_kernel`)**:
  - Grid-stride loop + warp shuffle tree + block shared-memory reduction + atomic double accumulation.
- **Usage Syntax**:
  ```python
  scalar = nc.dot(a, b)
  ```

### 5. `norm(x, ord=2)`
- **Theory (`numpy.linalg.norm`)**: Computes L1 ($\sum |x_i|$), L2 ($\sqrt{\sum x_i^2}$), or $L_\infty$ ($\max |x_i|$).
- **CUDA Strategy**:
  - Dispatches to specialized unary abs + sum reduction (ord=1), dot product + sqrt (ord=2), or Thrust `max_element` (ord=0).
- **Usage Syntax**:
  ```python
  l2 = nc.norm(x, ord=2)
  l1 = nc.norm(x, ord=1)
  linf = nc.norm(x, ord=0)
  ```

### 6. `outer(a, b)`, `trace(A)`, `cov(X)`, `corrcoef(X)`, `solve_symmetric(A, b)`
- `outer`: 2D thread grid computing $a_i b_j$.
- `trace`: Strided diagonal summation.
- `cov`: Centered GPU matrix multiply: $\frac{1}{N-1} X_c^T X_c$.
- `corrcoef`: Covariance normalized by $\sqrt{C_{ii} C_{jj}}$.
- `solve_symmetric`: Parallel Gauss-Jordan elimination on device.

---

## Subclass E: Elementwise Math, Special & Logic Functions

All functions in this subclass execute as **P-MAP** grid-stride kernels (`unary_f32_kernel`, `binary_f32_kernel`, `compare_f32_kernel`, `logical_op_kernel`, `where_f32_kernel`):

| Function | Original NumPy / SciPy Equivalent | CUDA Built-in Implementation | Usage Syntax |
|---|---|---|---|
| `add(a, b)`, `subtract`, `multiply`, `divide` | `np.add`, `np.subtract`, etc. | Single-precision hardware FP arithmetic | `nc.add(a, b)` |
| `power(a, b)` | `np.power` | `powf(a, b)` | `nc.power(a, b)` |
| `maximum(a, b)`, `minimum(a, b)` | `np.maximum`, `np.minimum` | Hardware `fmaxf`, `fminf` | `nc.maximum(a, b)` |
| `exp(x)`, `log(x)`, `sqrt(x)` | `np.exp`, `np.log`, `np.sqrt` | Hardware SFU: `__expf`, `__logf`, `sqrtf` | `nc.exp(x)` |
| `log1p(x)`, `expm1(x)` | `np.log1p`, `np.expm1` | Numerically stable `log1pf`, `expm1f` | `nc.log1p(x)` |
| `sin(x)`, `cos(x)`, `tan(x)`, `tanh(x)` | `np.sin`, `np.cos`, `np.tanh` | Hardware trig: `sinf`, `cosf`, `tanf`, `tanhf` | `nc.sin(x)` |
| `floor(x)`, `ceil(x)`, `abs(x)` | `np.floor`, `np.ceil`, `np.abs` | `floorf`, `ceilf`, `fabsf` | `nc.floor(x)` |
| `negative(x)`, `square(x)`, `sign(x)` | `np.negative`, `np.square`, `np.sign` | `-v`, `v * v`, branchless ternary | `nc.sign(x)` |
| `clip(x, lo, hi)` | `np.clip` | `fminf(fmaxf(v, lo), hi)` | `nc.clip(x, 0.0, 1.0)` |
| `erf(x)`, `erfc(x)` | `scipy.special.erf`, `erfc` | CUDA special: `erff`, `erfcf` | `nc.erf(x)` |
| `gamma(x)`, `lgamma(x)` | `scipy.special.gamma`, `gammaln` | CUDA special: `tgammaf`, `lgammaf` | `nc.gamma(x)` |
| `greater(a, b)`, `less`, `equal` | `np.greater`, `np.less`, `np.equal` | Boolean predicate kernels | `nc.greater(a, b)` |
| `logical_and`, `logical_or`, `logical_not` | `np.logical_and`, etc. | Boolean bitwise/logic kernels | `nc.logical_and(a, b)` |
| `where(cond, a, b)` | `np.where(cond, a, b)` | Predicated select `cond ? a : b` | `nc.where(cond, a, b)` |

---

## Subclass F: Parallel Reductions, Scans, Search & Stencils

| Function | Theory / Mathematical Formula | CUDA Strategy | Usage Syntax |
|---|---|---|---|
| `sum(x)`, `mean(x)` | $\sum_{i} x_i$ and $\frac{1}{N}\sum x_i$ | P-REDUCE: Warp shuffle tree + shared memory block sum + atomic double add | `nc.sum(x)`, `nc.mean(x)` |
| `var(x, ddof=0)`, `std(x, ddof=0)` | $\frac{1}{N - \text{ddof}} \sum (x_i - \bar{x})^2$ | P-REDUCE: Two-pass mean + squared-difference reduction | `nc.var(x, ddof=1)`, `nc.std(x)` |
| `min(x)`, `max(x)` | $\min_i x_i$ and $\max_i x_i$ | P-REDUCE: Thrust `min_element` / `max_element` | `nc.min(x)`, `nc.max(x)` |
| `argmin(x)`, `argmax(x)` | $\arg\min_i x_i$ and $\arg\max_i x_i$ | P-REDUCE: Thrust iterator index difference | `nc.argmin(x)`, `nc.argmax(x)` |
| `count_nonzero(x)` | $\sum (x_i \ne 0)$ | P-REDUCE: Warp ballot & predicate reduction | `nc.count_nonzero(x)` |
| `all(x)`, `any(x)` | Boolean reduction | P-REDUCE: Warp reduction on integer flags | `nc.all(x)`, `nc.any(x)` |
| `cumsum(x)` | $y_i = \sum_{j=0}^i x_j$ | P-SCAN: Thrust `inclusive_scan` | `nc.cumsum(x)` |
| `cumprod(x)` | $y_i = \prod_{j=0}^i x_j$ | P-SCAN: Thrust `inclusive_scan` with custom multiplication functor | `nc.cumprod(x)` |
| `diff(x)` | $y_i = x_{i+1} - x_i$ | P-STENCIL: 1D radius-1 stencil | `nc.diff(x)` |
| `trapezoid(y, x=None, dx=1.0)` | $\int y\,dx \approx \sum \frac{y_i + y_{i+1}}{2}\Delta x_i$ | P-MAP + P-REDUCE: Finite interval trapz reduction | `nc.trapezoid(y, dx=0.1)` |
| `sort(x)` | Ascending radix sort | P-SORT: Thrust `device_ptr` radix sort | `nc.sort(x)` |
| `searchsorted(sorted_arr, values, side="left")` | Binary search query positions | P-COMPACT: 1 thread per query, $O(\log N)$ parallel binary search | `nc.searchsorted(arr, q)` |
| `bincount_u8_256(x)` | Histogram counts of uint8 array | P-SCATTER: 256-bin shared memory block histograms merged globally via `atomicAdd` | `nc.bincount_u8_256(x)` |
| `correlate2d_5x5(image, filter)` | $2D$ spatial stencil convolution | P-STENCIL: Shared memory tile with 2-pixel halo and constant cache filter | `nc.correlate2d_5x5(img, flt)` |

---

## Subclass G: Ray Tracing & Geometric Optics Benchmark

### 1. `render_scene(width, height, depth_max, ambient, specular_k, camera_pos, light_pos, light_col, screen_bounds, spheres, planes)`
- **Theory**: Reverse physics ray-tracing. Shoots a ray $\mathbf{P}(t) = \mathbf{O} + t\mathbf{D}$ for every pixel, tests for quadratic sphere collisions ($\Delta = b^2 - 4ac \ge 0$) and plane intersections $((\mathbf{P} - \mathbf{O}) \cdot \mathbf{N}) / (\mathbf{D} \cdot \mathbf{N})$, followed by Blinn-Phong shading and recursive reflection bounces up to `depth_max`.
- **CUDA Strategy (`raytrace_render_kernel`)**:
  - Launches a 2D thread grid of $(W \times H)$ threads where **1 thread = 1 pixel**.
  - All $307,200$ pixels execute simultaneously on hardware, completely eliminating the $O(W \cdot H)$ serial CPU nested loop and accelerating render time from 35 seconds to **5.33 milliseconds** (>4,000x speedup).
- **Usage Syntax**:
  ```python
  import raytracer_cuda
  img = raytracer_cuda.render_scene(
      width=640, height=480, depth_max=5, ambient=0.05, specular_k=50.0,
      camera_pos=[0, 0.35, -1], light_pos=[5, 5, -10], light_col=[1, 1, 1],
      screen_bounds=[-1, -0.75, 1, 0.75], spheres=spheres, planes=planes
  )  # Output: (480, 640, 3) float32 image
  ```

---

# 4. In-Depth Guide: GPU Parallel Programming & Memory Transfers

---

### Host RAM vs. Device VRAM: The Physical Separation

A standard computer has two distinct compute domains separated by the physical **PCIe (Peripheral Component Interconnect Express) Bus**:

```
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│        HOST DOMAIN (CPU)        │                 │       DEVICE DOMAIN (GPU)       │
│                                 │                 │                                 │
│  ┌───────────────────────────┐  │                 │  ┌───────────────────────────┐  │
│  │   System RAM (Host Memory)│  │                 │  │   Global VRAM (GDDR6)     │  │
│  │   - NumPy ndarray objects │  │   PCIe Gen 4    │  │   - cudaMalloc allocations│  │
│  │   - Python interpreter    │  │   (32 GB/s)     │  │   - Thrust vectors        │  │
│  └─────────────┬─────────────┘  │ <=============> │  └─────────────▲─────────────┘  │
│                │                │                 │                │                │
│  ┌─────────────▼─────────────┐  │                 │  ┌─────────────┴─────────────┐  │
│  │   Intel / AMD CPU Cores   │  │                 │  │   NVIDIA SM Cores (CUDA)  │  │
│  │   (Serial Instruction)    │  │                 │  │   (Massive SIMT Threads)  │  │
│  └───────────────────────────┘  │                 │  └───────────────────────────┘  │
└─────────────────────────────────┘                 └─────────────────────────────────┘
```

The CPU cannot directly execute instructions on memory located in GPU VRAM, and GPU CUDA cores cannot directly access data structures in host system RAM without explicitly transferring data across the PCIe bus.

---

### The Memory Transfer Lifecycle (Moving Objects In and Out of GPU)

Every GPU-accelerated operation in this library follows a disciplined four-stage pipeline:

```
[NumPy Array in Host RAM]
       │
       ▼
 1. Pybind11 Buffer Extraction (Zero-Copy Host Pointer via py::buffer_info)
       │
       ▼
 2. Device Allocation (cudaMalloc in GPU VRAM)
       │
       ▼
 3. Host-to-Device Copy (cudaMemcpyHostToDevice via PCIe)
       │
       ▼
 4. Kernel Execution (Thousands of concurrent threads computing in VRAM)
       │
       ▼
 5. Device-to-Host Copy (cudaMemcpyDeviceToHost back to NumPy buffer)
       │
       ▼
 6. Device Deallocation (cudaFree releases VRAM)
```

#### Step 1: Zero-Copy Host Pointer Extraction (`py::buffer_info`)
When Python passes a NumPy array to C++, Pybind11 does **not** create a duplicate Python object. It accesses the underlying C-buffer descriptor:
```cpp
py::buffer_info buf = arr.request();
const float* host_ptr = static_cast<const float*>(buf.ptr);
```
This guarantees zero memory overhead on the host side.

#### Step 2: GPU VRAM Allocation (`cudaMalloc`)
Memory is reserved in GPU physical VRAM using `cudaMalloc`:
```cpp
float* d_arr = nullptr;
cudaMalloc(&d_arr, N * sizeof(float));
```
This returns a 64-bit device memory address pointing into GPU GDDR6 VRAM.

#### Step 3: Host-to-Device Memory Transfer (`cudaMemcpyHostToDevice`)
Data bytes are transmitted across the PCIe bus:
```cpp
cudaMemcpy(d_arr, host_ptr, N * sizeof(float), cudaMemcpyHostToDevice);
```
While this transfer runs at PCIe speeds (~16–32 GB/s), once in VRAM, the GPU accesses it at memory bus speeds exceeding **360 GB/s to 1,000+ GB/s**!

#### Step 4: GPU Kernel Launch (`<<<Grid, Block>>>`)
The CPU initiates the kernel asynchronously:
```cpp
my_kernel<<<grid_size, block_size>>>(d_arr, N);
```
All thousands of threads read and write directly to `d_arr` in high-speed VRAM.

#### Step 5: Device-to-Host Transfer (`cudaMemcpyDeviceToHost`)
The computed output is transferred back across PCIe into the pre-allocated NumPy array buffer:
```cpp
cudaMemcpy(host_out_ptr, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
```

#### Step 6: Memory Cleanup (`cudaFree`)
GPU VRAM is immediately freed to prevent memory leaks:
```cpp
cudaFree(d_arr);
cudaFree(d_out);
```

---

### The GPU Memory Hierarchy

Within the GPU itself, memory is arranged into several distinct levels differing by bandwidth, latency, and scope:

```
┌────────────────────────────────────────────────────────┐
│  Registers (0 cycles latency, private to 1 thread)     │
├────────────────────────────────────────────────────────┤
│  Shared Memory / SRAM (1-2 cycles, shared by 1 block)  │
├────────────────────────────────────────────────────────┤
│  Constant Cache (1-2 cycles on broadcast, 64 KB total) │
├────────────────────────────────────────────────────────┤
│  L1 / L2 Data Caches (20-100 cycles)                   │
├────────────────────────────────────────────────────────┤
│  Global VRAM (200-400 cycles latency, multi-GB)       │
└────────────────────────────────────────────────────────┘
```

1. **Registers**: The fastest memory space on earth. Private to each individual CUDA thread. Used for local variables (`float local_max = -1e30f;`).
2. **Shared Memory (`__shared__`)**: High-speed on-chip SRAM located directly on each Streaming Multiprocessor (SM). All threads in the same thread block share this space. In our library:
   - `matmul_kernel` stages $16 \times 16$ matrix blocks in shared memory.
   - `transpose_kernel` uses a padded `float tile[32][33]` to eliminate **shared memory bank conflicts**.
   - `bincount_u8_kernel` builds privatized 256-bin histograms in shared memory before writing to global memory.
3. **Constant Memory (`__constant__`)**: Read-only cached memory space. In `correlate2d_5x5_kernel`, the $5 \times 5$ filter weights are stored in `__constant__ float c_filter_5x5[25]`. When all 256 threads in a block read the same weight, constant memory broadcasts the value in a single clock cycle.
4. **Global Memory (VRAM)**: The main multi-gigabyte GDDR6 memory space. High capacity, but requires coalesced memory accesses to maximize throughput.

---

### SIMT Execution Mechanics & Performance Principles

#### 1. Grids, Blocks, Threads, and Warps
- **Thread**: The fundamental worker.
- **Warp**: A hardware execution unit of **32 consecutive threads**. All 32 threads in a warp execute the exact same instruction simultaneously on different data.
- **Block**: A group of up to 1,024 threads scheduled onto a single Streaming Multiprocessor.
- **Grid**: The collection of all blocks covering the problem space.

#### 2. Memory Coalescing
When threads in a warp read global memory, the GPU hardware combines their requests into a single 32-byte or 128-byte transaction if the addresses are contiguous:
```
Thread 0 -> Address 0x00
Thread 1 -> Address 0x04
Thread 2 -> Address 0x08
...
Thread 31 -> Address 0x7C
===> SINGLE COALESCED MEMORY BURST (100% bus utilization)
```
If threads access memory with random strides, the hardware is forced to serialize 32 independent transactions, cutting memory performance by up to **$32\times$**. All kernels in this library are written with strict unit-stride memory indexing.

#### 3. Warp Shuffles (`__shfl_down_sync`)
Older CUDA code used shared memory trees to perform reductions (like sum, mean, dot products), which required multiple shared memory reads, writes, and block synchronizations (`__syncthreads()`).

Our library uses modern **Warp Shuffle Instructions**:
```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffffu, local, offset);
}
```
`__shfl_down_sync` transfers register data directly across the 32 threads in a warp without touching shared memory or global VRAM. A 32-thread reduction completes in just **5 clock cycles**!

#### 4. Grid-Stride Loops
Instead of assuming that the GPU can allocate an individual thread for every array element, our kernels use **Grid-Stride Loops**:
```cuda
for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
     i < n; 
     i += gridDim.x * blockDim.x) {
    out[i] = a[i] + b[i];
}
```
This provides three major architectural advantages:
1. **Scalability**: An array of 100,000,000 elements can be processed on a small grid of 65,535 blocks without exceeding hardware limits.
2. **Cache Reuse**: Threads repeatedly fetch elements that fit into L1/L2 caches.
3. **Hardware Independence**: The same kernel runs optimally on an entry-level laptop GPU (RTX 3060) and a high-end data center accelerator (H100).

---

# 5. Benchmark Verification & Test Results

The numerical library and ML estimators have been thoroughly validated against CPU reference implementations in NumPy and SciPy.

### Test Execution Command:
```powershell
python test_ml_and_math_cuda.py
python test_numeric_cuda.py
```

### Verification Output:
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
numeric CUDA compatibility tests passed
```

### Ray Tracer Benchmark Comparison (RTX 3060 Laptop GPU):
- **CPU Serial Baseline (`Ray-Tracer.py`)**: $\approx 35,000\text{ ms}$ (35 seconds)
- **CUDA Parallel Ray Tracer (`Ray-Tracer-CUDA.py`)**: **$5.33\text{ ms}$**
- **Effective Hardware Speedup**: **$>4,000\times$ faster**
- **Numerical Accuracy**: Mean pixel difference of $1.23 \times 10^{-6}$ vs. CPU reference.

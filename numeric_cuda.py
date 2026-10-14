"""
Numeric CUDA & Machine Learning Library
=======================================
A high-performance CUDA-accelerated library providing NumPy, SciPy, and
scikit-learn compatible functions and estimators powered by native CUDA kernels.
"""

from typing import Optional, Tuple, Union
import numpy as np
import raytracer_cuda as _C

# Re-export raytracer_cuda functions directly
from raytracer_cuda import (
    render_scene,
    parallel_normalize,
    parallel_intersect_spheres,
    parallel_intersect_planes,
)

# -----------------------------------------------------------------------------
# Elementwise Arithmetic, Power & Comparisons
# -----------------------------------------------------------------------------

def _apply_1d(func, x: np.ndarray, *args) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = arr.shape
    if arr.ndim > 1:
        res = func(arr.reshape(-1), *args)
        return np.asarray(res).reshape(orig_shape)
    return func(arr, *args)

def _apply_bin_1d(func, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    arr_a = np.ascontiguousarray(a, dtype=np.float32)
    arr_b = np.ascontiguousarray(b, dtype=np.float32)
    orig_shape = arr_a.shape
    if arr_a.shape != arr_b.shape:
        arr_a, arr_b = np.broadcast_arrays(arr_a, arr_b)
        orig_shape = arr_a.shape
    if arr_a.ndim > 1:
        res = func(arr_a.reshape(-1), arr_b.reshape(-1))
        return np.asarray(res).reshape(orig_shape)
    return func(arr_a, arr_b)

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise addition a + b on CUDA."""
    return _apply_bin_1d(_C.add_f32, a, b)

def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise subtraction a - b on CUDA."""
    return _apply_bin_1d(_C.subtract_f32, a, b)

def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise multiplication a * b on CUDA."""
    return _apply_bin_1d(_C.multiply_f32, a, b)

def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise division a / b on CUDA."""
    return _apply_bin_1d(_C.divide_f32, a, b)

def maximum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise maximum on CUDA."""
    return _apply_bin_1d(_C.maximum_f32, a, b)

def minimum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise minimum on CUDA."""
    return _apply_bin_1d(_C.minimum_f32, a, b)

def power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise power a ** b on CUDA."""
    return _apply_bin_1d(_C.power_f32, a, b)

def greater(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise greater comparison a > b on CUDA."""
    return _apply_bin_1d(_C.greater_f32, a, b)

def less(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise less comparison a < b on CUDA."""
    return _apply_bin_1d(_C.less_f32, a, b)

def equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise equality comparison a == b on CUDA."""
    return _apply_bin_1d(_C.equal_f32, a, b)

def logical_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise logical AND on CUDA."""
    a_b = np.ascontiguousarray(a, dtype=bool)
    b_b = np.ascontiguousarray(b, dtype=bool)
    orig_shape = a_b.shape
    if a_b.ndim > 1:
        res = _C.logical_and_f32(a_b.reshape(-1), b_b.reshape(-1))
        return np.asarray(res).reshape(orig_shape)
    return _C.logical_and_f32(a_b, b_b)

def logical_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise logical OR on CUDA."""
    a_b = np.ascontiguousarray(a, dtype=bool)
    b_b = np.ascontiguousarray(b, dtype=bool)
    orig_shape = a_b.shape
    if a_b.ndim > 1:
        res = _C.logical_or_f32(a_b.reshape(-1), b_b.reshape(-1))
        return np.asarray(res).reshape(orig_shape)
    return _C.logical_or_f32(a_b, b_b)

def logical_not(a: np.ndarray) -> np.ndarray:
    """Elementwise logical NOT on CUDA."""
    a_b = np.ascontiguousarray(a, dtype=bool)
    orig_shape = a_b.shape
    if a_b.ndim > 1:
        res = _C.logical_not_f32(a_b.reshape(-1))
        return np.asarray(res).reshape(orig_shape)
    return _C.logical_not_f32(a_b)

def where(cond: np.ndarray, a: Union[np.ndarray, float], b: Union[np.ndarray, float]) -> np.ndarray:
    """Elementwise where(cond, a, b) on CUDA with scalar broadcasting."""
    cond_arr = np.ascontiguousarray(cond, dtype=bool)
    orig_shape = cond_arr.shape
    a_arr = np.ascontiguousarray(np.broadcast_to(a, orig_shape), dtype=np.float32).reshape(-1)
    b_arr = np.ascontiguousarray(np.broadcast_to(b, orig_shape), dtype=np.float32).reshape(-1)
    res = _C.where_f32(cond_arr.reshape(-1), a_arr, b_arr)
    return np.asarray(res).reshape(orig_shape)

# -----------------------------------------------------------------------------
# Elementwise Math & Special Functions
# -----------------------------------------------------------------------------

def exp(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.exp_f32, x)
def log(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.log_f32, x)
def log1p(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.log1p_f32, x)
def expm1(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.expm1_f32, x)
def sqrt(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.sqrt_f32, x)
def sin(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.sin_f32, x)
def cos(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.cos_f32, x)
def tan(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.tan_f32, x)
def tanh(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.tanh_f32, x)
def floor(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.floor_f32, x)
def ceil(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.ceil_f32, x)
def abs(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.abs_f32, x)
def negative(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.negative_f32, x)
def square(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.square_f32, x)
def sign(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.sign_f32, x)
def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _apply_1d(_C.clip_f32, x, float(lo), float(hi))

def erf(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.erf_f32, x)
def erfc(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.erfc_f32, x)
def gamma(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.gamma_f32, x)
def lgamma(x: np.ndarray) -> np.ndarray: return _apply_1d(_C.lgamma_f32, x)

# -----------------------------------------------------------------------------
# ML Activations & Loss Functions
# -----------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function 1 / (1 + exp(-x)) on CUDA."""
    return _apply_1d(_C.sigmoid_f32, x)

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit max(0, x) on CUDA."""
    return _apply_1d(_C.relu_f32, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation on CUDA."""
    return _apply_1d(_C.leaky_relu_f32, x, float(alpha))

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation on CUDA."""
    return _apply_1d(_C.gelu_f32, x)

def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit (SiLU / Swish) activation on CUDA."""
    return _apply_1d(_C.silu_f32, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along axis (default last axis) on CUDA with numerically stable max subtraction."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr2d = arr.reshape(1, -1)
        res = _C.softmax_2d_f32(arr2d)
        return res.reshape(-1)
    elif arr.ndim == 2:
        if axis == -1 or axis == 1:
            return _C.softmax_2d_f32(arr)
        else:
            return _C.transpose_f32(_C.softmax_2d_f32(_C.transpose_f32(arr)))
    else:
        raise ValueError("softmax supports 1-D and 2-D arrays")

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log-softmax along axis on CUDA."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr2d = arr.reshape(1, -1)
        res = _C.log_softmax_2d_f32(arr2d)
        return res.reshape(-1)
    elif arr.ndim == 2:
        if axis == -1 or axis == 1:
            return _C.log_softmax_2d_f32(arr)
        else:
            return _C.transpose_f32(_C.log_softmax_2d_f32(_C.transpose_f32(arr)))
    else:
        raise ValueError("log_softmax supports 1-D and 2-D arrays")

def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """LogSumExp reduction along rows on CUDA."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr2d = arr.reshape(1, -1)
        return _C.logsumexp_2d_f32(arr2d)[0]
    elif arr.ndim == 2:
        if axis == -1 or axis == 1:
            return _C.logsumexp_2d_f32(arr)
        else:
            return _C.logsumexp_2d_f32(_C.transpose_f32(arr))
    else:
        raise ValueError("logsumexp supports 1-D and 2-D arrays")

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross entropy loss on CUDA."""
    return _C.binary_cross_entropy_f32(
        np.ascontiguousarray(y_pred, dtype=np.float32).reshape(-1),
        np.ascontiguousarray(y_true, dtype=np.float32).reshape(-1),
        float(eps),
    )

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss on CUDA."""
    return _C.mse_loss_f32(
        np.ascontiguousarray(y_pred, dtype=np.float32).reshape(-1),
        np.ascontiguousarray(y_true, dtype=np.float32).reshape(-1),
    )

# Aliases for compatibility
mean_squared_error = mse_loss
cuda_mse_loss_f32 = mse_loss
bce_loss = binary_cross_entropy

# -----------------------------------------------------------------------------
# Reductions & Order Statistics
# -----------------------------------------------------------------------------

def sum(x: np.ndarray) -> float:
    return _C.sum_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def min(x: np.ndarray) -> float:
    return _C.min_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def max(x: np.ndarray) -> float:
    return _C.max_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def mean(x: np.ndarray) -> float:
    return _C.mean_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def var(x: np.ndarray, ddof: int = 0) -> float:
    return _C.var_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1), int(ddof))

def std(x: np.ndarray, ddof: int = 0) -> float:
    return _C.std_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1), int(ddof))

def argmin(x: np.ndarray) -> int:
    return _C.argmin_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def argmax(x: np.ndarray) -> int:
    return _C.argmax_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def count_nonzero(x: np.ndarray) -> int:
    return _C.count_nonzero_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def all(x: np.ndarray) -> bool:
    return _C.all_f32(np.ascontiguousarray(x, dtype=bool).reshape(-1))

def any(x: np.ndarray) -> bool:
    return _C.any_f32(np.ascontiguousarray(x, dtype=bool).reshape(-1))

# -----------------------------------------------------------------------------
# Linear Algebra & Matrices
# -----------------------------------------------------------------------------

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """1-D inner dot product on CUDA."""
    return _C.dot_f32(np.ascontiguousarray(a, dtype=np.float32).reshape(-1),
                      np.ascontiguousarray(b, dtype=np.float32).reshape(-1))

def norm(x: np.ndarray, ord: int = 2) -> float:
    """Vector / Frobenius norm on CUDA (ord=1, 2, or 0 for inf)."""
    return _C.norm_f32(np.ascontiguousarray(x, dtype=np.float32), int(ord))

def trace(x: np.ndarray) -> float:
    """Trace (sum of diagonal) of square matrix on CUDA."""
    return _C.trace_f32(np.ascontiguousarray(x, dtype=np.float32))

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of vectors on CUDA."""
    return _C.outer_f32(np.ascontiguousarray(a, dtype=np.float32).reshape(-1),
                        np.ascontiguousarray(b, dtype=np.float32).reshape(-1))

def matvec(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector multiplication A @ x on CUDA."""
    return _C.matvec_f32(np.ascontiguousarray(a, dtype=np.float32),
                         np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D Matrix multiplication A @ B on CUDA."""
    return _C.matmul_f32(np.ascontiguousarray(a, dtype=np.float32),
                         np.ascontiguousarray(b, dtype=np.float32))

def transpose(x: np.ndarray) -> np.ndarray:
    """2-D Matrix transpose on CUDA."""
    return _C.transpose_f32(np.ascontiguousarray(x, dtype=np.float32))

def cov(x: np.ndarray) -> np.ndarray:
    """Sample covariance matrix of shape (n_features, n_features) on CUDA."""
    return _C.cov_f32(np.ascontiguousarray(x, dtype=np.float32))

def corrcoef(x: np.ndarray) -> np.ndarray:
    """Correlation matrix of shape (n_features, n_features) on CUDA."""
    return _C.corrcoef_f32(np.ascontiguousarray(x, dtype=np.float32))

def solve_symmetric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve linear system A w = b on CUDA using GPU Gauss-Jordan elimination."""
    return _C.solve_symmetric_f32(np.ascontiguousarray(a, dtype=np.float32),
                                  np.ascontiguousarray(b, dtype=np.float32).reshape(-1))

# -----------------------------------------------------------------------------
# Spatial Distance Metrics
# -----------------------------------------------------------------------------

def cdist(xa: np.ndarray, xb: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix between XA (M, D) and XB (N, D) on CUDA.
    
    Supported metrics: 'euclidean', 'sqeuclidean', 'cosine', 'cityblock'/'manhattan'.
    """
    return _C.cdist_f32(np.ascontiguousarray(xa, dtype=np.float32),
                        np.ascontiguousarray(xb, dtype=np.float32),
                        str(metric))

# -----------------------------------------------------------------------------
# Scans, Sorting, Search, Differences
# -----------------------------------------------------------------------------

def cumsum(x: np.ndarray) -> np.ndarray:
    return _C.cumsum_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def cumprod(x: np.ndarray) -> np.ndarray:
    return _C.cumprod_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def diff(x: np.ndarray) -> np.ndarray:
    return _C.diff_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def trapezoid(y: np.ndarray, x: Optional[np.ndarray] = None, dx: float = 1.0) -> float:
    return _C.trapezoid_f32(np.ascontiguousarray(y, dtype=np.float32).reshape(-1),
                            np.ascontiguousarray(x, dtype=np.float32).reshape(-1) if x is not None else None,
                            float(dx))

def sort(x: np.ndarray) -> np.ndarray:
    return _C.sort_f32(np.ascontiguousarray(x, dtype=np.float32).reshape(-1))

def searchsorted(sorted_arr: np.ndarray, values: np.ndarray, side: str = "left") -> np.ndarray:
    return _C.searchsorted_f32(np.ascontiguousarray(sorted_arr, dtype=np.float32).reshape(-1),
                               np.ascontiguousarray(values, dtype=np.float32).reshape(-1),
                               str(side))

def bincount_u8_256(x: np.ndarray) -> np.ndarray:
    return _C.bincount_u8_256(np.ascontiguousarray(x, dtype=np.uint8).reshape(-1))

def correlate2d_5x5(image: np.ndarray, filter_5x5: np.ndarray) -> np.ndarray:
    return _C.correlate2d_5x5_f32(np.ascontiguousarray(image, dtype=np.float32),
                                 np.ascontiguousarray(filter_5x5, dtype=np.float32))

correlate2d = correlate2d_5x5

def euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances on CUDA."""
    return cdist(X, Y, metric="euclidean")

def cosine_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise Cosine distances on CUDA."""
    return cdist(X, Y, metric="cosine")

def manhattan_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise Manhattan (L1 / cityblock) distances on CUDA."""
    return cdist(X, Y, metric="cityblock")

# -----------------------------------------------------------------------------
# High-Level Machine Learning Estimators (Scikit-Learn Compatible)
# -----------------------------------------------------------------------------

class KMeans:
    """K-Means clustering powered by CUDA Lloyd's iteration.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Relative tolerance with regards to Frobenius norm of cluster center shift.
    random_state : int
        Random seed for initial centroid selection.
    """
    def __init__(self, n_clusters: int = 8, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray) -> "KMeans":
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        centers, labels, inertia = _C.kmeans_fit_f32(
            X_arr, self.n_clusters, self.max_iter, self.tol, self.random_state
        )
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("KMeans instance is not fitted yet.")
        dists = cdist(X, self.cluster_centers_, metric="sqeuclidean")
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space."""
        if self.cluster_centers_ is None:
            raise ValueError("KMeans instance is not fitted yet.")
        return cdist(X, self.cluster_centers_, metric="euclidean")


class PCA:
    """Principal Component Analysis powered by CUDA Covariance and Power Iteration.
    
    Parameters
    ----------
    n_components : int
        Number of components to keep.
    max_iter : int
        Maximum power iteration steps per component.
    tol : float
        Tolerance for eigenvalue/eigenvector convergence.
    """
    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        self.mean_ = np.mean(X_arr, axis=0)
        comps, ev, _ = _C.pca_fit_transform_f32(
            X_arr, self.n_components, self.max_iter, self.tol
        )
        self.components_ = comps
        self.explained_variance_ = ev
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        self.mean_ = np.mean(X_arr, axis=0)
        comps, ev, trans = _C.pca_fit_transform_f32(
            X_arr, self.n_components, self.max_iter, self.tol
        )
        self.components_ = comps
        self.explained_variance_ = ev
        return trans

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA instance is not fitted yet.")
        X_c = np.ascontiguousarray(X, dtype=np.float32) - self.mean_
        return matmul(X_c, transpose(self.components_))


class LinearRegression:
    """Ordinary Least Squares & Ridge Regression solved on CUDA.
    
    Solves w = (X^T X + alpha * I)^{-1} X^T y on GPU.
    """
    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        y_arr = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        n, d = X_arr.shape

        if self.fit_intercept:
            ones = np.ones((n, 1), dtype=np.float32)
            X_design = np.hstack([X_arr, ones])
            p = d + 1
        else:
            X_design = X_arr
            p = d

        # Compute X^T X on GPU
        Xt = transpose(X_design)
        XtX = matmul(Xt, X_design)

        # Ridge penalty (do not penalize intercept)
        if self.alpha > 0.0:
            eye = np.eye(p, dtype=np.float32)
            if self.fit_intercept:
                eye[-1, -1] = 0.0
            XtX = XtX + self.alpha * eye

        # Compute X^T y on GPU
        Xty = matvec(Xt, y_arr)

        # Solve (X^T X) w = X^T y on GPU
        w = solve_symmetric(XtX, Xty)

        if self.fit_intercept:
            self.coef_ = w[:d]
            self.intercept_ = float(w[d])
        else:
            self.coef_ = w
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        pred = matvec(X_arr, self.coef_)
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


class LogisticRegression:
    """Binary Logistic Regression with GPU Gradient Descent.
    
    Parameters
    ----------
    lr : float
        Learning rate.
    max_iter : int
        Number of gradient descent iterations.
    reg : float
        L2 regularization weight.
    """
    def __init__(self, lr: float = 0.1, max_iter: int = 500, reg: float = 0.0):
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.reg = float(reg)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        y_arr = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        w, b = _C.logistic_regression_fit_f32(
            X_arr, y_arr, self.lr, self.max_iter, self.reg
        )
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        z = matvec(X_arr, self.coef_) + self.intercept_
        p1 = sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(np.int32)


class KNeighborsClassifier:
    """K-Nearest Neighbors Classifier powered by GPU cdist.
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors k.
    metric : str
        Distance metric ('euclidean', 'cosine', 'cityblock', etc.).
    """
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        self.X_train_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train_ = np.ascontiguousarray(y).reshape(-1)
        return self

    def kneighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None:
            raise ValueError("Model is not fitted yet.")
        dists = cdist(X, self.X_train_, metric=self.metric)
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        row_idx = np.arange(X.shape[0])[:, None]
        top_dists = dists[row_idx, idx]
        return top_dists, idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, idx = self.kneighbors(X)
        neighbor_labels = self.y_train_[idx]
        preds = []
        for row in neighbor_labels:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)


class KNeighborsRegressor:
    """K-Nearest Neighbors Regressor powered by GPU cdist."""
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsRegressor":
        self.X_train_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train_ = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Model is not fitted yet.")
        dists = cdist(X, self.X_train_, metric=self.metric)
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        neighbor_vals = self.y_train_[idx]
        return np.mean(neighbor_vals, axis=1)


class GaussianNB:
    """Gaussian Naive Bayes classifier on CUDA.
    
    Computes class priors, feature means and variances on GPU.
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # means (n_classes, n_features)
        self.var_: Optional[np.ndarray] = None    # vars (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        y_arr = np.ascontiguousarray(y).reshape(-1)
        classes, counts = np.unique(y_arr, return_counts=True)
        self.classes_ = classes
        n_classes = len(classes)
        n_features = X_arr.shape[1]

        self.class_prior_ = counts.astype(np.float32) / len(y_arr)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float32)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float32)

        for idx, c in enumerate(classes):
            X_c = X_arr[y_arr == c]
            self.theta_[idx] = np.mean(X_c, axis=0)
            self.var_[idx] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_classes):
            prior = np.log(self.class_prior_[i])
            # Gaussian log-density: -0.5 * sum(log(2*pi*var) + (x - theta)^2 / var)
            diff = X_arr - self.theta_[i]
            var_eps = self.var_[i]
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * var_eps) + (diff ** 2) / var_eps, axis=1)
            joint_log_likelihood[:, i] = prior + n_ij

        # LogSumExp normalization across classes
        log_prob_x = logsumexp(joint_log_likelihood, axis=1)
        return joint_log_likelihood - log_prob_x[:, None]

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_prob = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_prob, axis=1)]

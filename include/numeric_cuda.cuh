#pragma once

#include <cstddef>

// --- Elementwise Arithmetic & Logic ---
void cuda_vector_add_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_subtract_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_multiply_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_divide_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_maximum_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_minimum_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_vector_power_f32(const float* a, const float* b, float* out, std::size_t n);
void cuda_greater_f32(const float* a, const float* b, bool* out, std::size_t n);
void cuda_less_f32(const float* a, const float* b, bool* out, std::size_t n);
void cuda_equal_f32(const float* a, const float* b, bool* out, std::size_t n);
void cuda_logical_and_f32(const bool* a, const bool* b, bool* out, std::size_t n);
void cuda_logical_or_f32(const bool* a, const bool* b, bool* out, std::size_t n);
void cuda_logical_not_f32(const bool* a, bool* out, std::size_t n);
void cuda_where_f32(const bool* cond, const float* a, const float* b, float* out, std::size_t n);

// --- Elementwise Math & Special Functions ---
void cuda_exp_f32(const float* x, float* out, std::size_t n);
void cuda_sqrt_f32(const float* x, float* out, std::size_t n);
void cuda_log_f32(const float* x, float* out, std::size_t n);
void cuda_log1p_f32(const float* x, float* out, std::size_t n);
void cuda_expm1_f32(const float* x, float* out, std::size_t n);
void cuda_sin_f32(const float* x, float* out, std::size_t n);
void cuda_cos_f32(const float* x, float* out, std::size_t n);
void cuda_tan_f32(const float* x, float* out, std::size_t n);
void cuda_tanh_f32(const float* x, float* out, std::size_t n);
void cuda_floor_f32(const float* x, float* out, std::size_t n);
void cuda_ceil_f32(const float* x, float* out, std::size_t n);
void cuda_abs_f32(const float* x, float* out, std::size_t n);
void cuda_negative_f32(const float* x, float* out, std::size_t n);
void cuda_square_f32(const float* x, float* out, std::size_t n);
void cuda_sign_f32(const float* x, float* out, std::size_t n);
void cuda_clip_f32(const float* x, float* out, std::size_t n, float lo, float hi);
void cuda_erf_f32(const float* x, float* out, std::size_t n);
void cuda_erfc_f32(const float* x, float* out, std::size_t n);
void cuda_gamma_f32(const float* x, float* out, std::size_t n);
void cuda_lgamma_f32(const float* x, float* out, std::size_t n);

// --- ML Activations & Loss Functions ---
void cuda_sigmoid_f32(const float* x, float* out, std::size_t n);
void cuda_relu_f32(const float* x, float* out, std::size_t n);
void cuda_leaky_relu_f32(const float* x, float* out, std::size_t n, float alpha);
void cuda_gelu_f32(const float* x, float* out, std::size_t n);
void cuda_silu_f32(const float* x, float* out, std::size_t n);
void cuda_softmax_2d_f32(const float* x, float* out, int rows, int cols);
void cuda_log_softmax_2d_f32(const float* x, float* out, int rows, int cols);
void cuda_logsumexp_2d_f32(const float* x, float* out, int rows, int cols);
double cuda_binary_cross_entropy_f32(const float* y_pred, const float* y_true, std::size_t n, float eps);
double cuda_mse_loss_f32(const float* y_pred, const float* y_true, std::size_t n);

// --- Reductions & Order Statistics ---
double cuda_sum_f32(const float* x, std::size_t n);
float cuda_min_f32(const float* x, std::size_t n);
float cuda_max_f32(const float* x, std::size_t n);
double cuda_mean_f32(const float* x, std::size_t n);
double cuda_var_f32(const float* x, std::size_t n, int ddof);
std::size_t cuda_argmin_f32(const float* x, std::size_t n);
std::size_t cuda_argmax_f32(const float* x, std::size_t n);
std::size_t cuda_count_nonzero_f32(const float* x, std::size_t n);
bool cuda_all_f32(const bool* x, std::size_t n);
bool cuda_any_f32(const bool* x, std::size_t n);

// --- Scans & Differences ---
void cuda_cumsum_f32(const float* in, float* out, int n);
void cuda_cumprod_f32(const float* in, float* out, int n);
void cuda_diff_f32(const float* in, float* out, int n);
double cuda_trapezoid_f32(const float* y, const float* x, int n, float dx);

// --- Sorting, Searching & Histograms ---
void cuda_sort_f32(float* values, int n);
void cuda_searchsorted_f32(const float* sorted_arr, int n_sorted, const float* values, int* out_indices, int n_val, bool right);
void cuda_bincount_u8_256(const unsigned char* x, unsigned int* out, std::size_t n);

// --- Linear Algebra & Matrix Math ---
void cuda_matmul_f32(const float* a, const float* b, float* c, int m, int n, int k);
void cuda_transpose_f32(const float* in, float* out, int rows, int cols);
double cuda_dot_f32(const float* a, const float* b, std::size_t n);
void cuda_outer_f32(const float* a, const float* b, float* out, int m, int n);
void cuda_matvec_f32(const float* a, const float* x, float* out, int m, int k);
double cuda_norm_f32(const float* x, std::size_t n, int ord);
double cuda_trace_f32(const float* x, int n);
void cuda_cov_f32(const float* x, float* out, int n_samples, int n_features);
void cuda_corrcoef_f32(const float* x, float* out, int n_samples, int n_features);
void cuda_solve_symmetric_f32(const float* a, const float* b, float* out, int n);

// --- Spatial / Distance Metrics ---
void cuda_cdist_f32(const float* xa, const float* xb, float* out, int m, int n, int d, int metric);

// --- Image / Stencil Filtering ---
void cuda_correlate2d_5x5_f32(const float* image, const float* filter, float* out, int height, int width);

// --- Full ML Algorithms on GPU ---
void cuda_kmeans_fit_f32(const float* x, float* centroids, int* labels, float* out_inertia,
                         int n_samples, int n_features, int k, int max_iters, float tol, unsigned int seed);
void cuda_pca_fit_transform_f32(const float* x, float* components, float* explained_var, float* x_trans,
                                int n_samples, int n_features, int n_components, int max_iters, float tol);
void cuda_logistic_regression_fit_f32(const float* x, const float* y, float* w, float* b_out,
                                      int n_samples, int n_features, float lr, int iters, float reg);

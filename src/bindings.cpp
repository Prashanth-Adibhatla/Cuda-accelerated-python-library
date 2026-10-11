#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
#include "../include/cuda_raytracer.cuh"
#include "../include/numeric_cuda.cuh"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using UInt8Array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;
using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

static void require_same_1d(const py::buffer_info& a, const py::buffer_info& b) {
    if (a.ndim != 1 || b.ndim != 1 || a.shape[0] != b.shape[0]) {
        throw std::runtime_error("Inputs must be contiguous 1-D arrays with the same length");
    }
}

static void require_2d(const py::buffer_info& a, const char* name) {
    if (a.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be a contiguous 2-D array");
    }
}

// ============================================================================
// Ray Tracer Pybind11 Wrappers
// ============================================================================

py::array_t<float> py_render_scene(
    int width,
    int height,
    int depth_max,
    float ambient,
    float specular_k,
    std::vector<float> camera_pos,
    std::vector<float> light_pos,
    std::vector<float> light_col,
    std::vector<float> screen_bounds,
    py::list py_spheres,
    py::list py_planes
) {
    RenderParams params;
    params.width = width;
    params.height = height;
    params.depth_max = depth_max;
    params.ambient = ambient;
    params.specular_k = specular_k;

    params.camera_O = Vec3(camera_pos[0], camera_pos[1], camera_pos[2]);
    params.light_L = Vec3(light_pos[0], light_pos[1], light_pos[2]);
    params.light_color = Vec3(light_col[0], light_col[1], light_col[2]);

    for (int i = 0; i < 4; ++i) {
        params.S[i] = screen_bounds[i];
    }

    std::vector<Sphere> spheres;
    for (auto item : py_spheres) {
        py::dict d = item.cast<py::dict>();
        std::vector<float> pos = d["position"].cast<std::vector<float>>();
        float radius = d["radius"].cast<float>();
        std::vector<float> col = d["color"].cast<std::vector<float>>();
        float refl = d.contains("reflection") ? d["reflection"].cast<float>() : 0.5f;
        float diff = d.contains("diffuse_c") ? d["diffuse_c"].cast<float>() : 1.0f;
        float spec = d.contains("specular_c") ? d["specular_c"].cast<float>() : 1.0f;

        spheres.emplace_back(
            Vec3(pos[0], pos[1], pos[2]),
            radius,
            Vec3(col[0], col[1], col[2]),
            refl, diff, spec
        );
    }

    std::vector<Plane> planes;
    for (auto item : py_planes) {
        py::dict d = item.cast<py::dict>();
        std::vector<float> pos = d["position"].cast<std::vector<float>>();
        std::vector<float> norm = d["normal"].cast<std::vector<float>>();
        std::vector<float> c0 = d.contains("color0") ? d["color0"].cast<std::vector<float>>() : std::vector<float>{1.0f, 1.0f, 1.0f};
        std::vector<float> c1 = d.contains("color1") ? d["color1"].cast<std::vector<float>>() : std::vector<float>{0.0f, 0.0f, 0.0f};
        float refl = d.contains("reflection") ? d["reflection"].cast<float>() : 0.25f;
        float diff = d.contains("diffuse_c") ? d["diffuse_c"].cast<float>() : 0.75f;
        float spec = d.contains("specular_c") ? d["specular_c"].cast<float>() : 0.5f;

        planes.emplace_back(
            Vec3(pos[0], pos[1], pos[2]),
            Vec3(norm[0], norm[1], norm[2]),
            Vec3(c0[0], c0[1], c0[2]),
            Vec3(c1[0], c1[1], c1[2]),
            refl, diff, spec
        );
    }

    py::array_t<float> result({height, width, 3});
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);

    render_scene_cuda(params, spheres, planes, ptr);

    return result;
}

py::array_t<float> py_parallel_normalize(py::array_t<float> in_arr) {
    py::buffer_info buf = in_arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input array must have shape (N, 3)");
    }
    int count = static_cast<int>(buf.shape[0]);
    py::array_t<float> out_arr({count, 3});
    py::buffer_info out_buf = out_arr.request();

    parallel_normalize_cuda(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(out_buf.ptr),
        count
    );

    return out_arr;
}

py::array_t<float> py_parallel_intersect_spheres(
    py::array_t<float> py_origins,
    py::array_t<float> py_dirs,
    py::list py_spheres
) {
    py::buffer_info buf_orig = py_origins.request();
    py::buffer_info buf_dirs = py_dirs.request();
    int num_rays = static_cast<int>(buf_orig.shape[0]);

    std::vector<Sphere> spheres;
    for (auto item : py_spheres) {
        py::dict d = item.cast<py::dict>();
        std::vector<float> pos = d["position"].cast<std::vector<float>>();
        float radius = d["radius"].cast<float>();
        spheres.emplace_back(Vec3(pos[0], pos[1], pos[2]), radius, Vec3(1.0f, 1.0f, 1.0f));
    }

    py::array_t<float> out_dist(num_rays);
    py::buffer_info out_buf = out_dist.request();

    parallel_intersect_spheres_cuda(
        static_cast<const float*>(buf_orig.ptr),
        static_cast<const float*>(buf_dirs.ptr),
        spheres.data(),
        static_cast<int>(spheres.size()),
        static_cast<float*>(out_buf.ptr),
        num_rays
    );

    return out_dist;
}

py::array_t<float> py_parallel_intersect_planes(
    py::array_t<float> py_origins,
    py::array_t<float> py_dirs,
    py::list py_planes
) {
    py::buffer_info buf_orig = py_origins.request();
    py::buffer_info buf_dirs = py_dirs.request();
    int num_rays = static_cast<int>(buf_orig.shape[0]);

    std::vector<Plane> planes;
    for (auto item : py_planes) {
        py::dict d = item.cast<py::dict>();
        std::vector<float> pos = d["position"].cast<std::vector<float>>();
        std::vector<float> norm = d["normal"].cast<std::vector<float>>();
        planes.emplace_back(Vec3(pos[0], pos[1], pos[2]), Vec3(norm[0], norm[1], norm[2]),
                            Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f));
    }

    py::array_t<float> out_dist(num_rays);
    py::buffer_info out_buf = out_dist.request();

    parallel_intersect_planes_cuda(
        static_cast<const float*>(buf_orig.ptr),
        static_cast<const float*>(buf_dirs.ptr),
        planes.data(),
        static_cast<int>(planes.size()),
        static_cast<float*>(out_buf.ptr),
        num_rays
    );

    return out_dist;
}

// ============================================================================
// Elementwise Math Pybind11 Helpers
// ============================================================================

template <typename KernelFn>
static py::array_t<float> py_binary_f32(FloatArray a, FloatArray b, KernelFn fn) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    require_same_1d(abuf, bbuf);
    py::array_t<float> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    fn(static_cast<const float*>(abuf.ptr),
       static_cast<const float*>(bbuf.ptr),
       static_cast<float*>(obuf.ptr),
       static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

template <typename KernelFn>
static py::array_t<bool> py_compare_f32(FloatArray a, FloatArray b, KernelFn fn) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    require_same_1d(abuf, bbuf);
    py::array_t<bool> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    fn(static_cast<const float*>(abuf.ptr),
       static_cast<const float*>(bbuf.ptr),
       static_cast<bool*>(obuf.ptr),
       static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

template <typename KernelFn>
static py::array_t<float> py_unary_f32(FloatArray x, KernelFn fn) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) {
        throw std::runtime_error("Input must be a contiguous 1-D array");
    }
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    fn(static_cast<const float*>(xbuf.ptr),
       static_cast<float*>(obuf.ptr),
       static_cast<std::size_t>(xbuf.shape[0]));
    return out;
}

py::array_t<float> py_add_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_add_f32); }
py::array_t<float> py_subtract_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_subtract_f32); }
py::array_t<float> py_multiply_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_multiply_f32); }
py::array_t<float> py_divide_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_divide_f32); }
py::array_t<float> py_maximum_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_maximum_f32); }
py::array_t<float> py_minimum_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_minimum_f32); }
py::array_t<float> py_power_f32(FloatArray a, FloatArray b) { return py_binary_f32(a, b, cuda_vector_power_f32); }

py::array_t<bool> py_greater_f32(FloatArray a, FloatArray b) { return py_compare_f32(a, b, cuda_greater_f32); }
py::array_t<bool> py_less_f32(FloatArray a, FloatArray b) { return py_compare_f32(a, b, cuda_less_f32); }
py::array_t<bool> py_equal_f32(FloatArray a, FloatArray b) { return py_compare_f32(a, b, cuda_equal_f32); }

py::array_t<bool> py_logical_and_f32(BoolArray a, BoolArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    if (abuf.ndim != 1 || bbuf.ndim != 1 || abuf.shape[0] != bbuf.shape[0]) {
        throw std::runtime_error("Inputs must be 1-D bool arrays of the same length");
    }
    py::array_t<bool> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_logical_and_f32(static_cast<const bool*>(abuf.ptr), static_cast<const bool*>(bbuf.ptr),
                         static_cast<bool*>(obuf.ptr), static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

py::array_t<bool> py_logical_or_f32(BoolArray a, BoolArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    if (abuf.ndim != 1 || bbuf.ndim != 1 || abuf.shape[0] != bbuf.shape[0]) {
        throw std::runtime_error("Inputs must be 1-D bool arrays of the same length");
    }
    py::array_t<bool> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_logical_or_f32(static_cast<const bool*>(abuf.ptr), static_cast<const bool*>(bbuf.ptr),
                        static_cast<bool*>(obuf.ptr), static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

py::array_t<bool> py_logical_not_f32(BoolArray a) {
    py::buffer_info abuf = a.request();
    if (abuf.ndim != 1) throw std::runtime_error("Input must be a 1-D bool array");
    py::array_t<bool> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_logical_not_f32(static_cast<const bool*>(abuf.ptr), static_cast<bool*>(obuf.ptr),
                         static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

py::array_t<float> py_where_f32(BoolArray cond, FloatArray a, FloatArray b) {
    py::buffer_info cbuf = cond.request();
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    if (cbuf.ndim != 1 || abuf.ndim != 1 || bbuf.ndim != 1 ||
        cbuf.shape[0] != abuf.shape[0] || abuf.shape[0] != bbuf.shape[0]) {
        throw std::runtime_error("cond, a, and b must be contiguous 1-D arrays with matching shapes");
    }
    py::array_t<float> out({abuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_where_f32(static_cast<const bool*>(cbuf.ptr),
                   static_cast<const float*>(abuf.ptr),
                   static_cast<const float*>(bbuf.ptr),
                   static_cast<float*>(obuf.ptr),
                   static_cast<std::size_t>(abuf.shape[0]));
    return out;
}

py::array_t<float> py_exp_f32(FloatArray x) { return py_unary_f32(x, cuda_exp_f32); }
py::array_t<float> py_sqrt_f32(FloatArray x) { return py_unary_f32(x, cuda_sqrt_f32); }
py::array_t<float> py_log_f32(FloatArray x) { return py_unary_f32(x, cuda_log_f32); }
py::array_t<float> py_log1p_f32(FloatArray x) { return py_unary_f32(x, cuda_log1p_f32); }
py::array_t<float> py_expm1_f32(FloatArray x) { return py_unary_f32(x, cuda_expm1_f32); }
py::array_t<float> py_sin_f32(FloatArray x) { return py_unary_f32(x, cuda_sin_f32); }
py::array_t<float> py_cos_f32(FloatArray x) { return py_unary_f32(x, cuda_cos_f32); }
py::array_t<float> py_tan_f32(FloatArray x) { return py_unary_f32(x, cuda_tan_f32); }
py::array_t<float> py_tanh_f32(FloatArray x) { return py_unary_f32(x, cuda_tanh_f32); }
py::array_t<float> py_floor_f32(FloatArray x) { return py_unary_f32(x, cuda_floor_f32); }
py::array_t<float> py_ceil_f32(FloatArray x) { return py_unary_f32(x, cuda_ceil_f32); }
py::array_t<float> py_abs_f32(FloatArray x) { return py_unary_f32(x, cuda_abs_f32); }
py::array_t<float> py_negative_f32(FloatArray x) { return py_unary_f32(x, cuda_negative_f32); }
py::array_t<float> py_square_f32(FloatArray x) { return py_unary_f32(x, cuda_square_f32); }
py::array_t<float> py_sign_f32(FloatArray x) { return py_unary_f32(x, cuda_sign_f32); }
py::array_t<float> py_erf_f32(FloatArray x) { return py_unary_f32(x, cuda_erf_f32); }
py::array_t<float> py_erfc_f32(FloatArray x) { return py_unary_f32(x, cuda_erfc_f32); }
py::array_t<float> py_gamma_f32(FloatArray x) { return py_unary_f32(x, cuda_gamma_f32); }
py::array_t<float> py_lgamma_f32(FloatArray x) { return py_unary_f32(x, cuda_lgamma_f32); }

// ML Activations
py::array_t<float> py_sigmoid_f32(FloatArray x) { return py_unary_f32(x, cuda_sigmoid_f32); }
py::array_t<float> py_relu_f32(FloatArray x) { return py_unary_f32(x, cuda_relu_f32); }
py::array_t<float> py_gelu_f32(FloatArray x) { return py_unary_f32(x, cuda_gelu_f32); }
py::array_t<float> py_silu_f32(FloatArray x) { return py_unary_f32(x, cuda_silu_f32); }

py::array_t<float> py_leaky_relu_f32(FloatArray x, float alpha) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a 1-D array");
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_leaky_relu_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr),
                        static_cast<std::size_t>(xbuf.shape[0]), alpha);
    return out;
}

py::array_t<float> py_clip_f32(FloatArray x, float lo, float hi) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D array");
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_clip_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr),
                  static_cast<std::size_t>(xbuf.shape[0]), lo, hi);
    return out;
}

// 2D Softmax, Log-Softmax, LogSumExp
py::array_t<float> py_softmax_2d_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int rows = static_cast<int>(xbuf.shape[0]);
    int cols = static_cast<int>(xbuf.shape[1]);
    py::array_t<float> out({rows, cols});
    py::buffer_info obuf = out.request();
    cuda_softmax_2d_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), rows, cols);
    return out;
}

py::array_t<float> py_log_softmax_2d_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int rows = static_cast<int>(xbuf.shape[0]);
    int cols = static_cast<int>(xbuf.shape[1]);
    py::array_t<float> out({rows, cols});
    py::buffer_info obuf = out.request();
    cuda_log_softmax_2d_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), rows, cols);
    return out;
}

py::array_t<float> py_logsumexp_2d_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int rows = static_cast<int>(xbuf.shape[0]);
    int cols = static_cast<int>(xbuf.shape[1]);
    py::array_t<float> out({rows});
    py::buffer_info obuf = out.request();
    cuda_logsumexp_2d_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), rows, cols);
    return out;
}

// Loss functions
double py_binary_cross_entropy_f32(FloatArray y_pred, FloatArray y_true, float eps) {
    py::buffer_info pbuf = y_pred.request();
    py::buffer_info tbuf = y_true.request();
    require_same_1d(pbuf, tbuf);
    return cuda_binary_cross_entropy_f32(static_cast<const float*>(pbuf.ptr),
                                         static_cast<const float*>(tbuf.ptr),
                                         static_cast<std::size_t>(pbuf.shape[0]),
                                         eps);
}

double py_mse_loss_f32(FloatArray y_pred, FloatArray y_true) {
    py::buffer_info pbuf = y_pred.request();
    py::buffer_info tbuf = y_true.request();
    require_same_1d(pbuf, tbuf);
    return cuda_mse_loss_f32(static_cast<const float*>(pbuf.ptr),
                             static_cast<const float*>(tbuf.ptr),
                             static_cast<std::size_t>(pbuf.shape[0]));
}

// Reductions
double py_sum_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D array");
    return cuda_sum_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

float py_min_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    return cuda_min_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

float py_max_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    return cuda_max_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

double py_mean_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    return cuda_mean_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

double py_var_f32(FloatArray x, int ddof) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    if (ddof < 0) throw std::runtime_error("ddof must be non-negative");
    return cuda_var_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]), ddof);
}

double py_std_f32(FloatArray x, int ddof) {
    return std::sqrt(py_var_f32(x, ddof));
}

std::size_t py_argmin_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    return cuda_argmin_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

std::size_t py_argmax_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] == 0) throw std::runtime_error("Input must be a non-empty contiguous 1-D array");
    return cuda_argmax_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

std::size_t py_count_nonzero_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a 1-D array");
    return cuda_count_nonzero_f32(static_cast<const float*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

bool py_all_f32(BoolArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a 1-D bool array");
    return cuda_all_f32(static_cast<const bool*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

bool py_any_f32(BoolArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a 1-D bool array");
    return cuda_any_f32(static_cast<const bool*>(xbuf.ptr), static_cast<std::size_t>(xbuf.shape[0]));
}

// Linear Algebra
double py_dot_f32(FloatArray a, FloatArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    require_same_1d(abuf, bbuf);
    return cuda_dot_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(bbuf.ptr),
                        static_cast<std::size_t>(abuf.shape[0]));
}

double py_norm_f32(FloatArray x, int ord) {
    py::buffer_info xbuf = x.request();
    std::size_t total = 1;
    for (int i = 0; i < xbuf.ndim; ++i) total *= static_cast<std::size_t>(xbuf.shape[i]);
    return cuda_norm_f32(static_cast<const float*>(xbuf.ptr), total, ord);
}

double py_trace_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    if (xbuf.shape[0] != xbuf.shape[1]) throw std::runtime_error("Input must be a square 2-D matrix");
    return cuda_trace_f32(static_cast<const float*>(xbuf.ptr), static_cast<int>(xbuf.shape[0]));
}

py::array_t<float> py_outer_f32(FloatArray a, FloatArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    if (abuf.ndim != 1 || bbuf.ndim != 1) throw std::runtime_error("Inputs must be 1-D arrays");
    int m = static_cast<int>(abuf.shape[0]);
    int n = static_cast<int>(bbuf.shape[0]);
    py::array_t<float> out({m, n});
    py::buffer_info obuf = out.request();
    cuda_outer_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(bbuf.ptr),
                   static_cast<float*>(obuf.ptr), m, n);
    return out;
}

py::array_t<float> py_matvec_f32(FloatArray a, FloatArray x) {
    py::buffer_info abuf = a.request();
    py::buffer_info xbuf = x.request();
    require_2d(abuf, "a");
    if (xbuf.ndim != 1) throw std::runtime_error("x must be a 1-D array");
    int m = static_cast<int>(abuf.shape[0]);
    int k = static_cast<int>(abuf.shape[1]);
    if (static_cast<int>(xbuf.shape[0]) != k) throw std::runtime_error("Dimensions must match for matvec");
    py::array_t<float> out({m});
    py::buffer_info obuf = out.request();
    cuda_matvec_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(xbuf.ptr),
                    static_cast<float*>(obuf.ptr), m, k);
    return out;
}

py::array_t<float> py_matmul_f32(FloatArray a, FloatArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    require_2d(abuf, "a");
    require_2d(bbuf, "b");
    int m = static_cast<int>(abuf.shape[0]);
    int k = static_cast<int>(abuf.shape[1]);
    int bk = static_cast<int>(bbuf.shape[0]);
    int n = static_cast<int>(bbuf.shape[1]);
    if (k != bk) throw std::runtime_error("Inner dimensions must match for matrix multiplication");
    py::array_t<float> out({m, n});
    py::buffer_info obuf = out.request();
    cuda_matmul_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(bbuf.ptr),
                    static_cast<float*>(obuf.ptr), m, n, k);
    return out;
}

py::array_t<float> py_transpose_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int rows = static_cast<int>(xbuf.shape[0]);
    int cols = static_cast<int>(xbuf.shape[1]);
    py::array_t<float> out({cols, rows});
    py::buffer_info obuf = out.request();
    cuda_transpose_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), rows, cols);
    return out;
}

py::array_t<float> py_cov_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int n_samples = static_cast<int>(xbuf.shape[0]);
    int n_features = static_cast<int>(xbuf.shape[1]);
    if (n_samples <= 1) throw std::runtime_error("Need at least 2 samples for covariance");
    py::array_t<float> out({n_features, n_features});
    py::buffer_info obuf = out.request();
    cuda_cov_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), n_samples, n_features);
    return out;
}

py::array_t<float> py_corrcoef_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int n_samples = static_cast<int>(xbuf.shape[0]);
    int n_features = static_cast<int>(xbuf.shape[1]);
    if (n_samples <= 1) throw std::runtime_error("Need at least 2 samples for correlation");
    py::array_t<float> out({n_features, n_features});
    py::buffer_info obuf = out.request();
    cuda_corrcoef_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), n_samples, n_features);
    return out;
}

py::array_t<float> py_solve_symmetric_f32(FloatArray a, FloatArray b) {
    py::buffer_info abuf = a.request();
    py::buffer_info bbuf = b.request();
    require_2d(abuf, "a");
    if (bbuf.ndim != 1) throw std::runtime_error("b must be a 1-D vector");
    int n = static_cast<int>(abuf.shape[0]);
    if (abuf.shape[1] != n || static_cast<int>(bbuf.shape[0]) != n) {
        throw std::runtime_error("a must be (N, N) and b must be (N)");
    }
    py::array_t<float> out({n});
    py::buffer_info obuf = out.request();
    cuda_solve_symmetric_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(bbuf.ptr),
                             static_cast<float*>(obuf.ptr), n);
    return out;
}

// Distance Metrics
py::array_t<float> py_cdist_f32(FloatArray xa, FloatArray xb, std::string metric) {
    py::buffer_info abuf = xa.request();
    py::buffer_info bbuf = xb.request();
    require_2d(abuf, "xa");
    require_2d(bbuf, "xb");
    int m = static_cast<int>(abuf.shape[0]);
    int d = static_cast<int>(abuf.shape[1]);
    int n = static_cast<int>(bbuf.shape[0]);
    if (static_cast<int>(bbuf.shape[1]) != d) throw std::runtime_error("xa and xb must have same number of columns");
    int m_code = 0;
    if (metric == "euclidean") m_code = 0;
    else if (metric == "sqeuclidean") m_code = 1;
    else if (metric == "cosine") m_code = 2;
    else if (metric == "cityblock" || metric == "manhattan") m_code = 3;
    else throw std::runtime_error("Unsupported metric: " + metric);
    py::array_t<float> out({m, n});
    py::buffer_info obuf = out.request();
    cuda_cdist_f32(static_cast<const float*>(abuf.ptr), static_cast<const float*>(bbuf.ptr),
                   static_cast<float*>(obuf.ptr), m, n, d, m_code);
    return out;
}

// Searchsorted, Diff, Trapezoid
py::array_t<int> py_searchsorted_f32(FloatArray sorted_arr, FloatArray values, std::string side) {
    py::buffer_info sbuf = sorted_arr.request();
    py::buffer_info vbuf = values.request();
    if (sbuf.ndim != 1 || vbuf.ndim != 1) throw std::runtime_error("Inputs must be 1-D arrays");
    bool right = (side == "right");
    py::array_t<int> out({vbuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_searchsorted_f32(static_cast<const float*>(sbuf.ptr), static_cast<int>(sbuf.shape[0]),
                          static_cast<const float*>(vbuf.ptr), static_cast<int*>(obuf.ptr),
                          static_cast<int>(vbuf.shape[0]), right);
    return out;
}

py::array_t<float> py_diff_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1 || xbuf.shape[0] < 2) throw std::runtime_error("Input must be a 1-D array with length >= 2");
    int n_out = static_cast<int>(xbuf.shape[0]) - 1;
    py::array_t<float> out({n_out});
    py::buffer_info obuf = out.request();
    cuda_diff_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), static_cast<int>(xbuf.shape[0]));
    return out;
}

double py_trapezoid_f32(FloatArray y, py::object x_obj, float dx) {
    py::buffer_info ybuf = y.request();
    if (ybuf.ndim != 1) throw std::runtime_error("y must be a 1-D array");
    const float* x_ptr = nullptr;
    if (!x_obj.is_none()) {
        FloatArray x = x_obj.cast<FloatArray>();
        py::buffer_info xbuf = x.request();
        if (xbuf.ndim != 1 || xbuf.shape[0] != ybuf.shape[0]) {
            throw std::runtime_error("x must have same length as y");
        }
        x_ptr = static_cast<const float*>(xbuf.ptr);
    }
    return cuda_trapezoid_f32(static_cast<const float*>(ybuf.ptr), x_ptr,
                              static_cast<int>(ybuf.shape[0]), dx);
}

// Scans, Bincount, Sort, Correlate
py::array_t<float> py_cumsum_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D array");
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_cumsum_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), static_cast<int>(xbuf.shape[0]));
    return out;
}

py::array_t<float> py_cumprod_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D array");
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    cuda_cumprod_f32(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr), static_cast<int>(xbuf.shape[0]));
    return out;
}

py::array_t<unsigned int> py_bincount_u8_256(UInt8Array x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D uint8 array");
    py::array_t<unsigned int> out({256});
    py::buffer_info obuf = out.request();
    cuda_bincount_u8_256(static_cast<const unsigned char*>(xbuf.ptr),
                         static_cast<unsigned int*>(obuf.ptr),
                         static_cast<std::size_t>(xbuf.shape[0]));
    return out;
}

py::array_t<float> py_correlate2d_5x5_f32(FloatArray image, FloatArray filter) {
    py::buffer_info ibuf = image.request();
    py::buffer_info fbuf = filter.request();
    require_2d(ibuf, "image");
    require_2d(fbuf, "filter");
    if (fbuf.shape[0] != 5 || fbuf.shape[1] != 5) throw std::runtime_error("filter must have shape (5, 5)");
    int height = static_cast<int>(ibuf.shape[0]);
    int width = static_cast<int>(ibuf.shape[1]);
    py::array_t<float> out({height, width});
    py::buffer_info obuf = out.request();
    cuda_correlate2d_5x5_f32(static_cast<const float*>(ibuf.ptr), static_cast<const float*>(fbuf.ptr),
                             static_cast<float*>(obuf.ptr), height, width);
    return out;
}

py::array_t<float> py_sort_f32(FloatArray x) {
    py::buffer_info xbuf = x.request();
    if (xbuf.ndim != 1) throw std::runtime_error("Input must be a contiguous 1-D array");
    py::array_t<float> out({xbuf.shape[0]});
    py::buffer_info obuf = out.request();
    std::memcpy(obuf.ptr, xbuf.ptr, static_cast<std::size_t>(xbuf.shape[0]) * sizeof(float));
    cuda_sort_f32(static_cast<float*>(obuf.ptr), static_cast<int>(xbuf.shape[0]));
    return out;
}

// ============================================================================
// Full ML Algorithms Wrappers
// ============================================================================

std::tuple<py::array_t<float>, py::array_t<int>, float>
py_kmeans_fit_f32(FloatArray x, int k, int max_iters, float tol, unsigned int seed) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int n_samples = static_cast<int>(xbuf.shape[0]);
    int n_features = static_cast<int>(xbuf.shape[1]);
    if (k <= 0 || k > n_samples) throw std::runtime_error("k must be in 1..n_samples");

    py::array_t<float> centroids({k, n_features});
    py::array_t<int> labels({n_samples});
    py::buffer_info cbuf = centroids.request();
    py::buffer_info lbuf = labels.request();
    float inertia = 0.0f;

    cuda_kmeans_fit_f32(static_cast<const float*>(xbuf.ptr),
                        static_cast<float*>(cbuf.ptr),
                        static_cast<int*>(lbuf.ptr),
                        &inertia,
                        n_samples, n_features, k, max_iters, tol, seed);

    return std::make_tuple(centroids, labels, inertia);
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
py_pca_fit_transform_f32(FloatArray x, int n_components, int max_iters, float tol) {
    py::buffer_info xbuf = x.request();
    require_2d(xbuf, "x");
    int n_samples = static_cast<int>(xbuf.shape[0]);
    int n_features = static_cast<int>(xbuf.shape[1]);
    if (n_components <= 0 || n_components > n_features) {
        throw std::runtime_error("n_components must be between 1 and n_features");
    }

    py::array_t<float> components({n_components, n_features});
    py::array_t<float> explained_var({n_components});
    py::array_t<float> x_trans({n_samples, n_components});

    py::buffer_info cbuf = components.request();
    py::buffer_info evbuf = explained_var.request();
    py::buffer_info tbuf = x_trans.request();

    cuda_pca_fit_transform_f32(static_cast<const float*>(xbuf.ptr),
                               static_cast<float*>(cbuf.ptr),
                               static_cast<float*>(evbuf.ptr),
                               static_cast<float*>(tbuf.ptr),
                               n_samples, n_features, n_components, max_iters, tol);

    return std::make_tuple(components, explained_var, x_trans);
}

std::pair<py::array_t<float>, float>
py_logistic_regression_fit_f32(FloatArray x, FloatArray y, float lr, int iters, float reg) {
    py::buffer_info xbuf = x.request();
    py::buffer_info ybuf = y.request();
    require_2d(xbuf, "x");
    if (ybuf.ndim != 1 || ybuf.shape[0] != xbuf.shape[0]) {
        throw std::runtime_error("y must be a 1-D vector with same number of rows as x");
    }
    int n_samples = static_cast<int>(xbuf.shape[0]);
    int n_features = static_cast<int>(xbuf.shape[1]);

    py::array_t<float> weights({n_features});
    py::buffer_info wbuf = weights.request();
    float bias = 0.0f;

    cuda_logistic_regression_fit_f32(static_cast<const float*>(xbuf.ptr),
                                     static_cast<const float*>(ybuf.ptr),
                                     static_cast<float*>(wbuf.ptr),
                                     &bias,
                                     n_samples, n_features, lr, iters, reg);

    return std::make_pair(weights, bias);
}

// ============================================================================
// Pybind11 Module Declaration
// ============================================================================

PYBIND11_MODULE(raytracer_cuda, m) {
    m.doc() = "CUDA High-Performance Numeric, Math & Machine Learning Library with Ray Tracing Benchmark";

    // Ray tracing functions
    m.def("render_scene", &py_render_scene, "Render entire scene in parallel on GPU");
    m.def("parallel_normalize", &py_parallel_normalize, "Parallel GPU vector normalization for (N, 3) numpy arrays");
    m.def("parallel_intersect_spheres", &py_parallel_intersect_spheres, "Parallel GPU ray-sphere intersection testing");
    m.def("parallel_intersect_planes", &py_parallel_intersect_planes, "Parallel GPU ray-plane intersection testing");

    // Elementwise Arithmetic & Logic
    m.def("add_f32", &py_add_f32, "Elementwise float32 vector addition on CUDA");
    m.def("subtract_f32", &py_subtract_f32, "Elementwise float32 vector subtraction on CUDA");
    m.def("multiply_f32", &py_multiply_f32, "Elementwise float32 vector multiplication on CUDA");
    m.def("divide_f32", &py_divide_f32, "Elementwise float32 vector division on CUDA");
    m.def("maximum_f32", &py_maximum_f32, "Elementwise float32 maximum on CUDA");
    m.def("minimum_f32", &py_minimum_f32, "Elementwise float32 minimum on CUDA");
    m.def("power_f32", &py_power_f32, "Elementwise float32 power (a ** b) on CUDA");
    m.def("greater_f32", &py_greater_f32, "Elementwise float32 greater-than comparison on CUDA");
    m.def("less_f32", &py_less_f32, "Elementwise float32 less-than comparison on CUDA");
    m.def("equal_f32", &py_equal_f32, "Elementwise float32 equality comparison on CUDA");
    m.def("logical_and_f32", &py_logical_and_f32, "Elementwise logical AND on CUDA");
    m.def("logical_or_f32", &py_logical_or_f32, "Elementwise logical OR on CUDA");
    m.def("logical_not_f32", &py_logical_not_f32, "Elementwise logical NOT on CUDA");
    m.def("where_f32", &py_where_f32, "Elementwise where(cond, a, b) on CUDA");

    // Elementwise Math & Special
    m.def("exp_f32", &py_exp_f32, "Elementwise float32 exponential on CUDA");
    m.def("sqrt_f32", &py_sqrt_f32, "Elementwise float32 square root on CUDA");
    m.def("log_f32", &py_log_f32, "Elementwise float32 natural logarithm on CUDA");
    m.def("log1p_f32", &py_log1p_f32, "Elementwise float32 log(1 + x) on CUDA");
    m.def("expm1_f32", &py_expm1_f32, "Elementwise float32 exp(x) - 1 on CUDA");
    m.def("sin_f32", &py_sin_f32, "Elementwise float32 sine on CUDA");
    m.def("cos_f32", &py_cos_f32, "Elementwise float32 cosine on CUDA");
    m.def("tan_f32", &py_tan_f32, "Elementwise float32 tangent on CUDA");
    m.def("tanh_f32", &py_tanh_f32, "Elementwise float32 hyperbolic tangent on CUDA");
    m.def("floor_f32", &py_floor_f32, "Elementwise float32 floor on CUDA");
    m.def("ceil_f32", &py_ceil_f32, "Elementwise float32 ceil on CUDA");
    m.def("abs_f32", &py_abs_f32, "Elementwise float32 absolute value on CUDA");
    m.def("negative_f32", &py_negative_f32, "Elementwise float32 negation on CUDA");
    m.def("square_f32", &py_square_f32, "Elementwise float32 square on CUDA");
    m.def("sign_f32", &py_sign_f32, "Elementwise float32 signum on CUDA");
    m.def("clip_f32", &py_clip_f32, "Elementwise float32 clipping on CUDA", py::arg("x"), py::arg("lo"), py::arg("hi"));
    m.def("erf_f32", &py_erf_f32, "Elementwise error function on CUDA");
    m.def("erfc_f32", &py_erfc_f32, "Elementwise complementary error function on CUDA");
    m.def("gamma_f32", &py_gamma_f32, "Elementwise gamma function on CUDA");
    m.def("lgamma_f32", &py_lgamma_f32, "Elementwise log-gamma function on CUDA");

    // ML Activations
    m.def("sigmoid_f32", &py_sigmoid_f32, "Elementwise sigmoid activation on CUDA");
    m.def("relu_f32", &py_relu_f32, "Elementwise ReLU activation on CUDA");
    m.def("leaky_relu_f32", &py_leaky_relu_f32, "Elementwise Leaky ReLU activation on CUDA", py::arg("x"), py::arg("alpha") = 0.01f);
    m.def("gelu_f32", &py_gelu_f32, "Elementwise GELU activation on CUDA");
    m.def("silu_f32", &py_silu_f32, "Elementwise SiLU / Swish activation on CUDA");

    // 2D Softmax, Log-Softmax, LogSumExp
    m.def("softmax_2d_f32", &py_softmax_2d_f32, "Row-wise softmax on 2-D float32 array on CUDA");
    m.def("log_softmax_2d_f32", &py_log_softmax_2d_f32, "Row-wise log-softmax on 2-D float32 array on CUDA");
    m.def("logsumexp_2d_f32", &py_logsumexp_2d_f32, "Row-wise logsumexp on 2-D float32 array on CUDA");

    // Loss functions
    m.def("binary_cross_entropy_f32", &py_binary_cross_entropy_f32, "Binary Cross Entropy loss on CUDA",
          py::arg("y_pred"), py::arg("y_true"), py::arg("eps") = 1e-7f);
    m.def("mse_loss_f32", &py_mse_loss_f32, "Mean Squared Error loss on CUDA",
          py::arg("y_pred"), py::arg("y_true"));

    // Reductions & Order Stats
    m.def("sum_f32", &py_sum_f32, "Float32 vector sum on CUDA");
    m.def("min_f32", &py_min_f32, "Float32 vector minimum on CUDA");
    m.def("max_f32", &py_max_f32, "Float32 vector maximum on CUDA");
    m.def("mean_f32", &py_mean_f32, "Float32 vector mean on CUDA");
    m.def("var_f32", &py_var_f32, "Float32 vector variance on CUDA", py::arg("x"), py::arg("ddof") = 0);
    m.def("std_f32", &py_std_f32, "Float32 vector standard deviation on CUDA", py::arg("x"), py::arg("ddof") = 0);
    m.def("argmin_f32", &py_argmin_f32, "Float32 vector argmin on CUDA");
    m.def("argmax_f32", &py_argmax_f32, "Float32 vector argmax on CUDA");
    m.def("count_nonzero_f32", &py_count_nonzero_f32, "Count nonzero elements on CUDA");
    m.def("all_f32", &py_all_f32, "Boolean all reduction on CUDA");
    m.def("any_f32", &py_any_f32, "Boolean any reduction on CUDA");

    // Linear Algebra
    m.def("dot_f32", &py_dot_f32, "1-D dot product on CUDA");
    m.def("norm_f32", &py_norm_f32, "Vector / matrix norm on CUDA (ord=1, 2, or 0 for inf)", py::arg("x"), py::arg("ord") = 2);
    m.def("trace_f32", &py_trace_f32, "Matrix trace (sum of diagonal) on CUDA");
    m.def("outer_f32", &py_outer_f32, "Vector outer product on CUDA");
    m.def("matvec_f32", &py_matvec_f32, "Matrix-vector product on CUDA");
    m.def("matmul_f32", &py_matmul_f32, "Matrix multiplication on CUDA");
    m.def("transpose_f32", &py_transpose_f32, "Matrix transpose on CUDA");
    m.def("cov_f32", &py_cov_f32, "Sample covariance matrix on CUDA");
    m.def("corrcoef_f32", &py_corrcoef_f32, "Correlation matrix on CUDA");
    m.def("solve_symmetric_f32", &py_solve_symmetric_f32, "Solve A w = b for symmetric/positive definite matrix on CUDA");

    // Distance Metrics
    m.def("cdist_f32", &py_cdist_f32, "Pairwise distance matrix on CUDA ('euclidean', 'sqeuclidean', 'cosine', 'cityblock')",
          py::arg("xa"), py::arg("xb"), py::arg("metric") = "euclidean");

    // Search & Scans
    m.def("searchsorted_f32", &py_searchsorted_f32, "Binary search on sorted array on CUDA",
          py::arg("sorted_arr"), py::arg("values"), py::arg("side") = "left");
    m.def("diff_f32", &py_diff_f32, "1-D finite differences on CUDA");
    m.def("trapezoid_f32", &py_trapezoid_f32, "Trapezoidal numerical integration on CUDA",
          py::arg("y"), py::arg("x") = py::none(), py::arg("dx") = 1.0f);
    m.def("cumsum_f32", &py_cumsum_f32, "Prefix sum on CUDA");
    m.def("cumprod_f32", &py_cumprod_f32, "Prefix product on CUDA");
    m.def("bincount_u8_256", &py_bincount_u8_256, "256-bin uint8 bincount on CUDA");
    m.def("sort_f32", &py_sort_f32, "Ascending sort on CUDA");
    m.def("correlate2d_5x5_f32", &py_correlate2d_5x5_f32, "2-D 5x5 constant-padding correlation on CUDA");

    // Full ML Algorithms
    m.def("kmeans_fit_f32", &py_kmeans_fit_f32, "K-Means clustering fit on CUDA (returns centroids, labels, inertia)",
          py::arg("x"), py::arg("k"), py::arg("max_iters") = 100, py::arg("tol") = 1e-4f, py::arg("seed") = 42);
    m.def("pca_fit_transform_f32", &py_pca_fit_transform_f32, "PCA fit and transform on CUDA (returns components, explained_var, x_trans)",
          py::arg("x"), py::arg("n_components"), py::arg("max_iters") = 100, py::arg("tol") = 1e-4f);
    m.def("logistic_regression_fit_f32", &py_logistic_regression_fit_f32, "Logistic regression fit on CUDA (returns weights, bias)",
          py::arg("x"), py::arg("y"), py::arg("lr") = 0.1f, py::arg("iters") = 500, py::arg("reg") = 0.0f);
}

import numpy as np

import raytracer_cuda as rc


def assert_close(actual, expected, **kwargs):
    np.testing.assert_allclose(actual, expected, rtol=kwargs.get("rtol", 1e-5), atol=kwargs.get("atol", 1e-6))


def test_add_exp_sum():
    a = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    b = np.linspace(2.0, -2.0, 1024, dtype=np.float32)
    positive = np.linspace(0.1, 5.0, 1024, dtype=np.float32)

    assert_close(rc.add_f32(a, b), np.add(a, b))
    assert_close(rc.subtract_f32(a, b), np.subtract(a, b))
    assert_close(rc.multiply_f32(a, b), np.multiply(a, b))
    assert_close(rc.divide_f32(a + 3.0, b + 4.0), np.divide(a + 3.0, b + 4.0))
    assert_close(rc.maximum_f32(a, b), np.maximum(a, b))
    assert_close(rc.minimum_f32(a, b), np.minimum(a, b))
    np.testing.assert_array_equal(rc.greater_f32(a, b), np.greater(a, b))
    np.testing.assert_array_equal(rc.less_f32(a, b), np.less(a, b))
    np.testing.assert_array_equal(rc.equal_f32(a, a), np.equal(a, a))
    assert_close(rc.where_f32(a > b, a, b), np.where(a > b, a, b))
    assert_close(rc.exp_f32(a), np.exp(a), rtol=1e-6)
    assert_close(rc.sqrt_f32(positive), np.sqrt(positive), rtol=1e-6)
    assert_close(rc.log_f32(positive), np.log(positive), rtol=1e-6)
    assert_close(rc.sin_f32(a), np.sin(a), rtol=1e-6)
    assert_close(rc.cos_f32(a), np.cos(a), rtol=1e-6)
    assert_close(rc.tan_f32(a / 4.0), np.tan(a / 4.0), rtol=1e-6)
    assert_close(rc.tanh_f32(a), np.tanh(a), rtol=1e-6)
    assert_close(rc.floor_f32(a * 3.0), np.floor(a * 3.0))
    assert_close(rc.ceil_f32(a * 3.0), np.ceil(a * 3.0))
    assert_close(rc.abs_f32(a), np.abs(a))
    assert_close(rc.negative_f32(a), np.negative(a))
    assert_close(rc.square_f32(a), np.square(a))
    assert_close(rc.clip_f32(a, -0.25, 0.5), np.clip(a, -0.25, 0.5))
    assert_close(rc.sum_f32(a), np.sum(a, dtype=np.float32), atol=1e-4)


def test_reductions():
    x = np.array([3.5, -1.0, 2.25, 10.0, 4.0, 10.0], dtype=np.float32)
    many = np.linspace(-5.0, 7.0, 4097, dtype=np.float32)

    assert_close(rc.min_f32(x), np.min(x))
    assert_close(rc.max_f32(x), np.max(x))
    assert_close(rc.mean_f32(many), np.mean(many, dtype=np.float32), atol=1e-5)
    assert_close(rc.var_f32(many), np.var(many, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert_close(rc.var_f32(many, ddof=1), np.var(many, dtype=np.float32, ddof=1), rtol=1e-5, atol=1e-5)
    assert_close(rc.std_f32(many), np.std(many, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert rc.argmin_f32(x) == int(np.argmin(x))
    assert rc.argmax_f32(x) == int(np.argmax(x))


def test_matmul_transpose():
    a = np.arange(6 * 9, dtype=np.float32).reshape(6, 9) / 10.0
    b = np.arange(9 * 5, dtype=np.float32).reshape(9, 5) / 20.0

    assert_close(rc.matmul_f32(a, b), np.matmul(a, b), rtol=1e-4, atol=1e-4)
    assert_close(rc.transpose_f32(a), np.ascontiguousarray(a.T))


def test_cumsum_bincount_sort():
    x = (np.arange(2048, dtype=np.float32) % 7) / 7.0
    u8 = (np.arange(4096, dtype=np.uint16) % 256).astype(np.uint8)
    unsorted = np.array([3.0, -1.0, 2.0, 2.0, 0.0, 4.5], dtype=np.float32)

    assert_close(rc.cumsum_f32(x), np.cumsum(x, dtype=np.float32), rtol=1e-5, atol=1e-4)
    assert_close(rc.cumprod_f32(x[:32] + 1.0), np.cumprod(x[:32] + 1.0, dtype=np.float32), rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(rc.bincount_u8_256(u8), np.bincount(u8, minlength=256).astype(np.uint32))
    assert_close(rc.sort_f32(unsorted), np.sort(unsorted))


def test_correlate2d_5x5():
    image = np.arange(11 * 13, dtype=np.float32).reshape(11, 13) / 100.0
    w = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    filt = np.outer(w, w).astype(np.float32) / np.float32(256.0)

    expected = np.zeros_like(image)
    padded = np.pad(image, 2, mode="constant")
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            expected[y, x] = np.sum(padded[y:y + 5, x:x + 5] * filt, dtype=np.float32)

    assert_close(rc.correlate2d_5x5_f32(image, filt), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_add_exp_sum()
    test_reductions()
    test_matmul_transpose()
    test_cumsum_bincount_sort()
    test_correlate2d_5x5()
    print("numeric CUDA compatibility tests passed")

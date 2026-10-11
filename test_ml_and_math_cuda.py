import numpy as np
import numeric_cuda as nc

def assert_allclose(actual, desired, rtol=1e-4, atol=1e-4, msg=""):
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=msg)

def test_math_elementwise():
    x = np.array([0.1, 0.5, 1.2, 2.5, 5.0], dtype=np.float32)
    y = np.array([0.2, 0.4, 1.0, 2.0, 3.0], dtype=np.float32)

    # Basic arithmetic
    assert_allclose(nc.add(x, y), x + y, msg="add failed")
    assert_allclose(nc.subtract(x, y), x - y, msg="subtract failed")
    assert_allclose(nc.multiply(x, y), x * y, msg="multiply failed")
    assert_allclose(nc.divide(x, y), x / y, msg="divide failed")
    assert_allclose(nc.maximum(x, y), np.maximum(x, y), msg="maximum failed")
    assert_allclose(nc.minimum(x, y), np.minimum(x, y), msg="minimum failed")
    assert_allclose(nc.power(x, y), x ** y, msg="power failed")

    # Comparisons
    np.testing.assert_array_equal(nc.greater(x, y), x > y)
    np.testing.assert_array_equal(nc.less(x, y), x < y)
    np.testing.assert_array_equal(nc.equal(x, x), x == x)

    # Logic & where
    b1 = np.array([True, False, True, False, True])
    b2 = np.array([False, False, True, True, False])
    np.testing.assert_array_equal(nc.logical_and(b1, b2), np.logical_and(b1, b2))
    np.testing.assert_array_equal(nc.logical_or(b1, b2), np.logical_or(b1, b2))
    np.testing.assert_array_equal(nc.logical_not(b1), np.logical_not(b1))
    assert_allclose(nc.where(b1, x, y), np.where(b1, x, y), msg="where failed")

    # Functions
    assert_allclose(nc.exp(x), np.exp(x), msg="exp failed")
    assert_allclose(nc.log(x), np.log(x), msg="log failed")
    assert_allclose(nc.log1p(x), np.log1p(x), msg="log1p failed")
    assert_allclose(nc.expm1(x), np.expm1(x), msg="expm1 failed")
    assert_allclose(nc.sqrt(x), np.sqrt(x), msg="sqrt failed")
    assert_allclose(nc.sin(x), np.sin(x), msg="sin failed")
    assert_allclose(nc.cos(x), np.cos(x), msg="cos failed")
    assert_allclose(nc.tan(x), np.tan(x), msg="tan failed")
    assert_allclose(nc.tanh(x), np.tanh(x), msg="tanh failed")
    assert_allclose(nc.floor(x), np.floor(x), msg="floor failed")
    assert_allclose(nc.ceil(x), np.ceil(x), msg="ceil failed")
    assert_allclose(nc.abs(-x), x, msg="abs failed")
    assert_allclose(nc.negative(x), -x, msg="negative failed")
    assert_allclose(nc.square(x), x * x, msg="square failed")
    assert_allclose(nc.sign(np.array([-2.0, 0.0, 3.0], dtype=np.float32)),
                    np.sign(np.array([-2.0, 0.0, 3.0], dtype=np.float32)), msg="sign failed")
    assert_allclose(nc.clip(x, 0.5, 2.0), np.clip(x, 0.5, 2.0), msg="clip failed")

    print("[PASS] Math elementwise tests")

def test_ml_activations_and_losses():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)

    # Sigmoid
    expected_sig = 1.0 / (1.0 + np.exp(-x))
    assert_allclose(nc.sigmoid(x), expected_sig, msg="sigmoid failed")

    # ReLU & LeakyReLU
    assert_allclose(nc.relu(x), np.maximum(0.0, x), msg="relu failed")
    assert_allclose(nc.leaky_relu(x, alpha=0.1), np.where(x > 0, x, 0.1 * x), msg="leaky_relu failed")

    # GELU & SiLU
    expected_silu = x / (1.0 + np.exp(-x))
    assert_allclose(nc.silu(x), expected_silu, msg="silu failed")

    # 2D Softmax & Log-Softmax
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, -0.5]], dtype=np.float32)
    exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    expected_softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    expected_log_softmax = np.log(expected_softmax)
    expected_lse = np.log(np.sum(exp_shifted, axis=1)) + np.max(logits, axis=1)

    assert_allclose(nc.softmax(logits), expected_softmax, msg="softmax failed")
    assert_allclose(nc.log_softmax(logits), expected_log_softmax, msg="log_softmax failed")
    assert_allclose(nc.logsumexp(logits), expected_lse, msg="logsumexp failed")

    # Losses
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    expected_bce = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    assert_allclose(nc.binary_cross_entropy(y_pred, y_true), expected_bce, rtol=1e-3, msg="BCE failed")

    expected_mse = np.mean((y_pred - y_true) ** 2)
    assert_allclose(nc.mse_loss(y_pred, y_true), expected_mse, msg="MSE failed")

    print("[PASS] ML activations & loss tests")

def test_linear_algebra_and_metrics():
    # Dot & Norms
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float32)
    assert_allclose(nc.dot(a, b), float(np.dot(a, b)), msg="dot failed")
    assert_allclose(nc.norm(a, ord=2), float(np.linalg.norm(a)), msg="norm L2 failed")
    assert_allclose(nc.norm(a, ord=1), float(np.sum(np.abs(a))), msg="norm L1 failed")
    assert_allclose(nc.norm(a, ord=0), float(np.max(np.abs(a))), msg="norm Linf failed")

    # Outer product
    assert_allclose(nc.outer(a, b), np.outer(a, b), msg="outer failed")

    # Matvec & Matmul
    A = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
    assert_allclose(nc.matvec(A, b), A @ b, msg="matvec failed")

    B = np.array([[1.0, -1.0],
                  [2.0, 0.5],
                  [-0.5, 1.0],
                  [0.0, 2.0]], dtype=np.float32)
    assert_allclose(nc.matmul(A, B), A @ B, msg="matmul failed")
    assert_allclose(nc.transpose(A), A.T, msg="transpose failed")

    # Trace
    S = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    assert_allclose(nc.trace(S), float(np.trace(S)), msg="trace failed")

    # Covariance & Correlation
    X = np.array([[1.0, 2.0, 5.0],
                  [2.0, 3.0, 4.0],
                  [3.0, 5.0, 2.0],
                  [4.0, 7.0, 1.0],
                  [5.0, 11.0, 0.5]], dtype=np.float32)
    expected_cov = np.cov(X, rowvar=False)
    expected_corr = np.corrcoef(X, rowvar=False)
    assert_allclose(nc.cov(X), expected_cov, rtol=1e-3, msg="cov failed")
    assert_allclose(nc.corrcoef(X), expected_corr, rtol=1e-3, msg="corrcoef failed")

    # Solve symmetric system
    w_true = np.array([1.5, -2.0], dtype=np.float32)
    rhs = S @ w_true
    w_solved = nc.solve_symmetric(S, rhs)
    assert_allclose(w_solved, w_true, msg="solve_symmetric failed")

    # Spatial cdist
    XA = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    XB = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d_sq = np.sum((XA[:, None, :] - XB[None, :, :]) ** 2, axis=-1)
    assert_allclose(nc.cdist(XA, XB, "sqeuclidean"), d_sq, msg="cdist sqeuclidean failed")
    assert_allclose(nc.cdist(XA, XB, "euclidean"), np.sqrt(d_sq), msg="cdist euclidean failed")
    d_l1 = np.sum(np.abs(XA[:, None, :] - XB[None, :, :]), axis=-1)
    assert_allclose(nc.cdist(XA, XB, "cityblock"), d_l1, msg="cdist cityblock failed")

    print("[PASS] Linear algebra & metric tests")

def test_scans_diff_search():
    x = np.array([1.0, 3.0, 6.0, 10.0, 15.0], dtype=np.float32)
    assert_allclose(nc.cumsum(x), np.cumsum(x), msg="cumsum failed")
    assert_allclose(nc.cumprod(x), np.cumprod(x), msg="cumprod failed")
    assert_allclose(nc.diff(x), np.diff(x), msg="diff failed")

    # Trapezoid integration
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    assert_allclose(nc.trapezoid(y, dx=0.5), np.trapezoid(y, dx=0.5), msg="trapezoid failed")

    # Searchsorted
    sorted_arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    queries = np.array([0.5, 3.0, 4.5, 7.0, 10.0], dtype=np.float32)
    np.testing.assert_array_equal(nc.searchsorted(sorted_arr, queries, "left"),
                                  np.searchsorted(sorted_arr, queries, "left"))
    np.testing.assert_array_equal(nc.searchsorted(sorted_arr, queries, "right"),
                                  np.searchsorted(sorted_arr, queries, "right"))

    # Reductions
    assert_allclose(nc.count_nonzero(np.array([0.0, 1.0, 0.0, 3.0], dtype=np.float32)), 2)
    assert nc.all(np.array([True, True, True])) == True
    assert nc.all(np.array([True, False, True])) == False
    assert nc.any(np.array([False, True, False])) == True
    assert nc.any(np.array([False, False, False])) == False

    print("[PASS] Scans, diff & search tests")

def test_machine_learning_models():
    np.random.seed(42)

    # 1. KMeans
    # Create two clear clusters
    c1 = np.random.randn(50, 3).astype(np.float32) + np.array([5.0, 5.0, 5.0], dtype=np.float32)
    c2 = np.random.randn(50, 3).astype(np.float32) - np.array([5.0, 5.0, 5.0], dtype=np.float32)
    X_km = np.vstack([c1, c2])

    kmeans = nc.KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_km)
    assert len(labels) == 100
    assert kmeans.cluster_centers_.shape == (2, 3)
    assert kmeans.inertia_ > 0.0
    # First 50 should have one label, last 50 another label
    assert len(np.unique(labels[:50])) == 1
    assert len(np.unique(labels[50:])) == 1
    assert labels[0] != labels[50]
    print("[PASS] KMeans clustering")

    # 2. PCA
    # 3D points primarily varying along 1st and 2nd axis
    X_pca = np.random.randn(100, 3).astype(np.float32)
    X_pca[:, 0] *= 10.0
    X_pca[:, 1] *= 5.0
    pca = nc.PCA(n_components=2)
    X_trans = pca.fit_transform(X_pca)
    assert X_trans.shape == (100, 2)
    assert pca.components_.shape == (2, 3)
    assert pca.explained_variance_[0] > pca.explained_variance_[1]
    print("[PASS] PCA dimensionality reduction")

    # 3. Linear Regression
    X_lr = np.random.randn(100, 4).astype(np.float32)
    w_true = np.array([2.5, -1.2, 0.8, -3.0], dtype=np.float32)
    y_lr = X_lr @ w_true + 1.5 + 0.01 * np.random.randn(100).astype(np.float32)

    lr = nc.LinearRegression(alpha=0.01, fit_intercept=True)
    lr.fit(X_lr, y_lr)
    assert_allclose(lr.coef_, w_true, rtol=0.05, atol=0.05, msg="Linear regression coef failed")
    assert_allclose(lr.intercept_, 1.5, rtol=0.05, atol=0.05, msg="Linear regression intercept failed")
    y_pred_lr = lr.predict(X_lr)
    assert np.mean((y_pred_lr - y_lr) ** 2) < 0.05
    print("[PASS] Linear Regression")

    # 4. Logistic Regression
    X_log = np.random.randn(200, 2).astype(np.float32)
    # Class 1 if x0 + x1 > 0 else 0
    y_log = ((X_log[:, 0] + X_log[:, 1]) > 0).astype(np.float32)
    clf = nc.LogisticRegression(lr=0.5, max_iter=300, reg=0.001)
    clf.fit(X_log, y_log)
    preds = clf.predict(X_log)
    acc = np.mean(preds == y_log)
    assert acc > 0.90, f"Logistic regression accuracy too low: {acc}"
    print(f"[PASS] Logistic Regression (accuracy: {acc * 100:.1f}%)")

    # 5. K-Nearest Neighbors Classifier & Regressor
    knn_c = nc.KNeighborsClassifier(n_neighbors=3)
    knn_c.fit(X_log, y_log)
    knn_preds = knn_c.predict(X_log)
    knn_acc = np.mean(knn_preds == y_log)
    assert knn_acc > 0.95, f"KNN accuracy too low: {knn_acc}"
    print(f"[PASS] KNeighborsClassifier (accuracy: {knn_acc * 100:.1f}%)")

    knn_r = nc.KNeighborsRegressor(n_neighbors=5)
    knn_r.fit(X_lr[:80], y_lr[:80])
    y_knn_r = knn_r.predict(X_lr[80:])
    assert np.mean((y_knn_r - y_lr[80:]) ** 2) < 5.0
    print("[PASS] KNeighborsRegressor")

    # 6. Gaussian Naive Bayes
    gnb = nc.GaussianNB()
    gnb.fit(X_km, labels)
    gnb_preds = gnb.predict(X_km)
    gnb_acc = np.mean(gnb_preds == labels)
    assert gnb_acc == 1.0, f"GaussianNB accuracy: {gnb_acc}"
    print(f"[PASS] GaussianNB (accuracy: {gnb_acc * 100:.1f}%)")

if __name__ == "__main__":
    test_math_elementwise()
    test_ml_activations_and_losses()
    test_linear_algebra_and_metrics()
    test_scans_diff_search()
    test_machine_learning_models()
    print("\nALL NUMERIC & ML CUDA TESTS PASSED PERFECTLY!")

"""Numerical gradient checks: compare each component's analytic backward
pass against central-difference finite differences, and do the same for
the full MLP end to end.

This is the pytest version of the gradient-check script the author used
during development. There is no generic "autodiff engine" to enumerate
ops for (see the README); the four hand-derived components below --
``Layer``, ``ReLU``, the fused ``Softmax``+``CrossEntropyLoss`` gradient,
and the full ``MLP`` -- are exactly what is checked.

Every test reports its max relative error via a module-level list so the
final test can print and assert on the overall maximum across all of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mnist_scratch.layers import CrossEntropyLoss, Layer, ReLU, Softmax
from mnist_scratch.model import MLP

RTOL_THRESHOLD = 1e-5
EPS = 1e-6

_max_relative_errors: dict[str, float] = {}


def _relative_error(analytic: np.ndarray, numerical: np.ndarray) -> np.ndarray:
    """Elementwise relative error, safe for entries where both are ~0."""
    denom = np.maximum(np.abs(analytic), np.abs(numerical))
    denom = np.where(denom < 1e-8, 1.0, denom)
    return np.abs(analytic - numerical) / denom


def _numerical_gradient(f, x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Central-difference gradient of scalar function f w.r.t. array x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f()
        x[idx] = original - eps
        minus = f()
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_layer_gradients():
    """Layer.backward's dX, dW, db against finite differences of
    sum(dA * layer.forward(X)) -- the standard trick for checking a
    backward pass against an arbitrary upstream gradient."""
    n_in, n_out, batch = 5, 3, 4
    layer = Layer(n_in, n_out)
    X = np.random.randn(batch, n_in)
    dA = np.random.randn(batch, n_out)

    out = layer.forward(X)
    dX_analytic = layer.backward(dA)
    dW_analytic = layer.dW
    db_analytic = layer.db

    def loss_from_X():
        return np.sum(dA * layer.forward(X))

    def loss_from_W():
        return np.sum(dA * ((X @ layer.W) + layer.b))

    def loss_from_b():
        return np.sum(dA * ((X @ layer.W) + layer.b))

    dX_numeric = _numerical_gradient(loss_from_X, X)
    dW_numeric = _numerical_gradient(loss_from_W, layer.W)
    db_numeric = _numerical_gradient(loss_from_b, layer.b)

    err_X = np.max(_relative_error(dX_analytic, dX_numeric))
    err_W = np.max(_relative_error(dW_analytic, dW_numeric))
    err_b = np.max(_relative_error(db_analytic, db_numeric))
    max_err = max(err_X, err_W, err_b)
    _max_relative_errors["Layer (dX, dW, db)"] = max_err

    assert max_err < RTOL_THRESHOLD, f"Layer gradient check failed: {max_err}"


def test_relu_gradient():
    """ReLU.backward's dX against finite differences of sum(dA * relu(X))."""
    batch, n = 4, 6
    relu = ReLU()
    X = np.random.randn(batch, n)
    dA = np.random.randn(batch, n)

    relu.forward(X)
    dX_analytic = relu.backward(dA)

    def loss_from_X():
        return np.sum(dA * np.maximum(0, X))

    dX_numeric = _numerical_gradient(loss_from_X, X)
    max_err = np.max(_relative_error(dX_analytic, dX_numeric))
    _max_relative_errors["ReLU (dX)"] = max_err

    assert max_err < RTOL_THRESHOLD, f"ReLU gradient check failed: {max_err}"


def test_softmax_crossentropy_fused_gradient():
    """The documented shortcut in layers.py: CrossEntropyLoss.backward()
    is claimed to equal d(loss)/d(pre-softmax logits), not d(loss)/d(probs).
    This checks that claim directly against finite differences taken on the
    logits through the full Softmax -> CrossEntropyLoss composition."""
    batch, n_classes = 4, 5
    logits = np.random.randn(batch, n_classes)
    labels = np.random.randint(0, n_classes, size=batch)
    y_true = np.zeros((batch, n_classes))
    y_true[np.arange(batch), labels] = 1.0

    softmax = Softmax()
    ce = CrossEntropyLoss()

    probs = softmax.forward(logits)
    ce.forward(probs, y_true)
    dlogits_analytic = ce.backward()

    def loss_from_logits():
        p = Softmax().forward(logits)
        return CrossEntropyLoss().forward(p, y_true)

    dlogits_numeric = _numerical_gradient(loss_from_logits, logits)
    max_err = np.max(_relative_error(dlogits_analytic, dlogits_numeric))
    _max_relative_errors["Softmax+CrossEntropy fused (d/d logits)"] = max_err

    assert max_err < RTOL_THRESHOLD, f"Fused softmax/CE gradient check failed: {max_err}"


def test_full_mlp_gradient():
    """Every Layer's dW/db inside a real MLP, checked against finite
    differences of MLP.compute_loss w.r.t. each weight and bias matrix."""
    batch = 6
    mlp = MLP([10, 7, 4])
    X = np.random.randn(batch, 10)
    labels = np.random.randint(0, 4, size=batch)
    y_true = np.zeros((batch, 4))
    y_true[np.arange(batch), labels] = 1.0

    mlp.compute_loss(X, y_true)
    mlp.backward()

    max_err = 0.0
    for layer in mlp.layers:
        if not isinstance(layer, Layer):
            continue

        def loss_fn():
            return mlp.compute_loss(X, y_true)

        dW_numeric = _numerical_gradient(loss_fn, layer.W)
        db_numeric = _numerical_gradient(loss_fn, layer.b)
        err_W = np.max(_relative_error(layer.dW, dW_numeric))
        err_b = np.max(_relative_error(layer.db, db_numeric))
        max_err = max(max_err, err_W, err_b)

    _max_relative_errors["Full MLP (all layers' dW, db)"] = max_err
    assert max_err < RTOL_THRESHOLD, f"Full MLP gradient check failed: {max_err}"


def test_zzz_report_max_relative_error():
    """Not a real assertion -- runs last (by name) purely to print the
    overall maximum relative error across every gradient check above."""
    assert _max_relative_errors, "no gradient checks ran"
    overall_max = max(_max_relative_errors.values())
    print("\nGradient check results (max relative error vs. finite differences):")
    for name, err in _max_relative_errors.items():
        print(f"  {name}: {err:.3e}")
    print(f"  OVERALL MAX: {overall_max:.3e}")
    assert overall_max < RTOL_THRESHOLD

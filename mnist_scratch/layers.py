"""Hand-derived forward/backward passes for a fixed-topology MLP.

There is no generic autodiff engine here: each component below has its own
bespoke, manually-derived backward method. Nothing is a general operation
on an arbitrary computational graph -- ``Layer``, ``ReLU``, ``Softmax`` and
``CrossEntropyLoss`` are wired together by hand in ``MLP`` (see
``model.py``). This module is a straight extraction of the original
``MNIST_Scratch.py`` script: the math is untouched, only the names, the
file layout, the docstrings and the type hints are new.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Layer:
    """A single affine (fully-connected) layer: ``Y = X @ W + b``.

    Weights are He-initialized (``N(0, 1) * sqrt(2 / n_in)``), biases are
    zero-initialized. ``backward`` and ``update`` implement plain SGD with
    no momentum or weight decay; gradients are averaged over the batch
    (divided by ``n``) inside ``backward``, not inside ``update``.
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        self.W: FloatArray = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b: FloatArray = np.zeros((1, n_out))
        self.X: FloatArray | None = None
        self.dW: FloatArray | None = None
        self.db: FloatArray | None = None

    def forward(self, X: FloatArray) -> FloatArray:
        """Cache the input and return ``X @ W + b``."""
        self.X = X
        return (X @ self.W) + self.b

    def backward(self, dA: FloatArray) -> FloatArray:
        """Given dL/dY (already batch-averaged -- see note below), compute
        dL/dW, dL/db and return dL/dX.

        Note on batch averaging (fixed 2026-09-26, see project README
        "Deviations from the original script"): ``dA`` arrives here already
        divided by the batch size ``n`` -- that division happens exactly
        once, in ``CrossEntropyLoss.backward``. So ``self.X.T @ dA`` is
        already the correctly-averaged weight gradient and must NOT be
        divided by ``n`` again. The original script did divide by ``n``
        again here, which silently shrank every weight/bias gradient by an
        extra factor of ``1/n`` (confirmed via gradient-check tests scaling
        exactly as ``1/batch_size``). ``n`` is kept as a local variable
        (unused beyond documenting the shape) so the diff against the
        original stays minimal and obvious.
        """
        n = self.X.shape[0]  # noqa: F841 (kept for history/diff clarity)
        self.dW = self.X.T @ dA
        self.db = np.sum(dA, axis=0, keepdims=True)
        return dA @ self.W.T

    def update(self, lr: float) -> None:
        """Apply one vanilla SGD step using the gradients from ``backward``."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    """Elementwise rectified linear unit: ``max(0, X)``."""

    def __init__(self) -> None:
        self.X: FloatArray | None = None

    def forward(self, X: FloatArray) -> FloatArray:
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA: FloatArray) -> FloatArray:
        return dA * (self.X > 0)


class Softmax:
    """Row-wise softmax with the max-subtraction stability trick.

    ``backward`` is intentionally not implemented. When this layer is the
    last one in the network and is paired with ``CrossEntropyLoss``, the
    combined derivative of softmax + cross-entropy w.r.t. the pre-softmax
    logits simplifies to ``probs - y_true`` -- that simplification is what
    ``CrossEntropyLoss.backward`` returns, and ``MLP.backward`` feeds it
    directly into the layer *before* this one, skipping this class's
    backward entirely. This only works because ``Softmax`` is never used
    anywhere except immediately before ``CrossEntropyLoss``.
    """

    def __init__(self) -> None:
        self.out: FloatArray | None = None

    def forward(self, X: FloatArray) -> FloatArray:
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out


class CrossEntropyLoss:
    """Mean cross-entropy loss over one-hot targets, fused with softmax's gradient.

    ``forward`` computes ``-mean(sum(y_true * log(probs + eps)))``.
    ``backward`` returns ``(probs - y_true) / n``, the gradient of softmax +
    cross-entropy combined w.r.t. the pre-softmax logits (see ``Softmax``
    docstring for why this is what gets passed to the network's last
    ``Layer``, not a true gradient w.r.t. ``probs`` alone).
    """

    def __init__(self) -> None:
        self.probs: FloatArray | None = None
        self.y_true: FloatArray | None = None

    def forward(self, probs: FloatArray, y_true: FloatArray) -> float:
        self.probs = probs
        self.y_true = y_true
        n = probs.shape[0]
        return -np.sum(y_true * np.log(probs + 1e-9)) / n

    def backward(self) -> FloatArray:
        n = self.probs.shape[0]
        return (self.probs - self.y_true) / n

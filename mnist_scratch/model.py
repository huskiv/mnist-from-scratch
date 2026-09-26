"""The MLP: a fixed list of ``Layer``/``ReLU``/``Softmax`` objects chained
by hand. Extracted unchanged from the original ``MNIST_Scratch.py``
(training-loop structure, hyperparameter defaults, and print cadence are
all identical); only names, module layout, docstrings and type hints
are new.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .layers import CrossEntropyLoss, Layer, ReLU, Softmax

FloatArray = NDArray[np.float64]


class MLP:
    """A multi-layer perceptron built from ``layer_sizes``.

    For ``layer_sizes = [n0, n1, ..., nk]``, this builds ``k`` ``Layer``
    objects (``n0->n1``, ``n1->n2``, ...), each followed by ``ReLU`` except
    the last, which is followed by ``Softmax``. E.g. ``[784, 128, 10]``
    gives ``Layer(784,128) -> ReLU -> Layer(128,10) -> Softmax``.
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        n_linear = len(layer_sizes) - 1
        self.layers: list[Layer | ReLU | Softmax] = []
        self.loss_fn = CrossEntropyLoss()

        for i in range(n_linear):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
            is_last = i == n_linear - 1
            self.layers.append(Softmax() if is_last else ReLU())

    def forward(self, X: FloatArray) -> FloatArray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, X: FloatArray, y_true: FloatArray) -> float:
        probs = self.forward(X)
        return self.loss_fn.forward(probs, y_true)

    def backward(self) -> FloatArray:
        """Backpropagate from the loss through every layer except the final
        ``Softmax`` (see ``Softmax``'s docstring for why it's skipped)."""
        dA = self.loss_fn.backward()
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA)
        return dA

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.update(lr)

    def evaluate(self, X: FloatArray, y: FloatArray) -> float:
        """Classification accuracy: fraction of rows where argmax matches."""
        preds = self.forward(X)
        return float(np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1)))

    def train(
        self,
        X_train: FloatArray,
        y_train: FloatArray,
        X_test: FloatArray,
        y_test: FloatArray,
        epochs: int,
        batch_size: int,
        lr: float = 0.1,
    ) -> None:
        """Mini-batch SGD. Reshuffles the training set every epoch using a
        ``default_rng(42)`` instance created once, before the epoch loop, so
        each epoch draws its own (still seed-42-derived) permutation."""
        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            losses = []
            perm = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for start in range(0, X_train.shape[0], batch_size):
                batch_loss = []
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                batch_loss.append(self.compute_loss(X_batch, y_batch))

                self.backward()

                self.update(lr)

                losses.append(np.mean(batch_loss))

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs}")
                print(np.mean(losses))
                print(f"Train eval result: {self.evaluate(X_shuffled, y_shuffled)}")
                print(f"Test eval result: {self.evaluate(X_test, y_test)}")

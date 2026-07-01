import numpy as np
import matplotlib.pyplot as plt
import sklearn

sklearn.datasets.fetch_openml('mnist_784')

class MNIST:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def load_data(self):
        with np.load(self.data_path) as data:
            self.train_images = data['x_train']
            self.train_labels = data['y_train']
            self.test_images = data['x_test']
            self.test_labels = data['y_test']

    def preprocess_data(self):
        self.train_images = self.train_images.astype(np.float32) / 255.0
        self.test_images = self.test_images.astype(np.float32) / 255.0

    class ReLU:
        def forward(self, X):
            self.X = X
            return np.maximum(0, X)

        def backward(self, dA):
            return dA * (self.X > 0)

    class Softmax:
        def forward(self, X):
            shifted = X - np.max(X, axis=1, keepdims=True)
            exp = np.exp(shifted)
            self.out = exp / np.sum(exp, axis=1, keepdims=True)
            return self.out

        def backward(self, dA):
            dX = np.empty_like(dA)
            for i, (s, d) in enumerate(zip(self.out, dA)):
                s = s.reshape(-1, 1)
                jacobian = np.diagflat(s) - s @ s.T
                dX[i] = jacobian @ d
            return dX
        
    class CrossEntropyLoss:
        def forward(self, probs, y_true):
            self.probs = probs
            self.y_true = y_true
            n = probs.shape[0]
            return -np.sum(y_true * np.log(probs + 1e-9)) / n

        def backward(self):
            n = self.probs.shape[0]
            return (self.probs - self.y_true) / n

    class Layer:
        def __init__(self, n_in, n_out):
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.b = np.zeros((1, n_out))

        def forward(self, X):
            self.X = X
            return (X @ self.W) + self.b

        def backward(self, dA):
            n = self.X.shape[0]
            self.dW = self.X.T @ dA / n
            self.db = np.sum(dA, axis=0, keepdims=True) / n
            return dA @ self.W.T

        def update(self, lr):
            self.W -= lr * self.dW
            self.b -= lr * self.db

    class MLP:
        def __init__(self, layer_sizes):
            n_linear = len(layer_sizes) - 1
            self.layers = []
            self.loss_fn = MNIST.CrossEntropyLoss()
            
            for i in range(n_linear):
                self.layers.append(MNIST.Layer(layer_sizes[i], layer_sizes[i + 1]))
                is_last = i == n_linear - 1
                self.layers.append(MNIST.Softmax() if is_last else MNIST.ReLU())
            

        def forward(self, X):
            for layer in self.layers:
                X = layer.forward(X)
            return X

        def compute_loss(self, X, y_true):
            probs = self.forward(X)
            return self.loss_fn.forward(probs, y_true)

        def backward(self):
            dA = self.loss_fn.backward()
            # last layer is Softmax; its gradient is fused into CrossEntropyLoss.backward,
            # so skip straight to the layer before it
            for layer in reversed(self.layers[:-1]):
                dA = layer.backward(dA)
            return dA

        def update(self, lr):
            for layer in self.layers:
                if isinstance(layer, MNIST.Layer):
                    layer.update(lr)

model = MNIST()


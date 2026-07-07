import numpy as np
from sklearn.datasets import fetch_openml

class MNIST:
    N_TRAIN = 60000
    N_CLASSES = 10

    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def load_data(self):
        X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
        y = y.astype(np.int64)
        
        rng = np.random.default_rng(42)
        perm = rng.permutation(X.shape[0])
        X, y = X[perm], y[perm]

        self.train_images = X[:self.N_TRAIN]
        self.test_images = X[self.N_TRAIN:]
        self.train_labels = self._one_hot(y[:self.N_TRAIN])
        self.test_labels = self._one_hot(y[self.N_TRAIN:])

    def _one_hot(self, labels):
        one_hot = np.zeros((labels.shape[0], self.N_CLASSES), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        return one_hot

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

        # def backward(self, dA):
        #     dX = np.empty_like(dA)
        #     for i, (s, d) in enumerate(zip(self.out, dA)):
        #         s = s.reshape(-1, 1)
        #         jacobian = np.diagflat(s) - s @ s.T
        #         dX[i] = jacobian @ d
        #     return dX
        
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
            for layer in reversed(self.layers[:-1]):
                dA = layer.backward(dA)
            return dA

        def update(self, lr):
            for layer in self.layers:
                if isinstance(layer, MNIST.Layer):
                    layer.update(lr)

        def evaluate(self, X, y):
            preds = self.forward(X)
            return np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

        def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, lr=0.1):
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

model = MNIST()
model.load_data()
model.preprocess_data()

mlp = MNIST.MLP([784, 128, 10])
mlp.train(model.train_images, model.train_labels, model.test_images, model.test_labels, 100, 128, 100)
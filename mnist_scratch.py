import numpy as np
import matplotlib.pyplot as plt

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

    class Layer:
        def __init__(self, n_in, n_out):
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.b = np.zeros((1, n_out))

        def forward(self, X):
            self.X = X
            return (X @ self.W) + self.b

    class MLP:
        def __init__(self, layer_sizes):
            self.layers = [
            MNIST.Layer(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ]
            
        def forward(self, X):
            for layer in self.layers:
                X = layer.forward(X)
            return X
        
model = MNIST()


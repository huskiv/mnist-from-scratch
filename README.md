# mnist-scratch

A multi-layer perceptron trained on MNIST, with backpropagation derived and
implemented by hand in NumPy: no PyTorch, no TensorFlow, no autograd
library. I built this to understand backprop at the level of individual
matrix derivatives, not to build a reusable framework. There is no
generic computational-graph engine here, and this README does not claim
one. What exists is four fixed components (an affine layer, ReLU, softmax,
cross-entropy loss) each with a manually-derived `backward` method, wired
together for one specific network shape. The original file was written entirely by hand. The tests and most of this readme are AI-generated.

## How it works

The network is a fixed pipeline, not a dynamic graph:

```
X --> Linear(784,128) --> ReLU --> Linear(128,10) --> Softmax --> CrossEntropyLoss
```

Forward pass: each stage's `forward` is called in order, caching whatever
it needs for its own backward pass (`Layer` caches its input, `ReLU`
caches its input, `Softmax` caches its output).

Backward pass: `MLP.backward` starts from `CrossEntropyLoss.backward()`
and walks the pipeline in reverse, calling each stage's own `backward`
with the gradient the next stage handed it:

```
dLoss/dLogits = CrossEntropyLoss.backward()      # see note below
dLoss/dHidden = Linear(128,10).backward(dLogits)
dLoss/dReLU_in = ReLU.backward(dLoss/dHidden)
dLoss/dW1, dLoss/db1 = Linear(784,128).backward(dLoss/dReLU_in)
```

**The softmax + cross-entropy shortcut.** `Softmax.backward` is not
implemented. When softmax is followed directly by cross-entropy loss (the
only way it's used here), the derivative of the combined loss with
respect to the *pre-softmax logits* simplifies algebraically to
`probs - y_true`. `CrossEntropyLoss.backward()` returns exactly that, and
`MLP.backward` feeds it straight into the layer before softmax, skipping
a separate softmax backward pass entirely. `tests/test_gradients.py`
checks this simplification directly against finite differences taken
through the full `Softmax -> CrossEntropyLoss` composition, rather than
just trusting the algebra.

Each `Layer` computes its own gradient by chain rule:
`dL/dW = X.T @ dA`, `dL/db = sum(dA, axis=0)`, `dL/dX = dA @ W.T`, where
`dA` is the gradient handed down from the layer after it.

## Deviations from the original script

This project began as a single file (`MNIST_Scratch.py`) written as a learning project over the summer, extracted here with full git history via
`git filter-repo`. Everything below is a deliberate, disclosed change from
that original. The math inside `Layer`, `ReLU`, `Softmax`,
`CrossEntropyLoss`, and `MLP` is otherwise untouched.

## Results

Ran end to end with `python train.py` (no hyperparameter tuning beyond
the lr correction above):

| | |
|---|---|
| Architecture | 784 -> 128 (ReLU) -> 10 (Softmax) |
| Epochs | 100 |
| Batch size | 128 |
| Learning rate | 0.1 |
| Final train accuracy | 99.99% |
| **Final test accuracy** | **97.95%** |
| Wall-clock training time | 69.2s |
| Hardware | Intel Xeon @ 2.10GHz, 2 vCPUs (cloud sandbox, not a local machine) |

Gradient checks (`pytest -v`, finite differences vs. analytic gradients):

| Component | Max relative error |
|---|---|
| Layer (dX, dW, db) | 1.8e-08 |
| ReLU (dX) | 2.3e-09 |
| Softmax + CrossEntropy fused gradient | 1.8e-08 |
| Full MLP (all layers' dW, db) | 2.2e-07 |
| **Overall max** | **2.2e-07** |

Train accuracy hitting 99.99% while test sits at 97.95% is a normal
overfitting gap for an unregularized MLP on MNIST at 100 epochs -- there's
no dropout, weight decay, or early stopping here.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```
python train.py
```

First run downloads and MD5-verifies ~12MB of MNIST data into `data/raw/`
(gitignored). Subsequent runs reuse the cache. Flags: `--epochs`,
`--batch-size`, `--lr`, `--hidden`, `--cache-dir` (see `python train.py -h`).

## Test

```
python -m pytest tests/ -v
```

Runs finite-difference gradient checks for every component and for the
full network, and prints the maximum relative error found.

## Project layout

```
mnist_scratch/
    __init__.py
    layers.py     # Layer, ReLU, Softmax, CrossEntropyLoss
    model.py      # MLP: wires the above into one forward/backward pipeline
    data.py       # MNIST download, MD5 verification, caching, preprocessing
train.py          # single-command training entry point
tests/
    test_gradients.py  # finite-difference gradient checks
requirements.txt
LICENSE
```

"""Single-command training entry point.

    python train.py

Layer sizes [784, 128, 10], 100 epochs, batch size 128. Learning rate
defaults to 0.1, not the original script's 10 -- see the project README's
"Deviations from the original script" section: the original
``Layer.backward`` divided the weight/bias gradient by the batch size
twice, shrinking it by an extra 1/128, and its lr=10 was empirically tuned
around that bug. With the bug fixed (see ``mnist_scratch/layers.py``),
lr=10 would apply an update roughly 128x too large. lr=0.1 restores a
comparable effective step size and happens to match the original
``MLP.train`` method signature's own untouched default parameter value.

Reproducibility note: the original script seeded the two ``default_rng``
instances (data shuffle, per-epoch batch shuffle) but never seeded
``np.random.randn``, which ``Layer.__init__`` uses for weight
initialization -- so two runs of the original script produced different
final weights and accuracy. This entry point adds one line,
``np.random.seed(SEED)``, before the model is constructed, purely to make
that legacy call deterministic too. No line inside ``mnist_scratch/`` was
changed to do this. This is a behavioral change from the original script
and is called out in the README.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from mnist_scratch import MLP, MNISTData

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128, help="hidden layer width")
    parser.add_argument("--cache-dir", type=str, default="data/raw")
    args = parser.parse_args()

    np.random.seed(SEED)  # see module docstring: makes weight init reproducible

    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or platform.machine()}")

    data = MNISTData(cache_dir=args.cache_dir)
    data.load_data()
    data.preprocess_data()

    mlp = MLP([784, args.hidden, 10])

    start = time.perf_counter()
    mlp.train(
        data.train_images,
        data.train_labels,
        data.test_images,
        data.test_labels,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    elapsed = time.perf_counter() - start

    final_test_acc = mlp.evaluate(data.test_images, data.test_labels)
    final_train_acc = mlp.evaluate(data.train_images, data.train_labels)

    print("---")
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Wall-clock training time: {elapsed:.1f}s "
          f"({args.epochs} epochs, batch size {args.batch_size})")


if __name__ == "__main__":
    main()

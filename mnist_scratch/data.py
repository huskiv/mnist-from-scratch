"""MNIST loading, caching, and preprocessing.

The original script loaded MNIST via ``sklearn.datasets.fetch_openml``,
which downloads from openml.org and caches under the user's home
directory (not the repo). That data source is not reachable from every
network this project might be built on, and caching outside the repo
means "never commit MNIST" was only true by accident.

This module is a **rewrite of data loading only** (explicitly permitted --
see the project README's "Deviations from the original script" section).
It downloads the original four gzipped IDX files from a pinned mirror
commit, verifies them against the well-known official MD5 checksums, and
caches them under ``<repo>/data/raw/`` (gitignored). Everything
downstream of "raw arrays of images and labels" -- the seeded shuffle,
the 60000/10000 split, one-hot encoding, and the /255 normalization -- is
copied unchanged from the original ``MNIST.load_data`` /
``MNIST.preprocess_data`` methods.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Pinned to a specific commit (not `master`) so the URLs never change
# contents out from under this project.
_MIRROR_COMMIT = "ae608a02b796c9ee908c082585a5f006fe05258e"
_MIRROR_BASE = f"https://raw.githubusercontent.com/fgnt/mnist/{_MIRROR_COMMIT}"

# filename -> (url path, official MD5 from http://yann.lecun.com/exdb/mnist/)
_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

N_TRAIN = 60000
N_CLASSES = 10


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_and_verify(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in _FILES.items():
        dest = cache_dir / filename
        if dest.exists() and _md5(dest) == expected_md5:
            continue
        url = f"{_MIRROR_BASE}/{filename}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            dest.write_bytes(resp.read())
        actual_md5 = _md5(dest)
        if actual_md5 != expected_md5:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
            )


def _read_idx_images(path: Path) -> IntArray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.int64)


def _read_idx_labels(path: Path) -> IntArray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


class MNISTData:
    """Loads, shuffles, splits, and normalizes MNIST.

    Usage mirrors the original script exactly:

        data = MNISTData()
        data.load_data()
        data.preprocess_data()
        # data.train_images, data.train_labels, data.test_images, data.test_labels
    """

    N_TRAIN = N_TRAIN
    N_CLASSES = N_CLASSES

    def __init__(self, cache_dir: str | Path = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.train_images: FloatArray | None = None
        self.train_labels: FloatArray | None = None
        self.test_images: FloatArray | None = None
        self.test_labels: FloatArray | None = None

    def load_data(self) -> None:
        _download_and_verify(self.cache_dir)

        # Concatenate the official train + test IDX files into one 70000-row
        # pool, exactly as the original script pooled OpenML's 70000 rows
        # before reshuffling. The pre-shuffle order doesn't affect the
        # result: a full permutation over all 70000 indices erases any
        # dependence on the input order.
        train_X = _read_idx_images(self.cache_dir / "train-images-idx3-ubyte.gz")
        train_y = _read_idx_labels(self.cache_dir / "train-labels-idx1-ubyte.gz")
        test_X = _read_idx_images(self.cache_dir / "t10k-images-idx3-ubyte.gz")
        test_y = _read_idx_labels(self.cache_dir / "t10k-labels-idx1-ubyte.gz")

        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        y = y.astype(np.int64)

        rng = np.random.default_rng(42)
        perm = rng.permutation(X.shape[0])
        X, y = X[perm], y[perm]

        self.train_images = X[: self.N_TRAIN]
        self.test_images = X[self.N_TRAIN :]
        self.train_labels = self._one_hot(y[: self.N_TRAIN])
        self.test_labels = self._one_hot(y[self.N_TRAIN :])

    def _one_hot(self, labels: IntArray) -> FloatArray:
        one_hot = np.zeros((labels.shape[0], self.N_CLASSES), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        return one_hot

    def preprocess_data(self) -> None:
        self.train_images = self.train_images.astype(np.float32) / 255.0
        self.test_images = self.test_images.astype(np.float32) / 255.0

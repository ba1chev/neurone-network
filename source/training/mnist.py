import numpy as np
from typing import Tuple
from sklearn.datasets import fetch_openml


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    pixels = bunch.data.astype(np.float64) / 255.0
    labels = bunch.target.astype(np.int64)
    x_train, x_test = pixels[:60000], pixels[60000:]
    y_train, y_test = labels[:60000], labels[60000:]
    return x_train, y_train, x_test, y_test

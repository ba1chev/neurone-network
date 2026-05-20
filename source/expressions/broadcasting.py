import numpy as np


def unbroadcast(gradient: np.ndarray, target_shape: tuple) -> np.ndarray:
    # When two tensors are broadcast in the forward pass, the upstream gradient
    # has the shape of the broadcasted output. To send it back to an input
    # whose shape was smaller, sum over (a) the leading axes that were added
    # and (b) the axes that were size - 1 in the input but expanded in the output
    while gradient.ndim > len(target_shape):
        gradient = gradient.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)

    return gradient

#import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
# from jax import random  # type: ignore
from typing import Optional

# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------
class Relu:
    def __init__(self):
        self.mask: Optional[jnp.ndarray] = None

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.mask = (x > 0).astype(jnp.float64)
        return x * self.mask

    def backward(self, dout: jnp.ndarray) -> jnp.ndarray:
        assert self.mask is not None, "Forward must be called before Backward."
        return dout * self.mask


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------
class Sigmoid:
    def __init__(self):
        self.out: Optional[jnp.ndarray] = None

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.out = 1.0 / (1.0 + jnp.exp(-x))
        return self.out

    def backward(self, dout: jnp.ndarray) -> jnp.ndarray:
        assert self.out is not None, "Forward must be called before Backward."
        return dout * self.out * (1.0 - self.out)



# ---------------------------------------------------------------------------
# Affine (Fully Connected)
# ---------------------------------------------------------------------------
class Affine:
    """
    output = x @ W + B
    W: (in_features, out_features)
    B: (out_features,)
    """

    def __init__(self, W: jnp.ndarray, B: jnp.ndarray):
        self.W = W
        self.B = B
        self.x: Optional[jnp.ndarray] = None
        self.dW: Optional[jnp.ndarray] = None
        self.dB: Optional[jnp.ndarray] = None

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.x = x
        return x @ self.W + self.B  # broadcasting adds B to every row

    def backward(self, dout: jnp.ndarray) -> jnp.ndarray:
        assert self.x is not None, "Forward must be called before Backward."
        dx = dout @ self.W.T          # (batch, in_features)
        self.dW = self.x.T @ dout     # (in_features, out_features)
        self.dB = dout.sum(axis=0)    # (out_features,)
        return dx


# ---------------------------------------------------------------------------
# SoftmaxWithLoss
# ---------------------------------------------------------------------------
def _softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Row-wise numerically stable softmax."""
    x_shifted = x - x.max(axis=1, keepdims=True)
    exp_x = jnp.exp(x_shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def _cross_entropy_error(y: jnp.ndarray, t: jnp.ndarray) -> float:
    delta = 1e-7
    batch_size = y.shape[0]

    if t.shape == y.shape:  # one-hot targets
        return float(-jnp.sum(t * jnp.log(y + delta)) / batch_size)

    # label-index targets: t shape (batch, 1)
    labels = t[:, 0].astype(jnp.int32)
    log_probs = jnp.log(y[jnp.arange(batch_size), labels] + delta)

    return float(-log_probs.sum() / batch_size)


class SoftmaxWithLoss:
    def __init__(self):
        self.Y: Optional[jnp.ndarray] = None
        self.T: Optional[jnp.ndarray] = None

    def forward(self, x: jnp.ndarray, t: jnp.ndarray) -> float:
        self.T = t
        self.Y = _softmax(x)
        return _cross_entropy_error(self.Y, self.T)

    def backward(self, dout: float = 1.0) -> jnp.ndarray:
        assert self.Y is not None and self.T is not None, "Forward must be called before Backward."

        batch_size = self.T.shape[0]

        if self.T.shape == self.Y.shape:  # one-hot
            dx = (self.Y - self.T) / batch_size
        else:  # label indices in column 0
            labels = self.T[:, 0].astype(jnp.int32)
            dx = jnp.array(self.Y)  # JAX arrays are immutable — create a copy
            dx = dx.at[jnp.arange(batch_size), labels].add(-1.0)
            dx = dx / batch_size

        if dout != 1.0:
            dx = dx * dout

        return dx

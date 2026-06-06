# import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from typing import Optional
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class BaseOptimizer(ABC):
    """
    Base class for optimizers that maintain velocity/accumulator state
    per weight matrix and bias vector.
    State is lazily initialized on the first call to update().
    """

    def __init__(self):
        self.vW: Optional[list[jnp.ndarray]] = None
        self.vB: Optional[list[jnp.ndarray]] = None

    def _initialize( self,
        weights: list[jnp.ndarray],
        biases: list[jnp.ndarray],
    ) -> None:
        if self.vW is None:
            self.vW = [jnp.zeros_like(w) for w in weights]
            self.vB = [jnp.zeros_like(b) for b in biases]

    @abstractmethod
    def update( self,
        weights: list[jnp.ndarray],
        biases: list[jnp.ndarray],
        dW: list[jnp.ndarray],
        dB: list[jnp.ndarray],
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        """Returns updated (weights, biases) — JAX arrays are immutable."""
    ...


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------
class SGD:
    """
    Stochastic Gradient Descent.
    w ← w - lr * dw
    """

    def __init__(self, rate: float = 0.01):
        self.rate = rate

    def update( self,
        weights: list[jnp.ndarray],
        biases: list[jnp.ndarray],
        dW: list[jnp.ndarray],
        dB: list[jnp.ndarray],
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        new_W = [w - self.rate * g for w, g in zip(weights, dW)]
        new_B = [b - self.rate * g for b, g in zip(biases, dB)]
        return new_W, new_B


# ---------------------------------------------------------------------------
# Momentum SGD
# ---------------------------------------------------------------------------
class Momentum(BaseOptimizer):
    """
    Momentum SGD.
    v ← momentum * v - lr * g
    w ← w + v
    """

    def __init__(self, rate: float = 0.01, momentum: float = 0.9):
        super().__init__()
        self.rate = rate
        self.momentum = momentum

    def update( self,
        weights: list[jnp.ndarray],
        biases: list[jnp.ndarray],
        dW: list[jnp.ndarray],
        dB: list[jnp.ndarray],
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        self._initialize(weights, biases)
        assert self.vW is not None and self.vB is not None

        new_W, new_B = [], []
        new_vW, new_vB = [], []

        for i in range(len(weights)):
            vw = self.momentum * self.vW[i] - self.rate * dW[i]
            vb = self.momentum * self.vB[i] - self.rate * dB[i]
            new_vW.append(vw)
            new_vB.append(vb)
            new_W.append(weights[i] + vw)
            new_B.append(biases[i] + vb)

        self.vW = new_vW
        self.vB = new_vB
        return new_W, new_B


# ---------------------------------------------------------------------------
# Nesterov Accelerated Gradient
# http://arxiv.org/abs/1212.0901
# ---------------------------------------------------------------------------
class Nesterov(BaseOptimizer):
    """
    Nesterov's Accelerated Gradient.
    v_new ← momentum * v - lr * g
    w ← w - momentum * v_prev + (1 + momentum) * v_new
    """

    def __init__(self, rate: float = 0.01, momentum: float = 0.9):
        super().__init__()
        self.rate = rate
        self.momentum = momentum

    def update( self, weights, biases, dW, dB ): 
        self._initialize( weights, biases )
        assert self.vW is not None and self.vB is not None

        new_W, new_B = [], []
        new_vW, new_vB = [], []

        for i in range(len(weights)):
            vw_prev = self.vW[i]
            vw = self.momentum * vw_prev - self.rate * dW[i]
            new_vW.append(vw)
            new_W.append( weights[i] - self.momentum * vw_prev + (1 + self.momentum) * vw )

            vb_prev = self.vB[i]
            vb = self.momentum * vb_prev - self.rate * dB[i]
            new_vB.append(vb)
            new_B.append( biases[i] - self.momentum * vb_prev + (1 + self.momentum) * vb )

        self.vW = new_vW
        self.vB = new_vB

        return new_W, new_B


# ---------------------------------------------------------------------------
# AdaGrad
# http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
# ---------------------------------------------------------------------------
class AdaGrad(BaseOptimizer):
    """
    AdaGrad: accumulates squared gradients to adapt the learning rate per parameter.
    h ← h + g²
    w ← w - lr * g / (sqrt(h) + ε)
    """

    def __init__(self, rate: float = 0.01):
        super().__init__()
        self.rate = rate

    def update( self, weights, biases, dW, dB ):        
        self._initialize(weights, biases)
        assert self.vW is not None and self.vB is not None

        new_W, new_B = [], []

        for i in range(len(weights)):
            self.vW[i] = self.vW[i] + dW[i] ** 2
            new_W.append(weights[i] - self.rate * dW[i] / (jnp.sqrt(self.vW[i]) + 1e-7))

            self.vB[i] = self.vB[i] + dB[i] ** 2
            new_B.append(biases[i] - self.rate * dB[i] / (jnp.sqrt(self.vB[i]) + 1e-7))

        return new_W, new_B


# ---------------------------------------------------------------------------
# RMSprop
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# ---------------------------------------------------------------------------
class RmsProp(BaseOptimizer):
    """
    RMSprop: exponential moving average of squared gradients.
    h ← decay * h + (1 - decay) * g²
    w ← w - lr * g / (sqrt(h) + ε)
    """

    def __init__(self, rate: float = 0.01, decay: float = 0.99):
        super().__init__()
        self.rate = rate
        self.decay = decay

    def update( self, weights, biases, dW, dB ):        
        self._initialize( weights, biases )
        assert self.vW is not None and self.vB is not None       

        new_W, new_B = [], []

        for i in range(len(weights)):
            self.vW[i] = self.decay * self.vW[i] + (1.0 - self.decay) * dW[i] ** 2
            new_W.append( weights[i] - self.rate * dW[i] / (jnp.sqrt(self.vW[i]) + 1e-7) )

            self.vB[i] = self.decay * self.vB[i] + (1.0 - self.decay) * dB[i] ** 2
            new_B.append( biases[i] - self.rate * dB[i] / (jnp.sqrt(self.vB[i]) + 1e-7) )

        return new_W, new_B


# ---------------------------------------------------------------------------
# Adam
# http://arxiv.org/abs/1412.6980v8
# ---------------------------------------------------------------------------
class Adam(BaseOptimizer):
    """
    Adam: adaptive moment estimation.
    m ← β1 * m + (1 - β1) * g          (first moment)
    v ← β2 * v + (1 - β2) * g²         (second moment)
    lr_t = lr * sqrt(1 - β2^t) / (1 - β1^t)   (bias-corrected step)
    w ← w - lr_t * m / (sqrt(v) + ε)
    """

    def __init__(self, rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__()
        self.rate = rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter: int = 0
        self.mW: Optional[list[jnp.ndarray]] = None
        self.mB: Optional[list[jnp.ndarray]] = None

    def update( self, weights, biases, dW, dB ):
        if self.mW is None:
            self.mW = [jnp.zeros_like(w) for w in weights]
            self.vW = [jnp.zeros_like(w) for w in weights]
            self.mB = [jnp.zeros_like(b) for b in biases]
            self.vB = [jnp.zeros_like(b) for b in biases]

            assert self.mW is not None and self.vW is not None
            assert self.mB is not None and self.vB is not None

        self.iter += 1
        lr_t = ( self.rate * jnp.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) )

        new_W, new_B = [], []

        for i in range(len(weights)):
            # Weights
            self.mW[i] = self.beta1 * self.mW[i] + (1.0 - self.beta1) * dW[i]
            self.vW[i] = self.beta2 * self.vW[i] + (1.0 - self.beta2) * dW[i] ** 2
            new_W.append(weights[i] - lr_t * self.mW[i] / (jnp.sqrt(self.vW[i]) + 1e-7))

            # Biases
            self.mB[i] = self.beta1 * self.mB[i] + (1.0 - self.beta1) * dB[i]
            self.vB[i] = self.beta2 * self.vB[i] + (1.0 - self.beta2) * dB[i] ** 2
            new_B.append( biases[i] - lr_t * self.mB[i] / (jnp.sqrt(self.vB[i]) + 1e-7) )

        return new_W, new_B
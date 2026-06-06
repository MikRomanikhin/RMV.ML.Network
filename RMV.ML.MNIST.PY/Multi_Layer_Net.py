import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from layers import Relu, Sigmoid, Affine, SoftmaxWithLoss
from optimizer import SGD, Momentum, Nesterov, Adam, AdaGrad, RmsProp


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ActivationType(Enum):
    Relu    = "Relu"
    Sigmoid = "Sigmoid"


class OptimizerType(Enum):
    SGD      = "SGD"
    Momentum = "Momentum"
    Nesterov = "Nesterov"
    Adam     = "Adam"
    AdaGrad  = "AdaGrad"
    RmsProp  = "RmsProp"

# ---------------------------------------------------------------------------
# AppSettings
# ---------------------------------------------------------------------------
@dataclass
class AppSettings:
    input:       int              = 784
    hidden:      list[int]        = field(default_factory=lambda: [100])
    output:      int              = 10
    activation:  ActivationType   = ActivationType.Relu
    optimizer:   OptimizerType    = OptimizerType.SGD
    decay:       float            = 0.0
    rate:        float            = 0.01
    momentum:    float            = 0.9
    iterations:  int              = 10000
    batch:       int              = 100
    epoch:       int              = 10
    weight_init_std: float        = 0.01
    print_interval: int          = 100
    stagnation: int              = 100
    train_path: str              = "D:\\MNIST\\mnist_train.csv"
    test_path:  str              = "D:\\MNIST\\mnist_test.csv"
    error_path: str              = "D:\\MNIST\\errors.txt"
    index_path: str              = "D:\\MNIST\\indexes.txt"


# ---------------------------------------------------------------------------
# MultiLayerNet
# ---------------------------------------------------------------------------
class MultiLayerNet:
    """
    Fully connected deep neural network with arbitrary hidden layers.
    Mirrors MultiLayerNet.cs — converted to JAX (immutable arrays).
    """

    def __init__(self, settings: AppSettings, key: jax.Array = random.PRNGKey(0)):
        self.input_size        = settings.input
        self.hidden_size_list  = list(settings.hidden)
        self.output_size       = settings.output
        self.hidden_layer_num  = len(settings.hidden)
        self.weight_decay_lambda = settings.decay

        # total_layers = self.hidden_layer_num + 1

        # Mutable lists — reassigned after every optimizer step
        self.weights: list[jnp.ndarray] = []
        self.biases:  list[jnp.ndarray] = []

        # Affine layer objects (hold .dW / .dB after backward)
        self.affine_layers: list[Affine] = []

        # Ordered forward/backward traversal sequence
        self.layers: list[Union[Affine, Relu, Sigmoid]] = []

        self.last_layer = SoftmaxWithLoss()
        self.optimizer  = self._build_optimizer(settings)

        # ---- weight initialisation ----------------------------------------
        key = self._init_weights(settings.activation, settings.weight_init_std, key)

        # ---- build layer stack ---------------------------------------------
        for i in range(self.hidden_layer_num):
            affine = Affine(self.weights[i], self.biases[i])
            self.affine_layers.append(affine)
            self.layers.append(affine)

            if settings.activation == ActivationType.Relu:
                self.layers.append(Relu())
            elif settings.activation == ActivationType.Sigmoid:
                self.layers.append(Sigmoid())

        last_idx   = self.hidden_layer_num
        last_affine = Affine(self.weights[last_idx], self.biases[last_idx])
        self.affine_layers.append(last_affine)
        self.layers.append(last_affine)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_optimizer(settings: AppSettings):
        match settings.optimizer:
            case OptimizerType.SGD:
                return SGD(rate=settings.rate)
            case OptimizerType.Momentum:
                return Momentum(rate=settings.rate, momentum=settings.momentum)
            case OptimizerType.Nesterov:
                return Nesterov(rate=settings.rate, momentum=settings.momentum)
            case OptimizerType.Adam:
                return Adam(rate=settings.rate)
            case OptimizerType.AdaGrad:
                return AdaGrad(rate=settings.rate)
            case OptimizerType.RmsProp:
                return RmsProp(rate=settings.rate)
            case _:
                raise ValueError(f"Unsupported optimizer: {settings.optimizer}")

    def _init_weights(
        self,
        activation: ActivationType,
        weight_init_std: float,
        key: jax.Array,
    ) -> jax.Array:
        """
        He   initialisation for ReLU  : scale = sqrt(2 / fan_in)
        Xavier initialisation for Sigmoid: scale = sqrt(1 / fan_in)
        Default                          : scale = weight_init_std (0.01)
        """
        all_sizes = [self.input_size] + self.hidden_size_list + [self.output_size]

        for i in range(len(all_sizes) - 1):
            fan_in = all_sizes[i]
            fan_out = all_sizes[i + 1]

            if activation == ActivationType.Relu:
                scale = jnp.sqrt(2.0 / fan_in)          # He
            elif activation == ActivationType.Sigmoid:
                scale = jnp.sqrt(1.0 / fan_in)          # Xavier
            else:
                scale = weight_init_std

            key, subkey = random.split(key)
            W = random.normal(subkey, shape=(fan_in, fan_out)) * scale
            b = jnp.zeros(fan_out)

            self.weights.append(W)
            self.biases.append(b)

        return key

    # ------------------------------------------------------------------
    # Forward / Predict
    # ------------------------------------------------------------------
    def _predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self, x: jnp.ndarray, t: jnp.ndarray) -> float:
        """Cross-entropy loss + L2 weight decay."""
        y = self._predict(x)

        weight_decay = sum(
            0.5 * self.weight_decay_lambda * float(jnp.sum(w ** 2))
            for w in self.weights
        )

        return self.last_layer.forward(y, t) + weight_decay

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    def accuracy( self, x: jnp.ndarray, t: jnp.ndarray ) -> tuple[float, list[int], list[int]]:
        """
        Returns (accuracy, misclassified_predicted_labels, misclassified_indexes).
        t may be one-hot (batch, classes) or label indices (batch, 1).
        """
        y = self._predict(x)
        batch_size = y.shape[0]

        y_pred = jnp.argmax(y, axis=1)  # (batch,)

        if t.shape[1] == 1:             # label indices
            t_idx = t[:, 0].astype(jnp.int32)
        else:                           # one-hot
            t_idx = jnp.argmax(t, axis=1)

        correct = int(jnp.sum(y_pred == t_idx))

        errors:  list[int] = []
        indexes: list[int] = []
        for i in range(batch_size):
            if int(y_pred[i]) != int(t_idx[i]):
                errors.append(int(y_pred[i]))
                indexes.append(i)

        return float(correct / batch_size), errors, indexes

    # ------------------------------------------------------------------
    # Gradient  (backpropagation)
    # ------------------------------------------------------------------
    def _gradient( self, x: jnp.ndarray, t: jnp.ndarray ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        """Backprop — returns (dW_list, dB_list)."""
        self.loss(x, t)                             # forward pass (populates layer state)

        dout = self.last_layer.backward(1.0)        # backward from loss

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        dW_list: list[jnp.ndarray] = []
        dB_list: list[jnp.ndarray] = []

        for i, affine in enumerate(self.affine_layers):
            assert affine.dW is not None and affine.dB is not None, "Affine layer has no gradient — was backward() called?"

            # L2 regularisation: grad_W += lambda * W
            dW_list.append(affine.dW + self.weight_decay_lambda * self.weights[i])
            dB_list.append(affine.dB)

        return dW_list, dB_list

    # ------------------------------------------------------------------
    # Update (one training step)
    # ------------------------------------------------------------------
    def update( self, x: jnp.ndarray, t: jnp.ndarray) -> None:
        """Compute gradients and apply one optimizer step."""
        dW, dB = self._gradient(x, t)

        # Optimizer returns new immutable JAX arrays
        new_W, new_B = self.optimizer.update(self.weights, self.biases, dW, dB)

        self.weights = new_W
        self.biases  = new_B

        # Sync affine layer weight references so the next forward pass
        # uses the updated parameters
        for i, affine in enumerate(self.affine_layers):
            affine.W = self.weights[i]
            affine.B = self.biases[i]
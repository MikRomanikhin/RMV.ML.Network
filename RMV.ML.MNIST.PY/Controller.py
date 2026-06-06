import time
import jax.numpy as jnp  # type: ignore
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from DataSet import DataSet
from Multi_Layer_Net import MultiLayerNet, AppSettings


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class Controller:
    """
    Trains and evaluates a MultiLayerNet on the given datasets.
    Mirrors Controller.cs — RunML() only (RunConv is commented out in C#).
    """

    LINE = 40   # separator line width

    def __init__(
        self,
        train_set: DataSet,
        test_set:  DataSet,
        settings:  AppSettings,
    ):
        self.train_set = train_set
        self.test_set  = test_set
        self.settings  = settings

    # ------------------------------------------------------------------
    # RunML
    # ------------------------------------------------------------------
    def run_ml(self) -> None:
        """
        Trains the network using mini-batches, validates at configured intervals,
        and stops early on stagnation.
        Mirrors C# RunML().
        """
        s = self.settings
        start = time.time()

        max_accuracy: float = float("-inf")
        stagnation:   int   = 0
        best_iter:    int   = 0

        # Full validation matrices (built once)
        val_x, val_t = self.test_set.to_matrices()

        elapsed = time.time() - start
        print(f"Loaded data sets. Time:{elapsed:.2f} sec")

        network = MultiLayerNet(s)

        for i in range(s.iterations):
            x_batch, t_batch = self.train_set.get_random_train(s.batch)

            network.update(x_batch, t_batch)                    # gradient step

            loss                    = network.loss(x_batch, t_batch)
            train_acc, _, _         = network.accuracy(x_batch, t_batch)

            if i % s.print_interval == 0:
                elapsed = time.time() - start
                print(
                    f"iter:{i}  loss:{loss:.4f}  accuracy:{train_acc:.4f}"
                    f"  Time={_fmt_elapsed(elapsed)}"
                )

            if i % s.epoch == 0:                                # validation pass
                test_acc, errors, indexes = network.accuracy(val_x, val_t)

                elapsed = time.time() - start
                print(
                    f"iter:{i}  validation:{test_acc:.4f}"
                    f"  Time={_fmt_elapsed(elapsed)}"
                )
                print("-" * self.LINE)

                if test_acc > max_accuracy:
                    max_accuracy = test_acc
                    stagnation   = 0
                    best_iter    = i

                    # Save misclassification info — mirrors File.WriteAllText + .Join()
                    _write_joined(s.error_path, errors)
                    _write_joined(s.index_path, indexes)
                    continue

                stagnation += 1
                if stagnation > s.stagnation:
                    elapsed = time.time() - start
                    print(
                        f"Stopping at {i} due to stagnation. "
                        f"Accuracy: {max_accuracy:.4f} at iteration {best_iter}. "
                        f"Time={_fmt_elapsed(elapsed)}"
                    )
                    break


# ---------------------------------------------------------------------------
# Helpers  (mirrors static helpers / Extender.Join in C#)
# ---------------------------------------------------------------------------
def _fmt_elapsed(seconds: float) -> str:
    """Formats elapsed seconds as hh:mm:ss — mirrors C# hh\\:mm\\:ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _write_joined(path: str, values: list, separator: str = ",") -> None:
    """
    Writes a comma-joined string to a file.
    Mirrors C# File.WriteAllText(path, errors.Join()) — Extender.Join().
    Writes empty string if list is empty (matching null-coalescing ?? string.Empty).
    """
    content = separator.join(str(v) for v in values) if values else ""
    with open(path, "w") as f:
        f.write(content)


def reshape_to_4d(matrix: jnp.ndarray, input_dim: list[int]) -> jnp.ndarray:
    """
    Reshapes a flat 2-D batch (N, C*H*W) → (N, C, H, W).
    Mirrors C# ReshapeTo4D(Matrix<double>, int[] inputDim).
    """
    c, h, w = input_dim[0], input_dim[1], input_dim[2]
    n = matrix.shape[0]
    return matrix.reshape(n, c, h, w)
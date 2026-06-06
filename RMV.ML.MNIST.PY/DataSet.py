import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore
import numpy as np


class DataSet:
    """
    Paired source/target dataset for supervised learning.
    Each source[i] corresponds to target[i].

    Internally stores data as numpy arrays for indexing flexibility;
    GetRandomTrain returns JAX arrays ready for network forward passes.
    """

    def __init__(
        self,
        source: list[np.ndarray] | np.ndarray,
        target: list[np.ndarray] | np.ndarray,
    ):
        # Store as numpy for efficient random-index slicing
        self.source: np.ndarray = (
            np.array(source) if not isinstance(source, np.ndarray) else source
        )
        self.target: np.ndarray = (
            np.array(target) if not isinstance(target, np.ndarray) else target
        )

    def __len__(self) -> int:
        return len(self.source)

    # ------------------------------------------------------------------
    # GetRandomBatch  →  DataSet
    # ------------------------------------------------------------------
    def get_random_batch(self, batch_size: int, key: jax.Array = random.PRNGKey(0)) -> "DataSet":
        """
        Returns a new DataSet containing a random subset of batch_size rows.
        Mirrors C# GetRandomBatch(int batchSize).
        """
        indices = np.random.permutation(len(self.source))[:batch_size]
        return DataSet(self.source[indices], self.target[indices])

    # ------------------------------------------------------------------
    # GetRandomTrain  →  (xBatch, tBatch) as JAX arrays
    # ------------------------------------------------------------------
    def get_random_train(
        self,
        batch_size: int,
        key: jax.Array = random.PRNGKey(0),
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Selects a random batch and returns (x_batch, t_batch) as JAX matrices.
        Mirrors C# GetRandomTrain(int batchSize).

        x_batch: (batch_size, input_size)
        t_batch: (batch_size, output_size)
        """
        indices = np.random.permutation(len(self.source))[:batch_size]

        x_batch = jnp.array(self.source[indices])   # (batch_size, input_size)
        t_batch = jnp.array(self.target[indices])   # (batch_size, output_size)

        return x_batch, t_batch

    # ------------------------------------------------------------------
    # Match / GetMaxItemIndex
    # ------------------------------------------------------------------
    def get_max_item_index(self, i: int) -> int:
        """
        Returns the index of the maximum value in target[i].
        Mirrors C# GetMaxItemIndex(int i).
        """
        return int(np.argmax(self.target[i]))

    def match(self, i: int, index: int) -> bool:
        """
        Returns True if the argmax of target[i] equals index.
        Mirrors C# Match(int i, int index).
        """
        return self.get_max_item_index(i) == index

    # ------------------------------------------------------------------
    # Convenience: full dataset as JAX matrices
    # ------------------------------------------------------------------
    def to_matrices(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the entire dataset as (x, t) JAX arrays."""
        return jnp.array(self.source), jnp.array(self.target)
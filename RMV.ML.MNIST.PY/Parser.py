# import jax.numpy as jnp  # type: ignore
import numpy as np
from typing import Optional

from DataSet import DataSet


class Parser:
    """
    Parses CSV lines and maps output labels to one-hot binary vectors.
    Mirrors Parser.cs.

    Parameters
    ----------
    outputs : int | None
        Number of output classes (required for one-hot encoding).
        Pass None only if BinaryVector / Run will not be used.
    """

    def __init__(self, outputs: Optional[int] = None):
        self.outputs = outputs

    # ------------------------------------------------------------------
    # Run  →  DataSet
    # ------------------------------------------------------------------
    def run(self, lines: list[str]) -> DataSet:
        """
        Parses CSV lines into a DataSet of normalised inputs and one-hot targets.
        Each line: label,pixel0,pixel1,...,pixelN
        Mirrors C# Run(string[] lines).
        """
        assert self.outputs is not None, "outputs must be set to use run()."

        inputs:  list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for line in lines:
            data = np.array(line.strip().split(","), dtype=np.float64)

            label  = int(data[0])       # first column is the label
            pixels = data[1:]           # remaining columns are features

            inputs.append(self._normalize(pixels))
            targets.append(self._binary_vector(label))

        return DataSet(
            source=np.array(inputs),   # (N, input_size)
            target=np.array(targets),  # (N, outputs)
        )

    # ------------------------------------------------------------------
    # GetImages  →  (images, labels)
    # ------------------------------------------------------------------
    @staticmethod
    def get_images(lines: list[str]) -> tuple[list[np.ndarray], list[int]]:
        """
        Parses CSV lines and returns raw integer pixel arrays and labels.
        Mirrors C# GetImages(string[] lines).
        """
        images: list[np.ndarray] = []
        labels: list[int]        = []

        for line in lines:
            data = np.array(line.strip().split(","), dtype=np.int32)

            labels.append(int(data[0]))   # first column is the label
            images.append(data[1:])       # remaining columns are pixel values

        return images, labels

    # ------------------------------------------------------------------
    # Normalize  — [0, 1] range  (÷255 for image data)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalises input to [0, 1] by dividing by 255.
        Mirrors C# Normalize(IEnumerable<double>).
        """
        return data / 255.0

    # ------------------------------------------------------------------
    # NormalizeM  — zero mean / unit variance  (commented out in C#)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mean( data: np.ndarray ) -> np.ndarray:
        """
        Normalises input to zero mean and unit variance.
        Mirrors C# NormalizeM (commented-out variant).
        """
        delta = 1e-8
        mean  = data.mean()
        std   = data.std()

        return (data - mean) / (std + delta)

    # ------------------------------------------------------------------
    # BinaryVector  —  one-hot encoding
    # ------------------------------------------------------------------
    def _binary_vector( self, value: int, min_val: float = 0.0, max_val: float = 1.0 ) -> np.ndarray:
        """
        Returns a one-hot vector of length self.outputs.
        result[value] = max_val, all others = min_val.
        Mirrors C# BinaryVector(int value, double min, double max).
        """
        assert self.outputs is not None
        result = np.full(self.outputs, min_val, dtype=np.float64)
        result[value] = max_val

        return result
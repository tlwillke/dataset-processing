from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

import numpy as np


class EmbeddingReader(ABC):
    @abstractmethod
    def iter_batches(self) -> Iterable[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class NpyEmbeddingReader(EmbeddingReader):
    def __init__(self, input_files: List[Path]) -> None:
        self.input_files = input_files

    def iter_batches(self) -> Iterable[np.ndarray]:
        for path in self.input_files:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"{path} must contain a 2D array, got shape {arr.shape}")
            yield np.asarray(arr, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "npy",
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }


def build_reader(source_type: str, input_files: List[Path]) -> EmbeddingReader:
    if source_type == "npy":
        return NpyEmbeddingReader(input_files)
    raise ValueError(f"Unsupported source_type: {source_type}")

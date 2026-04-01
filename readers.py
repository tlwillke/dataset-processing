import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Standard logging setup to match your output format
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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

class FvecsEmbeddingReader(EmbeddingReader):
    BATCH_SIZE = 32768

    def __init__(self, input_files: List[Path]) -> None:
        self.input_files = input_files

    def iter_batches(self) -> Iterable[np.ndarray]:
        for path in self.input_files:
            raw = np.memmap(path, dtype=np.int32, mode="r")
            if raw.size == 0:
                raise ValueError(f"{path} is empty")

            dim = int(raw[0])
            if dim <= 0:
                raise ValueError(f"{path} has invalid dimension header: {dim}")

            raw = np.memmap(path, dtype=np.int32, mode="r")
            if raw.size == 0:
                raise ValueError(f"{path} is empty")

            dim = int(raw[0])
            if dim <= 0:
                raise ValueError(f"{path} has invalid dimension header: {dim}")

            row_size = dim + 1
            if raw.size % row_size != 0:
                raise ValueError(
                    f"{path} size is not a multiple of fvecs row size {row_size}"
                )

            num_vectors = raw.size // row_size
            matrix = raw.reshape(num_vectors, row_size)

            self.current_file = str(path)

            for start in range(0, num_vectors, self.BATCH_SIZE):
                end = min(start + self.BATCH_SIZE, num_vectors)
                batch_rows = matrix[start:end]

                if not np.all(batch_rows[:, 0] == dim):
                    raise ValueError(
                        f"{path} has inconsistent dimension header in rows {start}:{end}"
                    )

                yield np.asarray(batch_rows[:, 1:].view(np.float32), dtype=np.float32)

    def describe(self) -> dict:
        return {
            "reader": "fvecs",
            "batch_size": self.BATCH_SIZE,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }

class ParquetEmbeddingReader(EmbeddingReader):
    BATCH_SIZE = 32768

    def __init__(self, input_files: List[Path], embedding_column: str) -> None:
        if not embedding_column:
            raise ValueError("ParquetEmbeddingReader requires a non-empty embedding_column")
        self.input_files = input_files
        self.embedding_column = embedding_column

    def iter_batches(self) -> Iterable[np.ndarray]:
        import pyarrow.parquet as pq

        total_vectors = 0
        batch_count = 0

        for path in self.input_files:
            # use_threads=True and memory_map=True improve I/O performance
            parquet_file = pq.ParquetFile(path, memory_map=True)

            if self.embedding_column not in parquet_file.schema_arrow.names:
                raise ValueError(
                    f"{path} does not contain parquet column '{self.embedding_column}'"
                )

            for record_batch in parquet_file.iter_batches(
                    columns=[self.embedding_column],
                    batch_size=self.BATCH_SIZE,
                    use_threads=True,
            ):
                # PERFORMANCE FIX: Direct buffer access instead of .to_pylist()
                # record_batch.column(0) is typically a ListArray or FixedSizeListArray
                col = record_batch.column(0)

                # .values gets the underlying flattened data; .to_numpy() is fast
                flattened_data = col.values.to_numpy(zero_copy_only=False)

                # Calculate dimensions and reshape back to 2D
                num_rows = len(record_batch)
                dim = len(flattened_data) // num_rows
                arr = flattened_data.reshape(num_rows, dim)

                total_vectors += num_rows

                self.current_file = str(path)

                yield np.ascontiguousarray(arr, dtype=np.float32)
                batch_count += 1

    def describe(self) -> dict:
        return {
            "reader": "parquet",
            "embedding_column": self.embedding_column,
            "batch_size": self.BATCH_SIZE,
            "num_files": len(self.input_files),
            "files": [str(p) for p in self.input_files],
        }


def build_reader(
        source_type: str,
        input_files: List[Path],
        parquet_embedding_column: Optional[str] = None,
) -> EmbeddingReader:
    if source_type == "npy":
        return NpyEmbeddingReader(input_files)

    if source_type == "parquet":
        if not parquet_embedding_column:
            raise ValueError(
                "source_type='parquet' requires parquet_embedding_column to be set"
            )
        return ParquetEmbeddingReader(input_files, parquet_embedding_column)

    if source_type == "fvecs":
        return FvecsEmbeddingReader(input_files)

    raise ValueError(f"Unsupported source_type: {source_type}")

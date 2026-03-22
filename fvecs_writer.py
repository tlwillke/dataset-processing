#!/usr/bin/env python3

import os
import struct
from pathlib import Path

import numpy as np


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Convert input to a C-contiguous float32 2D numpy array.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _dim_as_fvec_float(dim: int) -> np.float32:
    """
    Return the float32 value whose raw 4-byte representation matches
    the int32 dimension field required by fvecs.
    """
    if dim < 0:
        raise ValueError(f"Dimension must be non-negative, got {dim}")
    return np.array([dim], dtype=np.int32).view(np.float32)[0]


def write_fvecs(fname, arr):
    """
    Write a numpy array (shape: n x d) to an fvecs file.

    Each vector is stored as:
        [d (int32), x_0 (float32), x_1 (float32), ..., x_{d-1} (float32)]

    This overwrites the destination file.
    """
    arr = _normalize_array(arr)
    n, d = arr.shape

    fname = os.path.expanduser(str(fname))
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    with open(fname, "wb") as f:
        d_repr = _dim_as_fvec_float(d)

        # fvecs format on disk:
        # [[dim_bits_as_float, vec1...],
        #  [dim_bits_as_float, vec2...],
        #  ...]
        formatted = np.concatenate(
            (np.full((n, 1), d_repr, dtype=np.float32), arr),
            axis=1,
        )

        # Verify the first 4 bytes decode back to the expected dimension.
        if n > 0:
            assert struct.unpack("<I", formatted[0, 0].tobytes()) == (d,)

        formatted.tofile(f)


def append_fvecs(fname, arr):
    """
    Append a batch of vectors (shape: n x d) to an existing fvecs file.

    If the file already exists and is non-empty, this validates that the
    existing dimension matches the appended vectors' dimension.
    """
    arr = _normalize_array(arr)
    n, d = arr.shape

    fname = os.path.expanduser(str(fname))
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        existing_count, existing_dim = count_fvecs(path)
        if existing_dim != d:
            raise ValueError(
                f"Cannot append vectors of dim {d} to existing fvecs "
                f"with dim {existing_dim}: {path}"
            )

    with open(path, "ab") as f:
        d_repr = _dim_as_fvec_float(d)
        formatted = np.concatenate(
            (np.full((n, 1), d_repr, dtype=np.float32), arr),
            axis=1,
        )

        if n > 0:
            assert struct.unpack("<I", formatted[0, 0].tobytes()) == (d,)

        formatted.tofile(f)


def count_fvecs(fname):
    """
    Return (num_vectors, dim) for an fvecs file.

    Validates that the file size is consistent with the dimension encoded
    in the first record.

    Returns:
        (0, 0) for an empty file.
    """
    fname = os.path.expanduser(str(fname))
    path = Path(fname)

    if not path.exists():
        raise FileNotFoundError(f"fvecs file not found: {path}")

    size = path.stat().st_size
    if size == 0:
        return 0, 0

    with open(path, "rb") as f:
        dim_as_float = np.fromfile(f, dtype=np.float32, count=1)
        if dim_as_float.size == 0:
            return 0, 0

        dim = int(dim_as_float.view(np.int32)[0])

    if dim < 0:
        raise ValueError(f"Invalid negative dimension {dim} in {path}")

    record_size = 4 + 4 * dim
    if record_size == 0:
        raise ValueError(f"Invalid record size 0 for {path}")

    if size % record_size != 0:
        raise ValueError(
            f"Invalid fvecs file size for {path}: "
            f"{size} bytes is not divisible by record size {record_size}"
        )

    count = size // record_size
    return int(count), dim


def read_fvecs_shape(fname):
    """
    Convenience helper identical to count_fvecs, but named to emphasize
    its use for shape inspection.
    """
    return count_fvecs(fname)

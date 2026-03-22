# Vector Embedding Dataset Processing Pipeline

A reusable Python pipeline for turning downloaded embedding datasets into clean ANN benchmark artifacts from **`.npy` embedding inputs**.

Current outputs include:

- base vectors in `.fvecs`
- query vectors in `.fvecs`
- ground truth in `.ivecs`

The pipeline is designed for large embedding datasets and supports a staged workflow with logs, a run summary, and cleanup of intermediate files.

## Current scope

This repository currently supports **`.npy` input files only**.

More input readers and formats may be added later, but the current code is focused on a working `.npy`-to-benchmark-artifacts pipeline.

## What this project does

Given one or more `.npy` embedding files, this project can:

1. extract vectors into a single base `.fvecs` file
2. remove exact zero vectors
3. normalize vectors when needed
4. deduplicate vectors
5. sample query vectors without replacement from the cleaned vector set
6. generate exact ground-truth nearest neighbors for the final base/query split
7. log progress, output stats, and errors at each stage
8. clean up intermediate large `.fvecs` files after successful downstream stages

The final outputs are named using a common dataset prefix and the actual final vector counts produced.

## Repository structure

```text
.
├── config.py
├── pipeline.py
├── readers.py
├── fvecs_writer.py
├── fvecs_remove_zeros.py
├── fvecs_normalize.py
├── fvecs_deduplicator.py
├── fvecs_split.py
├── ivecs_check.py
├── knn_utils.py
└── runs/
```

## Example run

After editing `config.py`, run the pipeline from the repository root with:

```bash
python3 pipeline.py
```

Or with a specific Python interpreter:

```bash
/path/to/python pipeline.py
```

## Required `config.py` settings

At a minimum, you should set the following values in `config.py` before running the pipeline.

### Run and naming settings

- `RUN_NAME`  
  Name of the run directory under `runs/`

- `FILE_PREFIX`  
  Common prefix used to name output artifacts

Example:

```python
RUN_NAME = "wiki_mpnet_en_trial"
FILE_PREFIX = "wiki_mpnet_embeddings"
```

### Input settings

- `SOURCE_TYPE`  
  Currently should be set to `"npy"`

- `INPUT_FILES`  
  A list of `.npy` embedding files to process

Example:

```python
SOURCE_TYPE = "npy"

INPUT_FILES = [DATASET_DIR / f"emb_{i:03d}.npy" for i in range(10)] 
```

Be sure to create a .env file in the project's root directory and define `DATASET_ROOT`.

### Requested dataset sizes

- `NUM_BASE`  
  Requested final number of base vectors

- `NUM_QUERY`  
  Requested final number of query vectors

The pipeline initially extracts at least:

```text
NUM_BASE + NUM_QUERY
```

vectors from the input files.

Example:

```python
NUM_BASE = 100000
NUM_QUERY = 10000
```

### Ground truth settings

- `GT_K`  
  Number of nearest neighbors to compute

- `GT_METRIC`  
  `"ip"` or `"l2"`

- `GT_SHUFFLE`  
  Whether to let `knn_utils.py` shuffle before ground truth generation

- `GT_GPUS`  
  `"-1"` for CPU, or values such as `"0"` or `"0,1"` for GPU execution

Example:

```python
GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = False
GT_GPUS = "-1"
```

### Pipeline behavior settings

- `CLEANUP_INTERMEDIATE_FVECS`  
  If `True`, intermediate `.fvecs` files are deleted after successful downstream stages

- `OVERWRITE`  
  If `False`, stages with existing outputs are skipped

Example:

```python
CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False
```

## Minimal example `config.py` section

```python
from pathlib import Path
import sys

RUN_NAME = "wiki_mpnet_en_trial"
FILE_PREFIX = "wiki_mpnet_embeddings"
CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

RUN_DIR = Path("runs") / RUN_NAME

NUM_BASE = 100000
NUM_QUERY = 10000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = False
GT_GPUS = "-1"

SOURCE_TYPE = "npy"

dataset_root = os.environ.get("DATASET_ROOT")
if not dataset_root:
    raise RuntimeError(
        "DATASET_ROOT is not set. "
        "Example: export DATASET_ROOT=/path/to/your/datasets"
    )

DATASET_ROOT = Path(dataset_root)
DATASET_NAME = "mpnet-43m" # Just an example.  Put whatever you'd like
EMBED_SUBDIR = "data/en/embs"

DATASET_DIR = DATASET_ROOT / DATASET_NAME / EMBED_SUBDIR
INPUT_FILES = [DATASET_DIR / f"emb_{i:03d}.npy" for i in range(10)] # Match the file naming conventions from your download
```

## Notes on output

The pipeline writes outputs into:

```text
runs/<RUN_NAME>/
```

At the end of a successful run, the final artifacts are renamed using:

- `FILE_PREFIX`
- the actual final base count
- the actual final query count
- the ground truth metric and `k`

Typical final outputs look like:

- `<prefix>_base_<actual_count>.fvecs`
- `<prefix>_query_<actual_count>.fvecs`
- `<prefix>_gt_<metric>_<k>.ivecs`

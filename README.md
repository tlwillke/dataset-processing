# Vector Embedding Dataset Processing Pipeline

A reusable Python pipeline for turning downloaded embedding datasets into clean ANN benchmark artifacts.

### Current Outputs
- **Base vectors**: `.fvecs`
- **Query vectors**: `.fvecs`
- **Ground truth**: `.ivecs`

The pipeline is designed for large embedding datasets and supports a staged workflow with logs, a run summary, cleanup of intermediate files, and dataset-specific configuration.

---

## Current Scope

This repository currently supports:
- **.npy embedding inputs**
- **parquet embedding inputs**, where a configured parquet column contains one embedding vector per row as a list of floating-point values
- **.fvecs embedding inputs**, for cases where embeddings are already stored in ANN-style `.fvecs` files and should be processed directly

---

## What This Project Does

Given one or more embedding files, this project can:
1. Extract vectors into a single base `.fvecs` file
2. Remove exact zero vectors
3. Normalize vectors when needed
4. Deduplicate vectors
5. Sample query vectors without replacement from the cleaned vector set
6. Generate exact ground-truth nearest neighbors for the final base/query split
7. Log progress, output stats, and errors at each stage
8. Clean up intermediate large `.fvecs` files after successful downstream stages

---

## Getting Started

### Example Run
After editing `config.py`, run the pipeline from the repository root with:

    python3 processing.py

---

## Configuration (`config.py`)

At a minimum, you should set the following values in `config.py` before running the pipeline.

### Dataset Selection
Select a dataset specification module:

    from datasets import wiki_mpnet_embeddings as dataset
    # or:
    from datasets import tahoe_x1_embeddings as dataset

### Required Settings

| Setting | Description |
| :--- | :--- |
| **RUN_NAME** | Name of the run directory under runs/ |
| **FILE_PREFIX** | Common prefix for output artifacts |
| **NUM_BASE** | Requested final number of base vectors |
| **NUM_QUERY** | Requested final number of query vectors |
| **SOURCE_TYPE** | Input format: `"npy"`, `"parquet"`, or `"fvecs"` |

### Ground Truth Settings
- **GT_K**: Number of nearest neighbors to compute
- **GT_METRIC**: `"ip"` or `"l2"`
- **GT_SHUFFLE**: Whether to let knn_utils.py shuffle before ground truth generation
- **GT_GPUS**: `"-1"` for CPU, or values such as `"0"` or `"0,1"` for GPU execution

---

## Minimal Example Configuration

    from pathlib import Path
    import os
    from dotenv import load_dotenv
    from datasets import wiki_mpnet_embeddings as dataset

    load_dotenv()

    RUN_NAME = "wiki_mpnet_en_trial"
    FILE_PREFIX = dataset.FILE_PREFIX
    CLEANUP_INTERMEDIATE_FVECS = True
    OVERWRITE = False

    RUN_DIR = Path("runs") / RUN_NAME

    NUM_BASE = 100000
    NUM_QUERY = 10000

    GT_K = 100
    GT_METRIC = "ip"
    GT_SHUFFLE = False
    GT_GPUS = "-1"

    SOURCE_TYPE = dataset.SOURCE_TYPE
    PARQUET_EMBEDDING_COLUMN = dataset.PARQUET_EMBEDDING_COLUMN

    dataset_root = os.environ.get("DATASET_ROOT")
    if not dataset_root:
        raise RuntimeError("DATASET_ROOT is not set.")

    DATASET_ROOT = Path(dataset_root)
    INPUT_FILES = dataset.input_files(DATASET_ROOT, 10)

---

## Notes on Output

The pipeline writes outputs into `runs/<RUN_NAME>/`. Final artifacts are named using the `FILE_PREFIX`, actual vector counts, and ground truth parameters.

Example:
- `<prefix>_base_<actual_count>.fvecs`
- `<prefix>_query_<actual_count>.fvecs`
- `<prefix>_gt_<metric>_<k>.ivecs`

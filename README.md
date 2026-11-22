# Vector Embedding Dataset Processing Pipeline

A Python pipeline that turns embedding datasets into clean benchmark sets for approximate nearest-neighbor
(ANN) search: base vectors, held-out query vectors and exact ground truth.

It has two parts:

- **`hf_downloader.py`** downloads embedding files from a Hugging Face dataset repo.
- **`processing.py`** cleans them, splits off queries and computes ground truth.

Each part reads its settings from a small Python config file, one per dataset.

## Outputs

| File | Contents |
| :--- | :--- |
| `<prefix>_base_<count>.fvecs` | Base vectors (`.bvecs` for int8 datasets) |
| `<prefix>_query_<count>.fvecs` | Query vectors, sampled from the cleaned vectors and removed from the base set (`.bvecs` for int8) |
| `<prefix>_gt_<metric>_<k>.ivecs` | The exact `k` nearest base vectors of each query |
| `<prefix>_dedup_report.txt` | Duplicates found and removed |
| `<prefix>_pipeline.log` | Log of every stage |
| `<prefix>_summary.json` | Settings, per-stage counts and timings, and the final file paths |

`.fvecs` and `.ivecs` are the standard ANN benchmark formats: each vector is a little-endian `int32` dimension
followed by that many `float32` or `int32` values. `.bvecs` has the same layout with one byte per value. Here
those bytes are **signed** int8 (−128..127), unlike TexMex/BIGANN `.bvecs` files, which are unsigned (0..255).

## What the pipeline does

1. **Extract**: reads the selected input files into one base file.
2. **Remove zeros**: drops vectors whose length is at most `ZERO_TOLERANCE`.
3. **Normalize**: scales float vectors to unit length unless they already are, within
   `NORMALIZATION_TOLERANCE`. Int8 vectors are passed through unchanged, and the summary records the stage as
   skipped.
4. **Deduplicate**: removes exact duplicates and writes a report.
5. **Split**: samples `NUM_QUERY` query vectors at random without replacement (seed 42); the rest are the base
   vectors.
6. **Ground truth**: computes the exact `GT_K` nearest neighbors of each query with FAISS, on CPU or GPU,
   optionally shuffling the base and query vectors first (`GT_SHUFFLE`).

Intermediate files are deleted as soon as the next stage succeeds (`CLEANUP_INTERMEDIATE_FVECS`).

### Supported inputs

| `SOURCE_TYPE` | Input |
| :--- | :--- |
| `"npy"` | NumPy `.npy` arrays, one embedding per row |
| `"parquet"` | Parquet files with one embedding per row in the column `PARQUET_EMBEDDING_COLUMN` |
| `"fvecs"` | Embeddings already in `.fvecs` files |

**Int8 datasets:** set `SOURCE_INT8 = True` in the processing config to keep signed int8 embeddings (for
example, from parquet) as int8 through every stage, writing `.bvecs`. The reader stops with an error naming the
file if any value is not a whole number in −128..127, rather than silently changing it. Unsigned uint8 data
must be converted to signed values first.

## Getting started

### 1. Set up the environment

With conda (recommended; Python and every package pinned):

    conda env create -f environment.yml
    conda activate dataset-processing

Or with pip, in an environment of your own:

    pip install -r requirements.txt

### 2. Download a dataset

1. Copy `download_configs/example_download_config.py` to a new file in `download_configs/`, and set:
   - `REPO_ID`: the Hugging Face dataset repo, such as `"olmer/wiki_mpnet_embeddings"`
   - `LOCAL_DIR`: where to save the files
   - `SELECTION_MODE`: which files to download (see below)
   - `TOKEN`: `None` for public repos; for gated or private ones, a token, or run `hf auth login` once
2. Point `hf_downloader.py` at your config by editing its one import line:

       from download_configs import my_dataset_download as repo

3. Leave `DRY_RUN = True` for a first run, which lists the files it would download, then set it to `False`
   and run again:

       python3 hf_downloader.py

`SELECTION_MODE` chooses the files:

| Mode | Downloads |
| :--- | :--- |
| `"all"` | Every file in the repo |
| `"first_n"` | The first `FIRST_N` files that pass the filters below |
| `"pattern"` | Every file that passes the filters below |
| `"explicit"` | Exactly the repo paths listed in `EXPLICIT_FILES` |

The filters keep only paths containing one of `SELECT_SUBSTRINGS`, drop paths containing any of
`EXCLUDE_SUBSTRINGS`, and keep only paths ending in one of `ALLOWED_SUFFIXES`. An empty list allows everything.

### 3. Process it

1. Copy `processing_configs/example_proc_config.py` to a new file in `processing_configs/` and set at least:

   | Setting | Description |
   | :--- | :--- |
   | `OUTPUT_DIR` | Directory for every output and intermediate file |
   | `FILE_PREFIX` | Prefix for the output file names |
   | `SOURCE_TYPE` | `"npy"`, `"parquet"` or `"fvecs"` |
   | `PARQUET_EMBEDDING_COLUMN` | The embedding column, for parquet inputs; otherwise `None` |
   | `INPUT_DIR` | Directory of input files, usually the downloader's `LOCAL_DIR` |
   | `SELECTION_MODE` | Which input files to use: the same modes and filters as the downloader, with `EXPLICIT_INPUT_FILES` for `"explicit"` |
   | `NUM_QUERY` | Number of query vectors to hold out |
   | `NUM_BASE` | Number of base vectors, or `None` to use every remaining vector |
   | `GT_K`, `GT_METRIC` | Neighbors per query, and `"ip"` (inner product) or `"l2"` |
   | `GT_GPUS` | `"-1"` for CPU, or GPU ids such as `"0"` or `"0,1"` |
   | `SOURCE_INT8` | Optional; `True` for signed int8 embeddings |

   The run stops after extraction if the inputs hold fewer than `NUM_BASE + NUM_QUERY` vectors.
2. Point `config.py` at your config by editing its one import line:

       from processing_configs import my_dataset_proc as dataset

3. Run it from the repository root:

       python3 processing.py

   The final files land in `OUTPUT_DIR`; check `<prefix>_summary.json` for the counts actually produced.

`download_configs/` and `processing_configs/` hold ready-made configs for the datasets processed so far. Their
paths point at the machines they were run on, so edit `LOCAL_DIR`, `INPUT_DIR` and `OUTPUT_DIR` before reusing
one.

## Other tools

Each stage is also a standalone script with its own `--help`: `fvecs_remove_zeros.py`, `fvecs_normalize.py`,
`fvecs_deduplicator.py`, `fvecs_split.py` and `knn_utils.py` (ground truth). `ivecs_check.py` asks for an `.ivecs`
file and prints its row count, row length and first row.

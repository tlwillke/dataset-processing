from pathlib import Path
import sys
from dotenv import load_dotenv

# ------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------

RUN_NAME = "wiki_mpnet_10m"
FILE_PREFIX = "wiki_mpnet_embeddings"
CLEANUP_INTERMEDIATE_FVECS = True # Will delete intermediate files produced by zero removal, etc.

RUN_DIR = Path("runs") / RUN_NAME

# ------------------------------------------------------------
# Working files.  Do not update.
# ------------------------------------------------------------
RAW_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_raw_base.fvecs"
NONZERO_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_nonzero_base.fvecs"
NORMALIZED_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_normalized_base.fvecs"
DEDUP_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_base.fvecs"
SPLIT_QUERY_FVECS = RUN_DIR / f"{FILE_PREFIX}_base_query.fvecs"
SPLIT_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_base_base.fvecs"
SPLIT_QPARTS_DIR = Path(f"{DEDUP_BASE_FVECS.with_suffix('')}_qparts")
SPLIT_BPARTS_DIR = Path(f"{DEDUP_BASE_FVECS.with_suffix('')}_bparts")
GT_PROCESSED_BASE_FVECS = RUN_DIR / f"{FILE_PREFIX}_gt_processed_base.fvecs"
GT_PROCESSED_QUERY_FVECS = RUN_DIR / f"{FILE_PREFIX}_gt_processed_query.fvecs"
GROUND_TRUTH_FILE = RUN_DIR / "ground_truth.ivecs"
# ------------------------------------------------------------

DEDUP_REPORT = RUN_DIR / f"{FILE_PREFIX}_dedup_report.txt"
DEDUP_TEMP_DIR = RUN_DIR / f"{FILE_PREFIX}_dedup_temp"

NUM_QUERY = 10000 # Set an integer target number.  The final count may be less due to zero removal and dedup.
NUM_BASE = 10000000  # set an integer for truncation (otherwise all listed input files will be processed.  The final count may be less due to zero removal and dedup.
GT_K = 100
GT_METRIC = "ip"      # "ip" or "l2"
GT_SHUFFLE = True
GT_GPUS = "-1"        # "-1" for CPU, e.g. "0" for one GPU, "0,1" for multi-GPU

FINAL_GROUND_TRUTH = RUN_DIR / f"{FILE_PREFIX}_gt_{GT_METRIC}_{GT_K}.ivecs"

LOG_FILE = RUN_DIR / "pipeline.log"
SUMMARY_FILE = RUN_DIR / "summary.json"

OVERWRITE = False

# ------------------------------------------------------------
# Input data
# ------------------------------------------------------------

SOURCE_TYPE = "npy"

from pathlib import Path
import os

# Load the .env file from the current directory
load_dotenv()

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
INPUT_FILES = [DATASET_DIR / f"emb_{i:03d}.npy" for i in range(12)] # Match the file naming conventions from your download

# ------------------------------------------------------------
# External stage commands
# ------------------------------------------------------------

REMOVE_ZEROS_CMD = [
    sys.executable,
    "-u",
    "fvecs_remove_zeros.py",
    "--input", str(RAW_BASE_FVECS),
    "--output", str(NONZERO_BASE_FVECS),
]

NORMALIZE_CMD = [
    sys.executable,
    "-u",
    "fvecs_normalize.py",
    "--input", str(NONZERO_BASE_FVECS),
    "--output", str(NORMALIZED_BASE_FVECS),
]

DEDUP_CMD = [
    sys.executable,
    "-u",
    "fvecs_deduplicator.py",
    str(NORMALIZED_BASE_FVECS),
    "--output", str(DEDUP_BASE_FVECS),
    "--report_file", str(DEDUP_REPORT),
    "--reporting_threshold", "1",
    "--chunk_size", "200000",
    "--temp_dir", str(DEDUP_TEMP_DIR),
]

SPLIT_CMD = [
    sys.executable,
    "-u",
    "fvecs_split.py",
    str(DEDUP_BASE_FVECS),
    "--num_query", str(NUM_QUERY),
    "--seed", "47",
]

GROUND_TRUTH_CMD = [
    sys.executable,
    "-u",
    "knn_utils.py",
    "--base", str(SPLIT_BASE_FVECS),
    "--query", str(SPLIT_QUERY_FVECS),
    "--output", str(GROUND_TRUTH_FILE),
    "--processed_base_out", str(GT_PROCESSED_BASE_FVECS),
    "--processed_query_out", str(GT_PROCESSED_QUERY_FVECS),
    "--k", str(GT_K),
    "--metric", GT_METRIC,
    "--gpus", GT_GPUS,
]

if GT_SHUFFLE:
    GROUND_TRUTH_CMD.append("--shuffle")

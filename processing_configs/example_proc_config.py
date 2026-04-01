from pathlib import Path

# ------------------------------------------------------------
# Output location
# ------------------------------------------------------------

# All run artifacts go under this directory.
OUTPUT_DIR = Path("/path/to/output/run_dir")

# ------------------------------------------------------------
# Dataset / file format metadata
# ------------------------------------------------------------

# Used to build output filenames inside OUTPUT_DIR, e.g.
#   <FILE_PREFIX>_raw_base.fvecs
#   <FILE_PREFIX>_base_100000.fvecs
#
# Also suitable for logging / summary metadata.
FILE_PREFIX = "example_dataset_prefix"

# Used by readers.build_reader(...)
# Expected values in your current code: "npy" or "parquet"
SOURCE_TYPE = "npy"

# Used only when SOURCE_TYPE == "parquet".
# Set to None for .npy datasets.
PARQUET_EMBEDDING_COLUMN = None

# ------------------------------------------------------------
# Local input file selection
# ------------------------------------------------------------

# Directory containing local embedding files to process.
# This is independent of hf_downloader.py.
INPUT_DIR = Path("/path/to/local/embedding/files")

# Choose exactly one SELECTION_MODE:
#
#   "all"
#       Process every file under INPUT_DIR.
#
#   "first_n"
#       Process the first FIRST_N files after pattern filtering.
#
#   "explicit"
#       Process exactly the files listed in EXPLICIT_INPUT_FILES.
#       Entries may be absolute paths or relative to INPUT_DIR.
#
#   "pattern"
#       Process files selected only by the pattern filters below.
SELECTION_MODE = "first_n"

# Used only when SELECTION_MODE == "first_n".
FIRST_N = 12

# Used only when SELECTION_MODE == "explicit".
EXPLICIT_INPUT_FILES = [
    # "emb_000.npy",
    # "emb_001.npy",
    # "/absolute/path/to/emb_002.npy",
]

# Used when SELECTION_MODE == "first_n" or "pattern".
#
# SELECT_SUBSTRINGS:
#   Keep only files whose paths contain at least one of these substrings.
#   Leave empty to allow all paths.
#
# EXCLUDE_SUBSTRINGS:
#   Exclude any file whose path contains one of these substrings.
#
# ALLOWED_SUFFIXES:
#   Keep only files ending in one of these suffixes.
#   Leave empty to allow all suffixes.
SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".npy"]

# ------------------------------------------------------------
# Requested output sizes
# ------------------------------------------------------------

# fvecs_split.py target query count.
NUM_QUERY = 10000

# Initial extraction target is NUM_BASE + NUM_QUERY.
# If not enough vectors are available, processing.py fails before
# downstream processing.
# Set to None to process all selected input files.
NUM_BASE = 100000

# ------------------------------------------------------------
# Ground truth configuration
# ------------------------------------------------------------

GT_K = 100
GT_METRIC = "ip"      # "ip", "l2", or "cosine"
GT_SHUFFLE = True
GT_GPUS = "-1"        # "-1" for CPU, e.g. "0" or "0,1" for GPU

# ------------------------------------------------------------
# Run behavior
# ------------------------------------------------------------

# If True, delete intermediate fvecs files and temp dirs.
CLEANUP_INTERMEDIATE_FVECS = True

# If False, existing outputs cause stages to be skipped.
# If True, existing outputs may be replaced.
OVERWRITE = False

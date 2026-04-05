from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-clean")

FILE_PREFIX = "ada_002_100k"
SOURCE_TYPE = "fvecs"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/home/ted_willke/datasets-temp")

SELECTION_MODE = "explicit"
EXPLICIT_INPUT_FILES = ["ada_002_110000.fvecs"]
FIRST_N = 0

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".fvecs"]

NUM_QUERY = 10000
NUM_BASE = 100000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

ZERO_TOLERANCE = 1e-06
NORMALIZATION_TOLERANCE = 1e-05

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

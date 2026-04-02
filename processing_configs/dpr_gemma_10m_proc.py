from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/dpr-gemma-10m")

FILE_PREFIX = "dpr_gemma"
SOURCE_TYPE = "parquet"
PARQUET_EMBEDDING_COLUMN = "embedding"

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/dpr_gemma/data")

SELECTION_MODE = "first_n"
EXPLICIT_INPUT_FILES = []
FIRST_N = 220

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 10000
NUM_BASE = 10000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

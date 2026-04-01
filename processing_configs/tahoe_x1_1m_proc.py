from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/tahoe-x1-1m")

FILE_PREFIX = "tahoe_x1_embeddings"
SOURCE_TYPE = "parquet"
PARQUET_EMBEDDING_COLUMN = "mosaicfm-3b-prod-cont-MFMv2"

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/tahoe-x1-100m/data")

SELECTION_MODE = "first_n"
FIRST_N = 4
EXPLICIT_INPUT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 10000
NUM_BASE = 1000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

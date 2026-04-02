from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/esmc-600-1m")

FILE_PREFIX = "esmc_600_embeddings"
SOURCE_TYPE = "parquet"
PARQUET_EMBEDDING_COLUMN = "mean_embedding"

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/esmc-embeddings/data")

SELECTION_MODE = "first_n"
EXPLICIT_INPUT_FILES = []
FIRST_N = 100

SELECT_SUBSTRINGS = ["esmc_600m"]
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 10000
NUM_BASE = 1250000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

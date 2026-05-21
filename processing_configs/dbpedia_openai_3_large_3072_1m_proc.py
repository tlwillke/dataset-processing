from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-clean/openai-3072-1m")

FILE_PREFIX = "dbpedia_openai_3_large_3072"
SOURCE_TYPE = "parquet"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = "text-embedding-3-large-3072-embedding"

INPUT_DIR = Path("/home/ted_willke/datasets-temp/dbpedia-entities-openai3-text-embedding-3-large-3072-1M/data")

SELECTION_MODE = "all"
EXPLICIT_INPUT_FILES = []
FIRST_N = 800

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 1000
NUM_BASE = 999000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

ZERO_TOLERANCE = 1e-06
NORMALIZATION_TOLERANCE = 1e-5

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

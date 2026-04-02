from pathlib import Path

OUTPUT_DIR = Path("/mnt/raid10/datasets-ash/temp/mpnet-temp")

FILE_PREFIX = "wiki_mpnet_embeddings"
SOURCE_TYPE = "npy"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/mnt/raid10/datasets-ash/temp/mpnet-temp")

SELECTION_MODE = "first_n"
FIRST_N = 3
EXPLICIT_INPUT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".npy"]

NUM_QUERY = 10000
NUM_BASE = 100000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

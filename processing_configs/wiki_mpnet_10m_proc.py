from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/mpnet-10m")

FILE_PREFIX = "wiki_mpnet_embeddings"
SOURCE_TYPE = "npy"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/mpnet-43m/data/en/embs")

SELECTION_MODE = "first_n"
FIRST_N = 20
EXPLICIT_INPUT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".npy"]

NUM_QUERY = 10000
NUM_BASE = 10000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/temp")

FILE_PREFIX = "cohere_msmarco-v2-embed-english-v3"
SOURCE_TYPE = "fvecs"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/home/ted_willke/Projects/vector-datasets/cohere-msmarco-138m")

SELECTION_MODE = "explicit"
EXPLICIT_INPUT_FILES = ["cohere_msmarco-v2-embed-english-v3_base.fvecs"]
FIRST_N = 0

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".fvecs"]

NUM_QUERY = 10000
NUM_BASE = 1000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

ZERO_TOLERANCE = 0.0
NORMALIZATION_TOLERANCE = 1e-3

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

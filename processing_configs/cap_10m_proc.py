from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/temp")

FILE_PREFIX = "cap_embeddings"
SOURCE_TYPE = "parquet"
READER_BATCH_SIZE = 32768  # Tune for RAM vs throughput.
PARQUET_EMBEDDING_COLUMN = "embedding"

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/cap/data/TeraflopAI___Caselaw_Access_Project___Alibaba-NLP___gte-Qwen2-1.5B-instruct_clusters")

SELECTION_MODE = "first_n"
FIRST_N = 4000
EXPLICIT_INPUT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = ["cluster_0.parquet"]
ALLOWED_SUFFIXES = [".parquet"]

NUM_QUERY = 10000
NUM_BASE = 10000000

GT_K = 100
GT_METRIC = "ip"
GT_SHUFFLE = True
GT_GPUS = "-1"

ZERO_TOLERANCE = 0.0
NORMALIZATION_TOLERANCE = 1e-3

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

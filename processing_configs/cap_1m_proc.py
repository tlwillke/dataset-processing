from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/cap-1m")

FILE_PREFIX = "Caselaw_gte-Qwen2-1.5B_embeddings"
SOURCE_TYPE = "fvecs"
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/mnt/raid10/datasets-ash/downloads/cap/data/TeraflopAI___Caselaw_Access_Project___Alibaba-NLP___gte-Qwen2-1.5B-instruct_clusters")

SELECTION_MODE = "first_n"
EXPLICIT_INPUT_FILES = []
FIRST_N = 200

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

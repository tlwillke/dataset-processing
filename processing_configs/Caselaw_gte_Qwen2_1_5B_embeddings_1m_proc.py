from pathlib import Path

OUTPUT_DIR = Path("/home/ted_willke/datasets-ash/cap-1m")

FILE_PREFIX = "Caselaw_gte-Qwen2-1.5B_embeddings"
SOURCE_TYPE = "fvecs"
PARQUET_EMBEDDING_COLUMN = None

INPUT_DIR = Path("/mnt/raid10/datasets-ash/cap-6m")

SELECTION_MODE = "explicit"
EXPLICIT_INPUT_FILES = ["Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs"]
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

CLEANUP_INTERMEDIATE_FVECS = True
OVERWRITE = False

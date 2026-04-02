from pathlib import Path

REPO_ID = "tomaztc/wiki_dpr_gemma_embeddings"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/mnt/raid10/datasets-ash/downloads/dpr_gemma/data")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "all"
FIRST_N = 2
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = ["wiki_dpr_gemma_embeddings/"]
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]
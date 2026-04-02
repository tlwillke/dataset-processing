from pathlib import Path

REPO_ID = "biolm/human-proteome-esmc-embeddings"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/mnt/raid10/datasets-ash/downloads/esmc-embeddings/data")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "all"
FIRST_N = 2
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".parquet"]
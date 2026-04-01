from pathlib import Path

REPO_ID = "olmer/wiki_mpnet_embeddings"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/mnt/raid10/datasets-ash/temp/mpnet-temp")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "first_n"
FIRST_N = 3
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".npy"]
from pathlib import Path

REPO_ID = "justicedao/Caselaw_Access_Project_embeddings"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

LOCAL_DIR = Path("/mnt/raid10/datasets-ash/downloads/cap/data")
MAX_WORKERS = 16
DRY_RUN = False

PRINT_ALL_FILES = True
PRINT_SELECTED_FILES = True

SELECTION_MODE = "first_n"
FIRST_N = 5000
EXPLICIT_FILES = []

SELECT_SUBSTRINGS = ["TeraflopAI___Caselaw_Access_Project___Alibaba-NLP___gte-Qwen2-1.5B-instruct_clusters/"]
EXCLUDE_SUBSTRINGS = ["gte-small","gte-large"]
ALLOWED_SUFFIXES = [".parquet"]
from pathlib import Path

# ============================================================
# Example download configuration
#
# Copy this file to create a new repo-specific downloader config.
# Then edit the repo details, local directory, and selection mode.
#
# This file is imported by hf_downloader.py.
# ============================================================

# ------------------------------------------------------------
# Hugging Face repo settings
# ------------------------------------------------------------

REPO_ID = "olmer/wiki_mpnet_embeddings"
REPO_TYPE = "dataset"
REVISION = "main"
TOKEN = None

# ------------------------------------------------------------
# Local download settings
# ------------------------------------------------------------

# Directory where selected files will be downloaded.
LOCAL_DIR = Path("/path/to/mpnet-43m")

# Number of concurrent download workers.
MAX_WORKERS = 16

# If True, print what would be downloaded but do not download it.
DRY_RUN = True

# ------------------------------------------------------------
# Output / listing behavior
# ------------------------------------------------------------

# If True, print every file found in the repo before selection.
PRINT_ALL_FILES = True

# If True, print the selected files after selection.
PRINT_SELECTED_FILES = True

# ------------------------------------------------------------
# File selection
# ------------------------------------------------------------
#
# Choose exactly one SELECTION_MODE:
#
#   "all"
#       Download every file in the repo.
#
#   "first_n"
#       Download the first FIRST_N files after pattern filtering.
#       This is useful when you want a small subset of a large repo.
#
#   "explicit"
#       Download exactly the repo-relative file paths listed in
#       EXPLICIT_FILES.
#
#   "pattern"
#       Download files selected only by the pattern filters below.
#
# Active example below:
#   Download the first 12 .npy files in the repo.
# ------------------------------------------------------------

SELECTION_MODE = "first_n"

# Used only when SELECTION_MODE == "first_n".
FIRST_N = 12

# Used only when SELECTION_MODE == "explicit".
# These paths must match the repo's internal file paths exactly.
EXPLICIT_FILES = [
    # "data/en/embs/emb_000.npy",
    # "data/en/embs/emb_001.npy",
]

# Used when SELECTION_MODE == "first_n" or "pattern".
#
# SELECT_SUBSTRINGS:
#   Keep only files whose paths contain at least one of these substrings.
#   Leave empty to allow all paths.
#
# EXCLUDE_SUBSTRINGS:
#   Exclude any file whose path contains one of these substrings.
#
# ALLOWED_SUFFIXES:
#   Keep only files ending in one of these suffixes.
#   Leave empty to allow all suffixes.
SELECT_SUBSTRINGS = []
EXCLUDE_SUBSTRINGS = []
ALLOWED_SUFFIXES = [".npy"]

# ------------------------------------------------------------
# Other example modes
# Uncomment and adapt one of these, then comment out the active one.
# ------------------------------------------------------------

# Example: download all files in the repo
# SELECTION_MODE = "all"
# FIRST_N = None
# EXPLICIT_FILES = []
# SELECT_SUBSTRINGS = []
# EXCLUDE_SUBSTRINGS = []
# ALLOWED_SUFFIXES = []

# Example: download exactly these files
# SELECTION_MODE = "explicit"
# FIRST_N = None
# EXPLICIT_FILES = [
#     "data/en/embs/emb_000.npy",
#     "data/en/embs/emb_001.npy",
# ]
# SELECT_SUBSTRINGS = []
# EXCLUDE_SUBSTRINGS = []
# ALLOWED_SUFFIXES = []

# Example: download files by pattern only
# SELECTION_MODE = "pattern"
# FIRST_N = None
# EXPLICIT_FILES = []
# SELECT_SUBSTRINGS = ["data/en/embs/"]
# EXCLUDE_SUBSTRINGS = ["tmp", "backup"]
# ALLOWED_SUFFIXES = [".npy"]

#!/usr/bin/env python3

from pathlib import Path
from typing import Iterable, Optional
from huggingface_hub import HfApi, snapshot_download

from config import DATASET_ROOT, DATASET_NAME

# ============================================================
# Configuration
# ============================================================

# Hugging Face repo settings
REPO_ID = "olmer/wiki_mpnet_embeddings"
REPO_TYPE = "dataset"          # "dataset", "model", or "space"
REVISION = "main"
TOKEN = None                   # Set for private/gated repos if needed

# Local download settings
LOCAL_DIR = Path(DATASET_ROOT / DATASET_NAME) # From config.py
MAX_WORKERS = 16               # Parallel download workers
DRY_RUN = True                # True = list selected files only, no download

# Listing behavior
PRINT_ALL_FILES = True         # Print every file in the repo
PRINT_SELECTED_FILES = True    # Print only the chosen subset

# ------------------------------------------------------------
# File selection controls
# ------------------------------------------------------------
# A file is selected if:
#   1) it matches ANY SELECT_SUBSTRINGS (or all files if empty)
#   2) it does NOT match any EXCLUDE_SUBSTRINGS
#   3) it ends with one of ALLOWED_SUFFIXES (or any suffix if empty)
#
# Examples:
#   SELECT_SUBSTRINGS = ["data/en/embs/emb_"]
#   SELECT_SUBSTRINGS = ["data/en/embs/emb_", "data/en/embs/ids_"]
#   SELECT_SUBSTRINGS = ["data/en/paragraphs.zip"]
#
SELECT_SUBSTRINGS = [
    # "emb_000.npy",
    # "emb_001.npy",
    # "emb_002.npy",
    # "data/en/embs/ids_",
    # "data/en/paragraphs.zip",
]

EXCLUDE_SUBSTRINGS = [
    # ".md",
]

ALLOWED_SUFFIXES = [
    # ".npy",
    # ".zip",
]

# ============================================================
# Functions
# ============================================================

def list_repo_files(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str = "main",
    token: Optional[str] = None,
) -> list[str]:
    api = HfApi(token=token)
    return api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
    )


def matches_any_substring(path: str, needles: Iterable[str]) -> bool:
    needles = list(needles)
    if not needles:
        return True
    return any(s in path for s in needles)


def matches_any_suffix(path: str, suffixes: Iterable[str]) -> bool:
    suffixes = list(suffixes)
    if not suffixes:
        return True
    return any(path.endswith(s) for s in suffixes)


def select_files(
    files: list[str],
    select_substrings: Optional[Iterable[str]] = None,
    exclude_substrings: Optional[Iterable[str]] = None,
    allowed_suffixes: Optional[Iterable[str]] = None,
) -> list[str]:
    select_substrings = list(select_substrings or [])
    exclude_substrings = list(exclude_substrings or [])
    allowed_suffixes = list(allowed_suffixes or [])

    selected = []
    for path in files:
        if not matches_any_substring(path, select_substrings):
            continue
        if exclude_substrings and any(s in path for s in exclude_substrings):
            continue
        if not matches_any_suffix(path, allowed_suffixes):
            continue
        selected.append(path)

    return selected


def download_selected_files(
    repo_id: str,
    selected_files: list[str],
    local_dir: Path,
    repo_type: str = "dataset",
    revision: str = "main",
    token: Optional[str] = None,
    max_workers: int = 8,
) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)

    # Use exact selected repo paths as allow_patterns.
    # snapshot_download will download them concurrently.
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=selected_files,
        token=token,
        max_workers=max_workers,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        token=TOKEN,
    )

    print(f"\nFound {len(files)} files in {REPO_TYPE} repo '{REPO_ID}'.\n")

    if PRINT_ALL_FILES:
        print("All files:")
        for f in files:
            print(f)

    selected = select_files(
        files=files,
        select_substrings=SELECT_SUBSTRINGS,
        exclude_substrings=EXCLUDE_SUBSTRINGS,
        allowed_suffixes=ALLOWED_SUFFIXES,
    )

    print(f"\nSelected {len(selected)} file(s).\n")

    if PRINT_SELECTED_FILES:
        for f in selected:
            print(f)

    if not selected:
        print("\nNo files matched the configured filters.")
        return

    if DRY_RUN:
        print("\nDRY_RUN is True. No download performed.")
        return

    print(f"\nDownloading with max_workers={MAX_WORKERS} ...")
    snapshot_path = download_selected_files(
        repo_id=REPO_ID,
        selected_files=selected,
        local_dir=LOCAL_DIR,
        repo_type=REPO_TYPE,
        revision=REVISION,
        token=TOKEN,
        max_workers=MAX_WORKERS,
    )

    print(f"\nDone. Files downloaded under: {snapshot_path}")


if __name__ == "__main__":
    main()

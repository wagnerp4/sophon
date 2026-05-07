from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "google/gemma-4-31B-it"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gemma 4 31B Instruct weights from the Hub.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("GEMMA4_REPO_ID", DEFAULT_REPO),
        help="Hub model id (default: %(default)s or GEMMA4_REPO_ID).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(os.environ.get("GEMMA4_LOCAL_DIR", "models/google-gemma-4-31b-it")),
        help="Directory to store files (default: models/google-gemma-4-31b-it or GEMMA4_LOCAL_DIR).",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("GEMMA4_REVISION"),
        help="Optional git revision (branch name, tag, or commit).",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    kwargs = {
        "repo_id": args.repo_id,
        "local_dir": str(args.local_dir.resolve()),
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if token:
        kwargs["token"] = token
    path = snapshot_download(**kwargs)
    print(path)


if __name__ == "__main__":
    main()

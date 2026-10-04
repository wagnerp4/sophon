from __future__ import annotations

import os


def arxiv_tools_enabled() -> bool:
    return os.environ.get("SOPHON_ARXIV_TOOLS", "1").strip().lower() not in {"0", "false", "off", "no"}

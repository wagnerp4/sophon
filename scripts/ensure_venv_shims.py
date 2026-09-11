from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.device.platform import sync_shims_main


if __name__ == "__main__":
    sync_shims_main()

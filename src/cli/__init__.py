from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent orodruin.* imports end-to-end.

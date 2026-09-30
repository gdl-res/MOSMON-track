"""Make ``src/`` and the repo root importable for tests.

- ``src/`` so ``import mosmon_tracking`` works without an editable install.
- the repo root so cross-test helper imports (``from tests.test_track_postprocess
  import ...``) resolve regardless of the active environment / pytest import mode.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

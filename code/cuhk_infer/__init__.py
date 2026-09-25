"""cuhk_infer — CUHK-X Small Model Track inference package.

The ``sota_*`` inference modules live as top-level modules in the project root
(this package's parent directory). We add that root to ``sys.path`` on import so
the flat ``sota_*`` names import correctly whether you run the CLI, import the
package, or run ``sota_release_infer2.py`` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root = parent of this package dir. This is where the sota_*.py modules
# and sota_release_infer2.py live.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = ["cli", "PROJECT_ROOT"]
PROJECT_ROOT = _PROJECT_ROOT

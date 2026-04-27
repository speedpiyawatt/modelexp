from __future__ import annotations

import pathlib
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.weather.location_context import *  # noqa: F401,F403

"""Location of the versioned YAML configuration files.

Defaults to ``<repo>/configs`` when the package is imported from the source tree; when it is
installed elsewhere (the simulation image installs it system-wide) ``ARMBENCH_CONFIG_DIR``
points at a copy of that directory.
"""

import os
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
CONFIG_DIR: Final = Path(os.environ.get("ARMBENCH_CONFIG_DIR", REPO_ROOT / "configs"))
SKILLS_DIR: Final = Path(os.environ.get("ARMBENCH_SKILLS_DIR", REPO_ROOT / "skills"))
"""The frozen skill library (``library.json`` + ``FROZEN.json``); runs copy it next to the LLM
bundle and point the containers at that copy."""
ENERGY_REF_DIR: Final = Path(os.environ.get("ARMBENCH_ENERGY_REF_DIR", REPO_ROOT / "energy_ref"))
"""The energy reference of agents C and C+S (``reference.json``: baseline A's Wh per task, D8);
runs copy it next to the LLM bundle and point the containers at that copy."""

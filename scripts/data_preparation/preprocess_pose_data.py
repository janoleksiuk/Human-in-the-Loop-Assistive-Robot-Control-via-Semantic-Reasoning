from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    module_root = repo_root / "src" / "human_action_recognition" / "data"
    sys.path.insert(0, str(module_root))
    runpy.run_path(str(module_root / "data_preprocessing.py"), run_name="__main__")

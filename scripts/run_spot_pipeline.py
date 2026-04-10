from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_root = repo_root / "src" / "robotic_task_execution"
    sys.path.insert(0, str(module_root))

    from main import main as spot_main

    spot_main()


if __name__ == "__main__":
    main()

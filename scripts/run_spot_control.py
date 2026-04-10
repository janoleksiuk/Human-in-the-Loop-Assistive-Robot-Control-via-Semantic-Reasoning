from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shared_root = repo_root / "src" / "robotic_task_execution"
    module_root = shared_root / "spot_control"
    sys.path.insert(0, str(module_root))
    sys.path.insert(0, str(shared_root))

    from action_control import main as spot_control_main

    raise SystemExit(spot_control_main())


if __name__ == "__main__":
    main()

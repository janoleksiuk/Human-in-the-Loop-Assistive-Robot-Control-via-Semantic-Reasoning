from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_root = repo_root / "src" / "human_action_recognition"
    sys.path.insert(0, str(module_root))

    from predict import main as predict_main

    raise SystemExit(predict_main())


if __name__ == "__main__":
    main()

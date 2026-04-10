from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_root = repo_root / "src" / "ontology_reasoning"
    sys.path.insert(0, str(module_root))

    from main import main as ontology_main

    ontology_main()


if __name__ == "__main__":
    main()

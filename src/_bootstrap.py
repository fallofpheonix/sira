import sys
from pathlib import Path


def bootstrap() -> None:
    src_root = Path(__file__).resolve().parent
    project_root = src_root.parent

    for candidate in (project_root, src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
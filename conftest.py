"""Root conftest.py — ensures src/ is on sys.path so legacy imports work."""
import sys
from pathlib import Path

# Add src/ to path so legacy imports like `from generate_data import ...` work
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Also add repo root for `from src.core...` style imports
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

from src.symbolic.sindy import SINDy


def main():
    parser = argparse.ArgumentParser(description="Run SINDy baseline on vector-field dataset.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "data" / "processed" / "sir_vector_field.csv"),
    )
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    sindy = SINDy(threshold=args.threshold).fit(df)
    for eq, expr in sindy.get_equations().items():
        print(f"{eq} = {expr}")


if __name__ == "__main__":
    main()

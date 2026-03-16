import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd


def sindy_baseline(df, threshold=1e-3):
    """
    SINDy-style least-squares regression with polynomial library.
    Library: [S, I, R, S*I, S*R, I*R, S^2, I^2, R^2, 1]
    """
    S = df['S'].values
    I = df['I'].values
    R = df['R'].values

    Theta = np.column_stack([
        S, I, R,
        S*I, S*R, I*R,
        S**2, I**2, R**2,
        np.ones(len(S))
    ])

    feature_names = ['S', 'I', 'R', 'SI', 'SR', 'IR', 'S^2', 'I^2', 'R^2', '1']
    results = {}
    for target_col in ['dS_dt', 'dI_dt', 'dR_dt']:
        dy = df[target_col].values
        xi, _, _, _ = np.linalg.lstsq(Theta, dy, rcond=None)
        terms = [(name, coef) for name, coef in zip(feature_names, xi) if abs(coef) > threshold]
        results[target_col] = terms
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SINDy baseline on vector-field dataset.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "data" / "processed" / "sir_vector_field.csv"),
    )
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    results = sindy_baseline(df, threshold=args.threshold)
    for eq, terms in results.items():
        expr = " + ".join(f"{c:.4f}*{n}" for n, c in terms)
        print(f"{eq} = {expr}")


if __name__ == "__main__":
    main()

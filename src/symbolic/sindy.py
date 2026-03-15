import numpy as np
import pandas as pd
from pathlib import Path


class SINDy:
    def __init__(self, feature_names=None, threshold=1e-3):
        self.threshold = threshold
        self.feature_names = feature_names
        self.results_ = {}

    def fit(self, df):
        S = df['S'].values
        I = df['I'].values
        R = df['R'].values
        Theta = np.column_stack([
            S, I, R, S * I, S * R, I * R,
            S ** 2, I ** 2, R ** 2, np.ones(len(S))
        ])
        self.feature_names = self.feature_names or [
            'S', 'I', 'R', 'SI', 'SR', 'IR', 'S^2', 'I^2', 'R^2', '1']
        for target_col in ['dS_dt', 'dI_dt', 'dR_dt']:
            dy = df[target_col].values
            xi, _, _, _ = np.linalg.lstsq(Theta, dy, rcond=None)
            self.results_[target_col] = [
                (name, coef) for name, coef in zip(self.feature_names, xi)
                if abs(coef) > self.threshold
            ]
        return self

    def get_equations(self):
        return {eq: " + ".join(f"{c:.4f}*{n}" for n, c in terms)
                for eq, terms in self.results_.items()}

    def to_latex(self):
        lines = []
        for eq, terms in self.results_.items():
            expr = " + ".join(f"{c:.4f} {n}" for n, c in terms)
            # Extract variable name: e.g. 'dS_dt' -> 'S'
            var = eq.split('_')[0].lstrip('d') if '_' in eq else eq
            lines.append(f"\\frac{{d{var}}}{{dt}} = {expr}")
        return "\n".join(lines)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqs = self.get_equations()
        with open(path, 'w') as f:
            for eq, expr in eqs.items():
                f.write(f"{eq} = {expr}\n")

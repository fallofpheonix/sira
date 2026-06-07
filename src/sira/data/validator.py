import numpy as np
import pandas as pd


class QualityValidator:
    def __init__(self, conservation_tol=0.05):
        self.conservation_tol = conservation_tol

    def validate(self, df):
        issues = []
        if not self.check_nan(df):
            issues.append("NaN values detected")
        if not self.check_bounds(df):
            issues.append("Values out of bounds [0,1]")
        if not self.check_conservation(df):
            issues.append("S+I+R conservation violated")
        return len(issues) == 0, issues

    def check_nan(self, df):
        required = ['S', 'I', 'R', 'dS_dt', 'dI_dt', 'dR_dt']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"check_nan: required columns missing from DataFrame: {missing}"
            )
        return not df[required].isnull().any().any()

    def check_bounds(self, df):
        for col in ['S', 'I', 'R']:
            if col in df.columns:
                if (df[col] < -0.01).any() or (df[col] > 1.01).any():
                    return False
        return True

    def check_conservation(self, df):
        total = df['S'] + df['I'] + df['R']
        return bool((total - 1.0).abs().max() < self.conservation_tol)

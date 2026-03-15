# Constraints: SIRA

## Dataset Constraints
The dataset is **synthetically generated** — no external download is required. The `generate_data.py` script produces training data by Monte Carlo simulation of the stochastic SIR model with random `(β, γ)` samples.

Current script defaults:
- `num_samples=1000`
- `β ∈ [0.1, 0.5]` (uniform)
- `γ ∈ [0.05, 0.2]` (uniform)
- `num_points=100`

Output:
- `data/processed/sir_vector_field.csv` (vector field dataset)

## Software Constraints
- Python 3.10+
- PyTorch or TensorFlow for ML
- `torchdiffeq` if Neural ODE approach is used (optional)
- `PySR` for symbolic regression (optional)
- NumPy, SciPy for ODE solver and statistics

## Computational Constraints
- Dataset generation is CPU-bound (Monte Carlo)
- Training is lightweight — single GPU or CPU sufficient
- Expected training time: < 1 hour

## Reproducibility Constraints
- Deterministic runs require fixed seeds for NumPy and PyTorch.

## Evaluation Constraint
Final model must demonstrate:
- Vector-field MSE below a baseline numerical ODE solver threshold
- Symbolic regression must produce interpretable expressions

# Architecture & Roadmap: SIRA

## Phase 1 — Data Generation (Weeks 1–2)
- Implement Gillespie stochastic SIR simulator
- Sample `(β, γ)` parameter space
- Save per-sample trajectories as training dataset

Output: `data/sir_dataset.json`
Actual: `data/processed/sir_vector_field.csv`

## Phase 2 — ML Model Training (Weeks 3–5)
- Design MLP or Neural ODE architecture
- Train on `{β, γ} → {S(t), I(t), R(t)}` regression task
- Evaluate RMSE on held-out parameter pairs

Output: `models/sir_model.pth`
Actual: `models/vector_field_mlp.pth`

## Phase 3 — Symbolic Regression (Weeks 6–8)
- Extract residuals from ML predictions
- Apply PySR symbolic regression to approximate ODE terms
- Compare recovered expression with true ODE

Output: `results/symbolic_expression.txt`

## Phase 4 — Evaluation & Visualization (Weeks 9–10)
- Plot predicted vs. true trajectories for sampled parameter points
- Compute RMSE across full test set
- Visualize symbolic recovery accuracy

Output: `results/vector_field_parity.png`, `results/evaluation_report.md`

## Phase 5 — Packaging (Weeks 11–12)
- Finalize README and documentation
- Validate the lightweight submission demo pipeline
- Assemble the submission bundle with run instructions and checklist

Output: `submission/`

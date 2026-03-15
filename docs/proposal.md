# GSoC 2026 Proposal: SIRA
## *ML-Driven SIR Epidemic Model Discovery via Vector Field Learning*

---

## 1. Contact Information

| Field | Value |
|-------|-------|
| **Student Name** | Ujjwal Singh |
| **Email** | ujjosing@gmail.com |
| **GitHub** | [github.com/fallofpheonix](https://github.com/fallofpheonix) |
| **Time Zone** | IST (UTC+5:30) |
| **Mentor(s)** | Harrison Prosper, Olivia Prosper, Sergei Gleyzer |

---

## 2. Synopsis

Epidemic modeling relies on the Susceptible-Infected-Removed (SIR) system of ODEs, but fitting these equations to real-world outbreak data in real time remains a challenge. This project addresses the problem using modern machine learning for **system identification**: given stochastic SIR simulation trajectories, learn the governing vector field `F(S,I,R) = (dS/dt, dI/dt, dR/dt)` using a neural network, then apply symbolic regression (SINDy / PySR) to recover the interpretable ODE functional form.

The proposed approach is scientifically grounded in modern **Neural ODE discovery** methods. Unlike prior work that performs parameter-to-trajectory regression, this pipeline performs **state-space vector field regression** — recovering the equations of motion directly from data.

---

## 3. Benefits to Community

- Delivers an **openly available, reproducible ML tool** for rapid epidemic parameter estimation.
- Demonstrates applicability to COVID-19, Ebola, Mpox, and other SIR-governed outbreaks.
- The symbolic recovery output is interpretable by epidemiologists — not just ML researchers.
- Advances the intersection of **physics-informed ML and public health informatics**.

---

## 4. Technical Approach

### Problem Formulation

The learning target is the **vector field**, not raw trajectories:

```
Input:  (S, I, R)              ← normalized state at time t
Output: (dS/dt, dI/dt, dR/dt)  ← time derivatives
```

This allows symbolic regression to extract:
```
dS/dt ≈ -β * S * I
dI/dt ≈  β * S * I - γ * I
dR/dt ≈  γ * I
```

### Pipeline

```
Gillespie stochastic simulator (20 runs/parameter)
        ↓
Ensemble mean: E[S(t)], E[I(t)], E[R(t)]
        ↓
Finite-difference derivative estimation:
    dS/dt = np.gradient(S, t)
        ↓
Dataset: (S, I, R) → (dS_dt, dI_dt, dR_dt)
        ↓
MLP vector field learner (Tanh activations)
        ↓
SINDy sparse regression baseline
        ↓
Optional: PySR symbolic regression
        ↓
Recovered ODE + coefficient error
```

### ML Model

```python
class VectorFieldMLP(nn.Module):
    # Input: (S, I, R) — Output: (dS/dt, dI/dt, dR/dt)
    # 3 Tanh hidden layers, 128 units each
```

### Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Vector field MSE | `‖F_pred(X) − F_true(X)‖²` |
| Coefficient error | `|β_pred − β|`, `|γ_pred − γ|` |
| Trajectory RMSE | Simulation from recovered ODE vs. ground truth |

### Baselines Compared

1. Numerical ODE solver (SciPy `solve_ivp`)
2. Direct parameter fitting (curve_fit)
3. SINDy sparse regression
4. Neural vector field MLP ← this project

---

## 5. Deliverables

| # | Deliverable | Required/Optional |
|---|------------|-------------------|
| 1 | Gillespie simulator with ensemble averaging | Required |
| 2 | Derivative dataset generator | Required |
| 3 | Vector field MLP trainer | Required |
| 4 | SINDy baseline comparison | Required |
| 5 | PySR symbolic expression recovery | Optional |
| 6 | Evaluation notebook with all baselines | Required |
| 7 | Reproducible README and requirements | Required |

---

## 6. Timeline (175 hours)

| Period | Activity |
|--------|----------|
| **Pre-bonding** | Study SINDy, NeuralODE literature, set up environment |
| **Weeks 1–2** | Implement ensemble Gillespie simulator |
| **Weeks 3–4** | Build derivative dataset + data quality analysis |
| **Weeks 5–7** | Train and evaluate MLP vector field learner |
| **Weeks 8–9** | Implement SINDy baseline, compare all methods |
| **Week 10** | Optional: PySR symbolic regression step |
| **Weeks 11–12** | Evaluation notebook, documentation, final PR |

---

## 7. Related Work

- **SINDy** (Brunton et al., 2016): sparse regression for dynamics discovery — our baseline.
- **Neural ODEs** (Chen et al., 2018): learns continuous dynamics, used as comparison.
- **PySR** (Cranmer et al., 2020): genetic programming for symbolic regression.

This project differs by targeting epidemic-specific vector fields and providing a full reproducible benchmarking pipeline.

---

## 8. About Me

**Ujjwal Singh** | ujjosing@gmail.com | [GitHub](https://github.com) | [LinkedIn](https://linkedin.com) | VIT University, B.Tech CS (2023–2027) | IST (UTC+5:30)

I am a Computer Science undergraduate at VIT University with a focus on machine learning and applied scientific computing. My relevant background includes:

- **ML Experience:** Built a CNN-based plant disease detection system (*TerraHerb*) using TensorFlow/Keras with transfer learning and a Flask REST API — directly relevant to this project's neural ODE and regression training pipeline.
- **Data Analytics:** Virtual internship at Deloitte (Forage 2026) — structured dataset analysis, EDA, and dashboard design.
- **Cloud Architecture:** AWS Solutions Architecture Virtual Experience (Forage 2026) — distributed systems and scalability trade-offs.
- **Technical Skills:** Python, TensorFlow, Keras, NumPy, Pandas, Flask, FastAPI, Git, C++, Java.
- **Certifications:** AWS Technical Essentials, AI and Machine Learning Fundamentals.

I am familiar with scientific Python (NumPy, SciPy), eager to deepen expertise in dynamical systems and symbolic regression, and committed to producing reproducible, well-documented research code throughout the GSoC period.

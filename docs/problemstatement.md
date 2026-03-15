# Problem Statement: SIRA

## Problem Statement

During epidemic events like COVID-19, real-time estimation of epidemiological model parameters is critical for forecasting and policy decisions. The classic Susceptible-Infected-Removed (SIR) model is defined by a system of Ordinary Differential Equations (ODEs):

```
dS/dt = -β * S * I / N
dI/dt  =  β * S * I / N - γ * I
dR/dt  =  γ * I
```

Where:

```
S = susceptible population
I = infected population
R = removed population
N = total population
β = transmission rate
γ = recovery rate
```

The deterministic ODE system is powerful but hard to invert from noisy real-world observations.

The challenge is:

> **Given stochastically simulated epidemic trajectories, learn the governing vector field `F(S, I, R) = (dS/dt, dI/dt, dR/dt)` and recover the underlying parameters.**

This is formulated as a supervised regression task:

```
f: {S, I, R} → {dS/dt, dI/dt, dR/dt}
```

Or using symbolic regression to approximate:

```
dS/dt ≈ g_θ(S, I, R)
```

---

## Key Technical Challenges

### Stochastic Noise
Stochastic SIR simulations are inherently noisy — no two runs at the same parameter point are identical. The ML model must learn ensemble-averaged behavior from individual samples.

### Parameter Generalization
The model must extrapolate to unseen `(β, γ)` pairs and not merely memorize training trajectories.

### Symbolic Discovery
The stretch goal is to recover the functional form of the ODE system from data using auto-differentiation or symbolic regression — this requires the learned function to be interpretable.

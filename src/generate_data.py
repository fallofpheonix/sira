"""
SIRA Data Generator — Vector Field Learning

Generates ensemble-averaged stochastic SIR trajectories and computes
finite-difference derivative estimates for vector field learning.

Dataset schema:
  S, I, R             : mean trajectory values
  dS_dt, dI_dt, dR_dt : estimated vector field (dX/dt = F(X))
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from simulator import SIRSimulator
from tqdm import tqdm

def estimate_derivatives(t, S, I, R):
    """
    Estimate dS/dt, dI/dt, dR/dt via numpy finite differences.
    """
    dS = np.gradient(S, t)
    dI = np.gradient(I, t)
    dR = np.gradient(R, t)
    return dS, dI, dR

def generate_dataset(
    output_path,
    num_param_points=500,
    num_runs_per_param=20,
    population=1000,
    num_timepoints=100,
    beta_min=0.1,
    beta_max=0.9,
    gamma_min=0.02,
    gamma_max=0.4,
    seed=42,
):
    np.random.seed(seed)
    betas = np.random.uniform(beta_min, beta_max, num_param_points)
    gammas = np.random.uniform(gamma_min, gamma_max, num_param_points)

    records = []

    print(f"Generating {num_param_points} parameter points × {num_runs_per_param} runs each...")

    for idx in tqdm(range(num_param_points)):
        beta  = betas[idx]
        gamma = gammas[idx]

        # --- Ensemble averaging ---
        S_runs, I_runs, R_runs, t_grid = [], [], [], None

        for _ in range(num_runs_per_param):
            sim = SIRSimulator(population, beta, gamma)
            S0, I0, R0 = population - 5, 5, 0
            t, s, i, r = sim.simulate_gillespie(S0, I0, R0, max_time=150)
            t_u, s_u, i_u, r_u = sim.interpolate_simulation(
                t, s, i, r, num_points=num_timepoints
            )

            # Normalize
            S_runs.append(s_u / population)
            I_runs.append(i_u / population)
            R_runs.append(r_u / population)
            t_grid = t_u

        # Mean trajectories
        S_mean = np.mean(S_runs, axis=0)
        I_mean = np.mean(I_runs, axis=0)
        R_mean = np.mean(R_runs, axis=0)

        # Finite-difference vector field estimation
        dS_dt, dI_dt, dR_dt = estimate_derivatives(t_grid, S_mean, I_mean, R_mean)

        # Build flat per-timepoint records (correct ML input format)
        for k in range(num_timepoints):
            records.append({
                'beta':  beta,
                'gamma': gamma,
                't':     t_grid[k],
                'S':     S_mean[k],
                'I':     I_mean[k],
                'R':     R_mean[k],
                'dS_dt': dS_dt[k],
                'dI_dt': dI_dt[k],
                'dR_dt': dR_dt[k],
            })

    df = pd.DataFrame(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved: {output_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(df.head(3))

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate vector-field SIR dataset.")
    parser.add_argument(
        "--output-path",
        default=str(repo_root / "data" / "processed" / "sir_vector_field.csv"),
    )
    parser.add_argument("--num-param-points", type=int, default=500)
    parser.add_argument("--num-runs-per-param", type=int, default=20)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--num-timepoints", type=int, default=100)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=0.9)
    parser.add_argument("--gamma-min", type=float, default=0.02)
    parser.add_argument("--gamma-max", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(
        output_path=args.output_path,
        num_param_points=args.num_param_points,
        num_runs_per_param=args.num_runs_per_param,
        population=args.population,
        num_timepoints=args.num_timepoints,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        seed=args.seed,
    )

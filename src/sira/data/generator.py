import numpy as np
import pandas as pd
from pathlib import Path
from src.core.simulator import SIRSimulator


class RawSimulationGenerator:
    def __init__(self, population=1000, beta_min=0.1, beta_max=0.9,
                 gamma_min=0.02, gamma_max=0.4, seed=42):
        self.population = population
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.seed = seed

    def generate(self, num_param_points, num_runs_per_param, num_timepoints=100, max_time=150):
        np.random.seed(self.seed)
        betas = np.random.uniform(self.beta_min, self.beta_max, num_param_points)
        gammas = np.random.uniform(self.gamma_min, self.gamma_max, num_param_points)
        results = []
        for idx in range(num_param_points):
            runs = []
            for _ in range(num_runs_per_param):
                sim = SIRSimulator(self.population, betas[idx], gammas[idx])
                t, s, i, r = sim.simulate_gillespie(
                    self.population - 5, 5, 0, max_time=max_time)
                t_u, s_u, i_u, r_u = sim.interpolate_simulation(
                    t, s, i, r, num_points=num_timepoints)
                runs.append((t_u, s_u / self.population,
                              i_u / self.population, r_u / self.population))
            results.append({'beta': betas[idx], 'gamma': gammas[idx], 'runs': runs})
        return results


class EnsembleAverager:
    def average(self, runs):
        t_grids = [r[0] for r in runs]
        S_runs = [r[1] for r in runs]
        I_runs = [r[2] for r in runs]
        R_runs = [r[3] for r in runs]
        t_grid = t_grids[0]
        S_mean = np.mean(S_runs, axis=0)
        I_mean = np.mean(I_runs, axis=0)
        R_mean = np.mean(R_runs, axis=0)
        return S_mean, I_mean, R_mean, t_grid


class DerivativeEstimator:
    def estimate(self, t, S, I, R):
        dS = np.gradient(S, t)
        dI = np.gradient(I, t)
        dR = np.gradient(R, t)
        return dS, dI, dR


class DataPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.generator = RawSimulationGenerator(
            population=self.config.get('population', 1000),
            beta_min=self.config.get('beta_min', 0.1),
            beta_max=self.config.get('beta_max', 0.9),
            gamma_min=self.config.get('gamma_min', 0.02),
            gamma_max=self.config.get('gamma_max', 0.4),
            seed=self.config.get('seed', 42),
        )
        self.averager = EnsembleAverager()
        self.estimator = DerivativeEstimator()

    def run(self, output_path, num_param_points=None, num_runs_per_param=None,
            num_timepoints=None, max_time=None):
        num_param_points = num_param_points or self.config.get('num_param_points', 500)
        num_runs_per_param = num_runs_per_param or self.config.get('num_runs_per_param', 20)
        num_timepoints = num_timepoints or self.config.get('num_timepoints', 100)
        max_time = max_time or self.config.get('max_time', 150)

        raw = self.generator.generate(
            num_param_points, num_runs_per_param, num_timepoints, max_time)
        records = []
        for entry in raw:
            S_mean, I_mean, R_mean, t_grid = self.averager.average(entry['runs'])
            dS, dI, dR = self.estimator.estimate(t_grid, S_mean, I_mean, R_mean)
            for k in range(len(t_grid)):
                records.append({
                    'beta': entry['beta'], 'gamma': entry['gamma'], 't': t_grid[k],
                    'S': S_mean[k], 'I': I_mean[k], 'R': R_mean[k],
                    'dS_dt': dS[k], 'dI_dt': dI[k], 'dR_dt': dR[k],
                })
        df = pd.DataFrame(records)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

from multiprocessing import Pool
from functools import partial
from src.core.simulator import SIRSimulator
import numpy as np


def run_single_simulation(params, population, num_timepoints, max_time):
    beta, gamma = params
    sim = SIRSimulator(population, beta, gamma)
    S0, I0, R0 = population - 5, 5, 0
    t, s, i, r = sim.simulate_gillespie(S0, I0, R0, max_time=max_time)
    t_u, s_u, i_u, r_u = sim.interpolate_simulation(t, s, i, r, num_points=num_timepoints)
    return (t_u, s_u, i_u, r_u)


def run_parallel_simulations(param_list, population, num_timepoints, max_time, num_workers=4):
    fn = partial(run_single_simulation, population=population,
                 num_timepoints=num_timepoints, max_time=max_time)
    with Pool(processes=num_workers) as pool:
        results = pool.map(fn, param_list)
    return results

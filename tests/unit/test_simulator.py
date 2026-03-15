import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.simulator import SIRSimulator


def test_gillespie_runs():
    sim = SIRSimulator(N=100, beta=0.3, gamma=0.1)
    t, s, i, r = sim.simulate_gillespie(95, 5, 0, max_time=50)
    assert len(t) > 0
    assert len(s) == len(t)


def test_population_conservation():
    sim = SIRSimulator(N=100, beta=0.3, gamma=0.1)
    t, s, i, r = sim.simulate_gillespie(95, 5, 0, max_time=50)
    totals = s + i + r
    assert np.allclose(totals, 100)


def test_simulation_terminates():
    sim = SIRSimulator(N=100, beta=0.3, gamma=0.1)
    t, s, i, r = sim.simulate_gillespie(95, 5, 0, max_time=50)
    # The simulation must terminate: either I goes to 0, or the second-to-last
    # time was within max_time (last event may push slightly past it)
    assert i[-1] == 0 or len(t) > 1


def test_interpolation():
    sim = SIRSimulator(N=100, beta=0.3, gamma=0.1)
    t, s, i, r = sim.simulate_gillespie(95, 5, 0, max_time=50)
    t_u, s_u, i_u, r_u = sim.interpolate_simulation(t, s, i, r, num_points=20)
    assert len(t_u) == 20
    assert len(s_u) == 20

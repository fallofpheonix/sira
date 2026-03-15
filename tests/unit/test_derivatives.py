import numpy as np
import pytest

from src.data.generator import DerivativeEstimator


def test_derivative_linear():
    t = np.linspace(0, 1, 100)
    S = 2 * t
    I = t
    R = 0.5 * t
    estimator = DerivativeEstimator()
    dS, dI, dR = estimator.estimate(t, S, I, R)
    assert np.allclose(dS, 2.0, atol=0.01)
    assert np.allclose(dI, 1.0, atol=0.01)
    assert np.allclose(dR, 0.5, atol=0.01)


def test_derivative_constant():
    t = np.linspace(0, 1, 50)
    S = np.ones(50)
    I = np.ones(50) * 0.5
    R = np.zeros(50)
    estimator = DerivativeEstimator()
    dS, dI, dR = estimator.estimate(t, S, I, R)
    assert np.allclose(dS, 0.0, atol=1e-10)
    assert np.allclose(dI, 0.0, atol=1e-10)
    assert np.allclose(dR, 0.0, atol=1e-10)

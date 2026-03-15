# SIRA: Epidemic Simulator and Parameter Learning

## Problem Statement
Deducing deterministic epidemic parameters (SIR model) from stochastic simulation data. SIRA uses machine learning to learn the vector field of an epidemic directly from noisy synthetic data.

## System Architecture
An end-to-end ML pipeline:
1. Stochastic SIR simulation to generate training data.
2. Training a Multi-Layer Perceptron (MLP) to learn the SIR vector field.
3. Symbolic approximation of the recovered ODE terms.

## Key Modules
- `src/simulator.py`: Stochastic SIR model simulator.
- `src/train_ml.py`: Pytorch training loop for vector field learning.
- `src/symbolic_solver.py`: Tools for approximating ODE terms.

## Technology Stack
- **Language**: Python 3.x
- **ML Framework**: PyTorch
- **Libraries**: NumPy, Pandas, Matplotlib

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate data and train:
   ```bash
   python src/generate_data.py && python src/train_ml.py
   ```
3. Visualize results:
   ```bash
   python src/visualize_results.py
   ```

## Inputs and Outputs
- **Inputs**: Parameter ranges for Susceptible, Infected, and Removed populations.
- **Outputs**: Processed CSV datasets, trained `.pth` models, and parity plots comparing predicted vs. real vector fields.

---
*GSoC 2026 Project - Human AI*

# Architecture / Design

## High-level Flow

1. `scripts/run_experiment.py` orchestrates the workflow.
2. `sira.services.dataset_service` generates vector-field training data.
3. `sira.services.training_service` trains and checkpoints the model.
4. `src.symbolic.sindy` derives symbolic expressions from generated data.
5. `sira.api.app` serves inference endpoints.

## Module Responsibilities

- `src/sira/core/`
  - shared project paths and runtime primitives.
- `src/sira/config/`
  - YAML configuration loading.
- `src/sira/services/`
  - business orchestration logic for dataset, training, reporting, and experiment runs.
- `src/sira/api/`
  - FastAPI app factory, routes, and schema validation.
- `src/sira/utils/`
  - cross-cutting helpers (e.g., deterministic seeding).
- `src/` (legacy domain modules)
  - simulator, model architectures, symbolic discovery, trainer internals.

## Design Choices

- Keep orchestration in thin services while reusing mature domain modules.
- Keep API state explicit via lifespan-managed service objects.
- Preserve CLI entrypoints (`src/generate_data.py`, `src/train_ml.py`, `src/visualize_results.py`) for backward compatibility.

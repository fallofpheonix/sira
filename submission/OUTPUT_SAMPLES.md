# Output Samples

Artifacts included in `output_samples/`:

- `symbolic_expression.txt` — symbolic regression output from a full run.
- `vector_field_parity.png` — parity plot from model predictions.
- `submission_demo_symbolic_expression.txt` — symbolic output from submission demo config.

If a file is absent, regenerate via:

```bash
cd source_code
source .venv/bin/activate
python scripts/run_experiment.py --config config/base.yaml --data-config config/data.yaml --model-config config/model.yaml
```

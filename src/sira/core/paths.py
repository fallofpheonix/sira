from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "sir_vector_field.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "vector_field_mlp.pth"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "vector_field_parity.png"
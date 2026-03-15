from src.models.architectures.mlp import VectorFieldMLP
from src.models.architectures.neural_ode import NeuralODEBaseline
from src.models.architectures.physics_informed import PhysicsInformedMLP

MODEL_REGISTRY = {
    'VectorFieldMLP': VectorFieldMLP,
    'NeuralODEBaseline': NeuralODEBaseline,
    'PhysicsInformedMLP': PhysicsInformedMLP,
}


def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    return list(MODEL_REGISTRY.keys())

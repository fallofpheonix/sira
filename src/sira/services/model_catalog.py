from src.models.registry import list_models as list_registered_models


def list_available_models() -> list[str]:
    return list_registered_models()
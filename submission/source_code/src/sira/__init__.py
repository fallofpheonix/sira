"""SIRA application package."""

def create_app():
	from sira.api.app import create_app as app_factory

	return app_factory()

__all__ = ["create_app"]
"""
Entrypoint shim for running uvicorn directly with 'main:app'.
Re-exports the FastAPI app instance from app.main.
"""

from app.main import app

__all__ = ["app"]

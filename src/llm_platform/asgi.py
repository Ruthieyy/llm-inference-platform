"""Uvicorn entrypoint: ``uvicorn llm_platform.asgi:app``."""

from llm_platform.app import create_app

app = create_app()

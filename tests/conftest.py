"""Fixtures partagées par les tests."""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from fire_safety_rag.storage.database import connect


@pytest.fixture
def connection() -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
    """Connexion à la vraie base ; le test est sauté si elle est injoignable (cas de la CI)."""
    try:
        conn = connect()
    except (ValueError, psycopg.OperationalError) as exc:
        pytest.skip(f"Base PostgreSQL injoignable : {exc}")
    with conn:
        yield conn

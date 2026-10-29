"""Tests unitaires et d'intégration pour le module database."""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from fire_safety_rag.config import Settings
from fire_safety_rag.storage.database import connect


def test_connect_sans_mot_de_passe_leve_une_erreur() -> None:
    """Sans POSTGRES_PASSWORD, connect() échoue clairement, sans contacter la base."""
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        connect(Settings(postgres_password=None))


@pytest.fixture
def connection() -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
    """Connexion à la vraie base ; le test est sauté si elle est injoignable (cas de la CI)."""
    try:
        conn = connect()
    except (ValueError, psycopg.OperationalError) as exc:
        pytest.skip(f"Base PostgreSQL injoignable : {exc}")
    with conn:
        yield conn


def test_extension_vector_installee(connection: psycopg.Connection[tuple[Any, ...]]) -> None:
    """L'extension pgvector est installée dans la base."""
    row = connection.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None


def test_aller_retour_d_un_vecteur(connection: psycopg.Connection[tuple[Any, ...]]) -> None:
    """Un vecteur Python envoyé à PostgreSQL revient identique (objet pgvector Vector)."""
    row = connection.execute("SELECT %s::vector", ([0.5, -1.0, 2.0],)).fetchone()
    assert row is not None
    assert row[0].to_list() == [0.5, -1.0, 2.0]

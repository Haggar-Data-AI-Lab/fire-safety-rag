"""Connexion à la base PostgreSQL + pgvector."""

from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from fire_safety_rag.config import Settings, get_settings


def connect(settings: Settings | None = None) -> psycopg.Connection[tuple[Any, ...]]:
    """Ouvre une connexion à la base et y déclare le type `vector` de pgvector.

    Les paramètres sont passés séparément à psycopg (et non dans une URL) :
    le mot de passe n'apparaît ainsi dans aucune chaîne de connexion.

    Args:
        settings: configuration à utiliser ; par défaut, celle de l'application.

    Returns:
        Connexion ouverte, à fermer par l'appelant (idéalement avec `with`).

    Raises:
        ValueError: si POSTGRES_PASSWORD n'est pas défini.
        psycopg.OperationalError: si la base est injoignable ou refuse
            l'authentification.
    """
    settings = settings or get_settings()
    if settings.postgres_password is None:
        raise ValueError("POSTGRES_PASSWORD n'est pas défini (voir .env.example)")

    connection = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        connect_timeout=5,
    )
    # Permet d'envoyer et de recevoir des vecteurs Python <-> colonnes `vector`.
    # Échoue si l'extension n'est pas installée (voir db/init/01-extensions.sql).
    register_vector(connection)
    return connection

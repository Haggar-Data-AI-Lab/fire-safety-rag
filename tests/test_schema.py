"""Tests d'intégration du schéma (sautés si la base est injoignable).

La fixture `connection` est définie dans conftest.py.
"""

from datetime import date
from typing import Any

import psycopg

from fire_safety_rag.storage.schema import EMBEDDING_DIMENSION, init_schema

Connection = psycopg.Connection[tuple[Any, ...]]


def test_init_schema_idempotente(connection: Connection) -> None:
    """init_schema peut être appelée deux fois de suite sans erreur."""
    init_schema(connection)
    init_schema(connection)


def test_colonnes_attendues(connection: Connection) -> None:
    """Les tables documents et chunks ont les colonnes prévues."""
    init_schema(connection)
    rows = connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
        " WHERE table_name IN ('documents', 'chunks')"
    ).fetchall()
    columns = {(table, column) for table, column in rows}

    assert ("documents", "content_hash") in columns
    assert ("documents", "embedding_model") in columns
    assert ("chunks", "position") in columns
    assert ("chunks", "history") in columns
    assert ("chunks", "embedding") in columns


def test_colonne_embedding_en_768_dimensions(connection: Connection) -> None:
    """La colonne embedding est de type vector(768)."""
    init_schema(connection)
    row = connection.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute"
        " WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()

    assert row == (f"vector({EMBEDDING_DIMENSION})",)


def test_suppression_document_supprime_ses_chunks(connection: Connection) -> None:
    """ON DELETE CASCADE : supprimer un document supprime ses chunks.

    Tout se passe dans une transaction annulée à la fin : rien ne reste en base.
    """
    init_schema(connection)
    with connection.transaction(force_rollback=True):
        connection.execute(
            "INSERT INTO documents (document_id, content_hash, embedding_model)"
            " VALUES ('doc-test', 'hash', 'modele')"
        )
        connection.execute(
            "INSERT INTO chunks (document_id, position, article_id, chunk_index, version,"
            " valid_from, valid_until, history, text, page_start, page_end, embedding)"
            " VALUES ('doc-test', 0, 'R1', 0, 'VERSION', %s, NULL, %s, 'Texte.', 1, 1, %s)",
            (date(2020, 1, 1), ["Création Arrêté"], [0.0] * EMBEDDING_DIMENSION),
        )
        connection.execute("DELETE FROM documents WHERE document_id = 'doc-test'")
        row = connection.execute(
            "SELECT count(*) FROM chunks WHERE document_id = 'doc-test'"
        ).fetchone()

        assert row == (0,)

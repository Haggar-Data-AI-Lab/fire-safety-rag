"""Tests du schéma ; ceux qui utilisent la base sont sautés si elle est injoignable.

La fixture `connection` est définie dans conftest.py.
"""

from datetime import date
from typing import Any

import psycopg
import pytest

from fire_safety_rag.storage.schema import (
    EMBEDDING_DIMENSION,
    EXPECTED_COLUMNS,
    SchemaError,
    check_schema,
    init_schema,
    missing_columns,
)

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
    assert ("chunks", "section") in columns


def test_colonne_embedding_en_768_dimensions(connection: Connection) -> None:
    """La colonne embedding est de type vector(768)."""
    init_schema(connection)
    row = connection.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute"
        " WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()

    assert row == (f"vector({EMBEDDING_DIMENSION})",)


def test_search_vector_calcule_en_francais(connection: Connection) -> None:
    """search_vector est remplie automatiquement, avec les règles du français."""
    init_schema(connection)
    with connection.transaction(force_rollback=True):
        connection.execute(
            "INSERT INTO documents (document_id, content_hash, embedding_model)"
            " VALUES ('doc-test', 'hash', 'modele')"
        )
        connection.execute(
            "INSERT INTO chunks (document_id, position, article_id, chunk_index, version,"
            " valid_from, valid_until, history, text, page_start, page_end, embedding)"
            " VALUES ('doc-test', 0, 'R1', 0, 'VERSION', %s, NULL, %s,"
            " 'Les sorties de secours.', 1, 1, %s)",
            (date(2020, 1, 1), ["Création Arrêté"], [0.0] * EMBEDDING_DIMENSION),
        )
        row = connection.execute(
            "SELECT search_vector @@ to_tsquery('french', 'sortie'),"
            " search_vector @@ 'les'::tsquery"
            " FROM chunks WHERE document_id = 'doc-test'"
        ).fetchone()

        assert row == (True, False)


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


def test_missing_columns_schema_complet() -> None:
    """Toutes les colonnes attendues présentes : rien ne manque (sans base)."""
    existing = {(table, column) for table, cols in EXPECTED_COLUMNS.items() for column in cols}
    assert missing_columns(existing) == []


def test_missing_columns_signale_la_colonne_absente() -> None:
    """Une base d'avant la colonne section : seule chunks.section est signalée."""
    existing = {(table, column) for table, cols in EXPECTED_COLUMNS.items() for column in cols}
    existing.discard(("chunks", "section"))
    assert missing_columns(existing) == ["chunks.section"]


def test_check_schema_apres_init_schema(connection: Connection) -> None:
    """Après init_schema, le schéma de la vraie base est jugé complet."""
    init_schema(connection)
    check_schema(connection)


def test_check_schema_erreur_explicite() -> None:
    """Le message nomme les colonnes manquantes et la commande à lancer."""

    class FakeConnection:
        def execute(self, query: str, params: object) -> "FakeConnection":
            return self

        def fetchall(self) -> list[tuple[str, str]]:
            return [("documents", "document_id")]

    with pytest.raises(SchemaError, match=r"chunks\.section.*indexing\.indexer"):
        check_schema(FakeConnection())  # type: ignore[arg-type]

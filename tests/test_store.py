"""Tests de l'écriture et de la recherche (store).

Les tests avec base utilisent la fixture `connection` (conftest.py) et
s'exécutent dans une transaction annulée à la fin : rien ne reste en base.
Les vecteurs sont fabriqués à la main : on teste la mécanique SQL, pas la
qualité sémantique du modèle.
"""

from collections.abc import Iterator
from datetime import date
from typing import Any

import psycopg
import pytest

from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.schema import EMBEDDING_DIMENSION, init_schema
from fire_safety_rag.storage.store import get_indexed_document, replace_document, search

Connection = psycopg.Connection[tuple[Any, ...]]
MODEL = "modele-de-test"


def _chunk(position: int, valid_until: date | None = None) -> Chunk:
    """Construit un chunk de test distinct pour chaque position."""
    return Chunk(
        document_id="doc-test",
        article_id=f"R{position}",
        version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
        valid_from=date(2008, 5, 1),
        valid_until=valid_until,
        history=("Création Arrêté du 1er janvier 2008",),
        chunk_index=0,
        text=f"Texte du chunk {position}.",
        page_start=position + 1,
        page_end=position + 1,
    )


def _unit_vector(axis: int) -> list[float]:
    """Vecteur de norme 1 qui pointe le long d'un seul axe."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[axis] = 1.0
    return vector


@pytest.fixture
def db(connection: Connection) -> Iterator[Connection]:
    """Base avec schéma, dans une transaction annulée à la fin du test."""
    init_schema(connection)
    with connection.transaction(force_rollback=True):
        yield connection


class TestValidation:
    """Vérifications faites avant toute écriture (sans base)."""

    def test_nombre_de_vecteurs_incorrect(self) -> None:
        """Un nombre de vecteurs différent du nombre de chunks est refusé."""
        with pytest.raises(ValueError, match="2 chunks mais 1 vecteurs"):
            replace_document(
                None,  # type: ignore[arg-type]  # jamais utilisée : l'erreur arrive avant
                "doc-test",
                "hash",
                MODEL,
                [_chunk(0), _chunk(1)],
                [_unit_vector(0)],
            )

    def test_dimension_incorrecte(self) -> None:
        """Un vecteur qui n'a pas 768 dimensions est refusé."""
        with pytest.raises(ValueError, match="3 dimensions au lieu de 768"):
            replace_document(
                None,  # type: ignore[arg-type]  # jamais utilisée : l'erreur arrive avant
                "doc-test",
                "hash",
                MODEL,
                [_chunk(0)],
                [[1.0, 0.0, 0.0]],
            )


class TestAvecBase:
    """Écriture et recherche sur la vraie base (sautés si elle est injoignable)."""

    def test_reecriture_sans_doublon(self, db: Connection) -> None:
        """Réécrire un document remplace ses chunks au lieu de les ajouter."""
        chunks = [_chunk(position) for position in range(3)]
        vectors = [_unit_vector(position) for position in range(3)]

        replace_document(db, "doc-test", "hash-1", MODEL, chunks, vectors)
        replace_document(db, "doc-test", "hash-2", MODEL, chunks, vectors)

        row = db.execute("SELECT count(*) FROM chunks WHERE document_id = 'doc-test'").fetchone()
        assert row == (3,)

    def test_recherche_renvoie_le_plus_proche(self, db: Connection) -> None:
        """Le chunk dont le vecteur est identique à la question arrive en premier."""
        chunks = [_chunk(position) for position in range(3)]
        replace_document(db, "doc-test", "hash", MODEL, chunks, [_unit_vector(p) for p in range(3)])

        results = search(db, _unit_vector(2), MODEL, k=3)

        assert results[0].chunk == chunks[2]  # chunk reconstruit à l'identique
        assert results[0].score == pytest.approx(1.0)
        assert len(results) == 3

    def test_chunk_expire_jamais_renvoye(self, db: Connection) -> None:
        """Un chunk dont la date de fin de vigueur est passée n'est pas renvoyé."""
        chunks = [_chunk(0), _chunk(1, valid_until=date(2000, 1, 1))]
        replace_document(db, "doc-test", "hash", MODEL, chunks, [_unit_vector(0), _unit_vector(1)])

        results = search(db, _unit_vector(1), MODEL, k=5)

        assert [result.chunk.article_id for result in results] == ["R0"]

    def test_autre_modele_jamais_renvoye(self, db: Connection) -> None:
        """Les chunks vectorisés avec un autre modèle ne sont pas comparés."""
        replace_document(db, "doc-test", "hash", "autre-modele", [_chunk(0)], [_unit_vector(0)])

        assert search(db, _unit_vector(0), MODEL, k=5) == []

    def test_get_indexed_document(self, db: Connection) -> None:
        """get_indexed_document renvoie None, puis (empreinte, modèle) après écriture."""
        assert get_indexed_document(db, "doc-test") is None

        replace_document(db, "doc-test", "hash-1", MODEL, [_chunk(0)], [_unit_vector(0)])

        assert get_indexed_document(db, "doc-test") == ("hash-1", MODEL)

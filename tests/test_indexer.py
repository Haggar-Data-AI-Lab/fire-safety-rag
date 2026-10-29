"""Tests de l'indexation du corpus.

Les tests avec base s'exécutent dans une transaction annulée à la fin et
utilisent un faux embedder : on teste la logique d'indexation (saut,
réindexation), pas le modèle.
"""

from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest

from fire_safety_rag.indexing.indexer import file_sha256, index_corpus
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.schema import EMBEDDING_DIMENSION, init_schema

Connection = psycopg.Connection[tuple[Any, ...]]

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
CODE_TRAVAIL_ID = "code-travail-r4227-incendie"
CODE_TRAVAIL_PDF = DATA_RAW / f"{CODE_TRAVAIL_ID}.pdf"


def test_file_sha256(tmp_path: Path) -> None:
    """L'empreinte de "abc" est la valeur de référence publiée pour SHA-256."""
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def _fake_embed(chunks: Sequence[Chunk]) -> list[list[float]]:
    """Faux embedder : le même vecteur unitaire pour chaque chunk."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = 1.0
    return [vector for _ in chunks]


@pytest.mark.skipif(
    not CODE_TRAVAIL_PDF.is_file(),
    reason="Corpus absent : les PDF ne sont pas versionnés (voir README)",
)
class TestIndexCorpus:
    """Indexation du vrai PDF du Code du travail (sautés sans base ni PDF)."""

    @pytest.fixture
    def db(self, connection: Connection) -> Iterator[Connection]:
        """Base avec schéma, dans une transaction annulée à la fin du test."""
        init_schema(connection)
        with connection.transaction(force_rollback=True):
            yield connection

    @pytest.fixture
    def catalogue(self, tmp_path: Path) -> Path:
        """Catalogue limité au Code du travail."""
        path = tmp_path / "sources.csv"
        path.write_text(
            f"id,fichier\n{CODE_TRAVAIL_ID},{CODE_TRAVAIL_PDF.name}\n", encoding="utf-8"
        )
        return path

    def _index(self, db: Connection, catalogue: Path, model: str, force: bool = False) -> list[str]:
        """Lance index_corpus avec le faux embedder et une date fixe."""
        return index_corpus(
            db, catalogue, DATA_RAW, _fake_embed, model, date(2026, 10, 19), force=force
        )

    def test_premier_passage_indexe(self, db: Connection, catalogue: Path) -> None:
        """Le premier passage indexe les 57 chunks du document."""
        assert self._index(db, catalogue, "modele-a") == [CODE_TRAVAIL_ID]

        row = db.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s", (CODE_TRAVAIL_ID,)
        ).fetchone()
        assert row == (57,)

    def test_deuxieme_passage_saute(self, db: Connection, catalogue: Path) -> None:
        """Un document inchangé, avec le même modèle, est sauté."""
        self._index(db, catalogue, "modele-a")
        assert self._index(db, catalogue, "modele-a") == []

    def test_force_reindexe(self, db: Connection, catalogue: Path) -> None:
        """Avec force=True, un document inchangé est réindexé."""
        self._index(db, catalogue, "modele-a")
        assert self._index(db, catalogue, "modele-a", force=True) == [CODE_TRAVAIL_ID]

    def test_changement_de_modele_reindexe(self, db: Connection, catalogue: Path) -> None:
        """Changer de modèle d'embedding réindexe le document."""
        self._index(db, catalogue, "modele-a")
        assert self._index(db, catalogue, "modele-b") == [CODE_TRAVAIL_ID]

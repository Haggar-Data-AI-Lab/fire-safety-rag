"""Tests unitaires et d'intégration pour le module pipeline."""

import json
from datetime import date
from pathlib import Path

import pytest

from fire_safety_rag.ingestion.chunker import Article, Chunk
from fire_safety_rag.ingestion.pipeline import (
    chunk_corpus,
    is_in_force,
    load_catalogue,
    write_chunks_jsonl,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CODE_TRAVAIL_ID = "code-travail-r4227-incendie"
CODE_TRAVAIL_PDF = DATA_DIR / "raw" / f"{CODE_TRAVAIL_ID}.pdf"


def _article(valid_until: date | None) -> Article:
    """Construit un Article minimal avec la date de fin de vigueur donnée."""
    return Article(
        document_id="doc",
        article_id="GN 4",
        version="VERSION EN VIGUEUR ...",
        valid_from=date(1980, 8, 15),
        valid_until=valid_until,
        text="Texte.",
        page_start=1,
        page_end=1,
        line_pages=(1,),
    )


def test_load_catalogue(tmp_path: Path) -> None:
    """Le catalogue CSV est lu en une ligne (dictionnaire) par document."""
    catalogue = tmp_path / "sources.csv"
    catalogue.write_text(
        "id,fichier\ndoc-a,doc-a.pdf\ndoc-b,doc-b.pdf\n",
        encoding="utf-8",
    )
    assert load_catalogue(catalogue) == [
        {"id": "doc-a", "fichier": "doc-a.pdf"},
        {"id": "doc-b", "fichier": "doc-b.pdf"},
    ]


class TestIsInForce:
    """Tests unitaires de la fonction is_in_force."""

    def test_sans_date_de_fin(self) -> None:
        """Un article sans date de fin est en vigueur."""
        assert is_in_force(_article(None), date(2026, 10, 19))

    def test_date_de_fin_future(self) -> None:
        """Un article dont la fin est programmée plus tard est en vigueur."""
        assert is_in_force(_article(date(2027, 6, 1)), date(2026, 10, 19))

    def test_date_de_fin_passee(self) -> None:
        """Un article dont la date de fin est passée n'est plus en vigueur."""
        assert not is_in_force(_article(date(2000, 6, 21)), date(2026, 10, 19))

    def test_jour_de_fin(self) -> None:
        """Le jour de sa date de fin, l'article n'est plus en vigueur."""
        assert not is_in_force(_article(date(2027, 6, 1)), date(2027, 6, 1))


def test_write_chunks_jsonl(tmp_path: Path) -> None:
    """Chaque chunk est écrit sur une ligne JSON, accents et dates lisibles."""
    chunk = Chunk(
        document_id="doc",
        article_id="R4227-4",
        version="VERSION EN VIGUEUR DU 01/05/2008 AU 01/01/2027",
        valid_from=date(2008, 5, 1),
        valid_until=date(2027, 1, 1),
        chunk_index=0,
        text="Les dégagements sont toujours libres.",
        page_start=2,
        page_end=2,
    )
    output = tmp_path / "processed" / "chunks.jsonl"

    write_chunks_jsonl([chunk, chunk], output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "dégagements" in lines[0]  # accents non échappés (pas de é)
    record = json.loads(lines[0])
    assert record["valid_from"] == "2008-05-01"
    assert record["valid_until"] == "2027-01-01"
    assert record["text"] == chunk.text


@pytest.mark.skipif(
    not CODE_TRAVAIL_PDF.is_file(),
    reason="Corpus absent : les PDF ne sont pas versionnés (voir README)",
)
class TestChunkCorpusCodeTravail:
    """Tests d'intégration de chunk_corpus sur le vrai PDF du Code du travail."""

    @staticmethod
    def _catalogue(tmp_path: Path) -> Path:
        """Écrit un catalogue limité au Code du travail."""
        catalogue = tmp_path / "sources.csv"
        catalogue.write_text(
            f"id,fichier\n{CODE_TRAVAIL_ID},{CODE_TRAVAIL_PDF.name}\n",
            encoding="utf-8",
        )
        return catalogue

    def test_decoupe_en_chunks(self, tmp_path: Path) -> None:
        """À la date de téléchargement, les 56 articles donnent 57 chunks."""
        chunks = chunk_corpus(
            self._catalogue(tmp_path), CODE_TRAVAIL_PDF.parent, date(2026, 10, 19)
        )

        # Valeurs constatées le 2026-10-29 sur le PDF téléchargé le 2026-10-19
        assert len(chunks) == 57
        assert chunks[0].article_id == "R4227-1"
        assert all(chunk.document_id == CODE_TRAVAIL_ID for chunk in chunks)
        assert all(len(chunk.text) <= 1500 for chunk in chunks)

    def test_exclut_articles_plus_en_vigueur(self, tmp_path: Path) -> None:
        """Au 01/01/2027, les 2 articles dont la vigueur s'arrête ce jour sont exclus."""
        chunks = chunk_corpus(self._catalogue(tmp_path), CODE_TRAVAIL_PDF.parent, date(2027, 1, 1))

        assert len({chunk.article_id for chunk in chunks}) == 54
        assert all(
            chunk.valid_until is None or chunk.valid_until > date(2027, 1, 1) for chunk in chunks
        )

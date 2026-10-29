"""Tests unitaires et d'intégration pour le module pipeline."""

import json
from pathlib import Path

import pytest

from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.ingestion.pipeline import chunk_corpus, load_catalogue, write_chunks_jsonl

DATA_DIR = Path(__file__).parent.parent / "data"
CODE_TRAVAIL_ID = "code-travail-r4227-incendie"
CODE_TRAVAIL_PDF = DATA_DIR / "raw" / f"{CODE_TRAVAIL_ID}.pdf"


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


def test_write_chunks_jsonl(tmp_path: Path) -> None:
    """Chaque chunk est écrit sur une ligne JSON, accents conservés."""
    chunk = Chunk(
        document_id="doc",
        article_id="R4227-4",
        version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
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
    assert Chunk(**json.loads(lines[0])) == chunk


@pytest.mark.skipif(
    not CODE_TRAVAIL_PDF.is_file(),
    reason="Corpus absent : les PDF ne sont pas versionnés (voir README)",
)
def test_chunk_corpus_code_travail(tmp_path: Path) -> None:
    """Le pipeline découpe le vrai PDF du Code du travail en chunks."""
    catalogue = tmp_path / "sources.csv"
    catalogue.write_text(
        f"id,fichier\n{CODE_TRAVAIL_ID},{CODE_TRAVAIL_PDF.name}\n",
        encoding="utf-8",
    )

    chunks = chunk_corpus(catalogue, CODE_TRAVAIL_PDF.parent)

    # Valeur constatée le 2026-10-29 : 56 articles donnent 57 chunks
    assert len(chunks) == 57
    assert chunks[0].article_id == "R4227-1"
    assert all(chunk.document_id == CODE_TRAVAIL_ID for chunk in chunks)
    assert all(len(chunk.text) <= 1500 for chunk in chunks)

"""Indexation du corpus : chunks + vecteurs dans PostgreSQL + pgvector.

Pour chaque document du catalogue, on calcule l'empreinte SHA-256 du PDF. Si
la base contient déjà ce document avec la même empreinte et le même modèle
d'embedding, il est sauté (ni extraction ni vectorisation). Sinon, il est
découpé, vectorisé, puis remplacé en base en une transaction.

Limite : modifier le code du découpage sans changer le PDF ne change pas
l'empreinte. Utiliser alors --force pour tout réindexer.

Utilisation :
    python -m fire_safety_rag.indexing.indexer [--force]
"""

import argparse
import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import psycopg

from fire_safety_rag.config import get_settings
from fire_safety_rag.embedding.embedder import embed_passages
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.ingestion.pipeline import (
    DEFAULT_CATALOGUE,
    DEFAULT_RAW_DIR,
    chunk_document,
    load_catalogue,
)
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.schema import init_schema
from fire_safety_rag.storage.store import get_indexed_document, replace_document

logger = logging.getLogger(__name__)

# Fonction de vectorisation : embed_passages en usage réel, une fausse
# fonction dans les tests (pas besoin du modèle de 1 Go).
EmbedFunction = Callable[[Sequence[Chunk]], list[list[float]]]


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier, lu par blocs de 1 Mo.

    Args:
        path: chemin du fichier.
        block_size: taille des blocs lus, en octets.

    Returns:
        Empreinte hexadécimale (64 caractères).
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def index_corpus(
    connection: psycopg.Connection[tuple[Any, ...]],
    catalogue_path: Path,
    raw_dir: Path,
    embed: EmbedFunction,
    embedding_model: str,
    reference_date: date,
    force: bool = False,
) -> list[str]:
    """Indexe en base les documents du catalogue qui ont changé.

    Args:
        connection: connexion ouverte, avec le schéma créé (init_schema).
        catalogue_path: chemin du catalogue des sources (CSV).
        raw_dir: dossier contenant les fichiers PDF du corpus.
        embed: fonction qui vectorise une liste de chunks.
        embedding_model: nom du modèle utilisé par `embed`, enregistré en base.
        reference_date: date à laquelle on évalue la vigueur des articles.
        force: si True, réindexe même les documents inchangés.

    Returns:
        Identifiants des documents (ré)indexés ; les documents sautés n'y
        figurent pas.
    """
    indexed: list[str] = []
    for entry in load_catalogue(catalogue_path):
        document_id = entry["id"]
        pdf_path = raw_dir / entry["fichier"]
        content_hash = file_sha256(pdf_path)

        if not force and get_indexed_document(connection, document_id) == (
            content_hash,
            embedding_model,
        ):
            logger.info("%s : inchangé, sauté", document_id)
            continue

        start = time.perf_counter()
        chunks = chunk_document(pdf_path, document_id, reference_date)
        vectors = embed(chunks)
        replace_document(connection, document_id, content_hash, embedding_model, chunks, vectors)
        logger.info(
            "%s : %d chunks indexés en %.0f s",
            document_id,
            len(chunks),
            time.perf_counter() - start,
        )
        indexed.append(document_id)
    return indexed


def main() -> None:
    """Indexe le corpus avec les chemins par défaut et le modèle configuré."""
    parser = argparse.ArgumentParser(description="Indexe le corpus dans PostgreSQL + pgvector.")
    parser.add_argument(
        "--force", action="store_true", help="réindexe même les documents inchangés"
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    with connect(settings) as connection:
        init_schema(connection)
        indexed = index_corpus(
            connection,
            DEFAULT_CATALOGUE,
            DEFAULT_RAW_DIR,
            embed=embed_passages,
            embedding_model=settings.embedding_model,
            reference_date=date.today(),
            force=args.force,
        )
    logger.info("%d document(s) (ré)indexé(s)", len(indexed))


if __name__ == "__main__":
    main()

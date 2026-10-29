"""Écriture des chunks vectorisés et recherche des chunks les plus proches."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector import Vector

from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.schema import EMBEDDING_DIMENSION

# ---------------------------------------------------------------------------
# Requêtes SQL
# ---------------------------------------------------------------------------

_INSERT_DOCUMENT = (
    "INSERT INTO documents (document_id, content_hash, embedding_model) VALUES (%s, %s, %s)"
)

_INSERT_CHUNK = (
    "INSERT INTO chunks (document_id, position, article_id, chunk_index, version,"
    " valid_from, valid_until, history, text, page_start, page_end, embedding)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)

# <=> : distance cosinus de pgvector (0 = même direction, 2 = opposés).
# Le filtre de vigueur complète celui de l'ingestion : un article dont la date
# de fin arrive (ex. 01/06/2027) disparaît des résultats sans réindexation.
_SEARCH = """
SELECT c.document_id, c.article_id, c.version, c.valid_from, c.valid_until,
       c.history, c.chunk_index, c.text, c.page_start, c.page_end,
       c.embedding <=> %(query)s AS distance
FROM chunks AS c
JOIN documents AS d USING (document_id)
WHERE d.embedding_model = %(model)s
  AND (c.valid_until IS NULL OR c.valid_until > current_date)
ORDER BY distance
LIMIT %(k)s
"""


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Un chunk retrouvé par la recherche, avec son score.

    Attributes:
        chunk: chunk retrouvé, avec ses métadonnées (pages, dates, historique).
        score: similarité cosinus avec la question (1 - distance) ; plus il est
            élevé, plus le chunk est proche. Seul l'ordre est significatif.
    """

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def _check_vectors(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
    """Vérifie qu'il y a un vecteur par chunk, chacun de la bonne dimension.

    Raises:
        ValueError: si le nombre ou la dimension des vecteurs est incorrect.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks mais {len(vectors)} vecteurs")
    for position, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Vecteur {position} : {len(vector)} dimensions au lieu de {EMBEDDING_DIMENSION}"
            )


def replace_document(
    connection: psycopg.Connection[tuple[Any, ...]],
    document_id: str,
    content_hash: str,
    embedding_model: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Remplace un document et tous ses chunks, en une seule transaction.

    L'ancienne version du document est supprimée (ses chunks aussi, grâce à
    ON DELETE CASCADE), puis la nouvelle est insérée. Réindexer ne crée donc
    jamais de doublons, et une erreur en cours de route annule tout.

    Args:
        connection: connexion ouverte (voir storage.database.connect).
        document_id: identifiant du document dans le catalogue.
        content_hash: empreinte du fichier source du document.
        embedding_model: modèle qui a produit les vecteurs.
        chunks: chunks du document, dans l'ordre ; leur rang devient `position`.
        vectors: un vecteur par chunk, dans le même ordre.

    Raises:
        ValueError: si les vecteurs ne correspondent pas aux chunks (vérifié
            avant toute écriture).
    """
    _check_vectors(chunks, vectors)
    with connection.transaction():
        connection.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
        connection.execute(_INSERT_DOCUMENT, (document_id, content_hash, embedding_model))
        with connection.cursor() as cursor:
            cursor.executemany(
                _INSERT_CHUNK,
                [
                    (
                        document_id,
                        position,
                        chunk.article_id,
                        chunk.chunk_index,
                        chunk.version,
                        chunk.valid_from,
                        chunk.valid_until,
                        list(chunk.history),
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        Vector(list(vector)),
                    )
                    for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )


def search(
    connection: psycopg.Connection[tuple[Any, ...]],
    query_vector: Sequence[float],
    embedding_model: str,
    k: int = 5,
) -> list[SearchResult]:
    """Renvoie les k chunks en vigueur les plus proches d'un vecteur de question.

    Seuls les chunks vectorisés avec `embedding_model` sont comparés : les
    vecteurs de deux modèles différents ne sont pas comparables.

    Args:
        connection: connexion ouverte (voir storage.database.connect).
        query_vector: vecteur de la question (voir embedding.embed_query).
        embedding_model: modèle qui a produit `query_vector`.
        k: nombre maximal de résultats.

    Returns:
        Résultats du plus proche au moins proche.
    """
    rows = connection.execute(
        _SEARCH, {"query": Vector(list(query_vector)), "model": embedding_model, "k": k}
    ).fetchall()
    return [
        SearchResult(
            chunk=Chunk(
                document_id=row[0],
                article_id=row[1],
                version=row[2],
                valid_from=row[3],
                valid_until=row[4],
                history=tuple(row[5]),
                chunk_index=row[6],
                text=row[7],
                page_start=row[8],
                page_end=row[9],
            ),
            score=1.0 - row[10],
        )
        for row in rows
    ]

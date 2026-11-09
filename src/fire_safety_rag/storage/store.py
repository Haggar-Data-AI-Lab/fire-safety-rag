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

# plainto_tsquery exige tous les mots (ET) : trop strict pour une question en
# langage courant. Remplacer & par | garde les chunks qui en partagent au moins un.
_FULL_TEXT_SEARCH = """
WITH q AS (
    SELECT replace(plainto_tsquery('french', %(question)s)::text, '&', '|')::tsquery AS query
)
SELECT c.document_id, c.article_id, c.version, c.valid_from, c.valid_until,
       c.history, c.chunk_index, c.text, c.page_start, c.page_end,
       ts_rank(c.search_vector, q.query) AS rank
FROM chunks AS c
JOIN documents AS d USING (document_id)
CROSS JOIN q
WHERE d.embedding_model = %(model)s
  AND (c.valid_until IS NULL OR c.valid_until > current_date)
  AND c.search_vector @@ q.query
ORDER BY rank DESC, c.document_id, c.position
LIMIT %(k)s
"""

# Constante de la fusion RRF, valeur de l'article d'origine (Cormack et al., 2009).
RRF_CONSTANT = 60

# Candidats demandés à chaque méthode avant la fusion : assez pour qu'un chunk
# moyennement classé des deux côtés puisse remonter. Valeur non optimisée.
HYBRID_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Un chunk retrouvé par la recherche, avec son score.

    Attributes:
        chunk: chunk retrouvé, avec ses métadonnées (pages, dates, historique).
        score: similarité cosinus (recherche vectorielle), ts_rank (plein
            texte) ou score RRF (fusion) ; plus il est élevé, mieux le chunk
            est classé. Seul l'ordre est significatif : ces échelles ne sont
            pas comparables.
    """

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    """Reconstruit un Chunk à partir des dix premières colonnes d'une recherche."""
    return Chunk(
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
    )


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


def get_indexed_document(
    connection: psycopg.Connection[tuple[Any, ...]], document_id: str
) -> tuple[str, str] | None:
    """Renvoie l'empreinte et le modèle d'un document déjà indexé.

    Args:
        connection: connexion ouverte (voir storage.database.connect).
        document_id: identifiant du document dans le catalogue.

    Returns:
        Couple (content_hash, embedding_model), ou None si le document n'est
        pas en base.
    """
    row = connection.execute(
        "SELECT content_hash, embedding_model FROM documents WHERE document_id = %s",
        (document_id,),
    ).fetchone()
    return None if row is None else (row[0], row[1])


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
    return [SearchResult(chunk=_row_to_chunk(row), score=1.0 - row[10]) for row in rows]


def full_text_search(
    connection: psycopg.Connection[tuple[Any, ...]],
    question: str,
    embedding_model: str,
    k: int = 5,
) -> list[SearchResult]:
    """Renvoie les k chunks en vigueur qui partagent le plus de mots avec la question.

    Args:
        connection: connexion ouverte (voir storage.database.connect).
        question: question en langage courant.
        embedding_model: seuls les chunks de ce modèle sont cherchés, pour
            travailler sur les mêmes chunks que la recherche vectorielle.
        k: nombre maximal de résultats.

    Returns:
        Résultats du mieux au moins bien classé ; liste vide si la question ne
        contient que des mots vides.
    """
    rows = connection.execute(
        _FULL_TEXT_SEARCH, {"question": question, "model": embedding_model, "k": k}
    ).fetchall()
    return [SearchResult(chunk=_row_to_chunk(row), score=row[10]) for row in rows]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], k: int = 5
) -> list[SearchResult]:
    """Fusionne plusieurs classements en un seul (Reciprocal Rank Fusion).

    Chaque chunk gagne 1 / (RRF_CONSTANT + rang) dans chaque classement où il
    apparaît. Seuls les rangs comptent : les scores d'origine ne sont pas
    comparables entre eux. À score égal, le chunk vu en premier reste devant.

    Args:
        rankings: classements à fusionner, chacun du meilleur au moins bon.
        k: nombre maximal de résultats.

    Returns:
        Résultats fusionnés, du meilleur au moins bon ; `score` est le score RRF.
    """
    scores: dict[Chunk, float] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk] = scores.get(result.chunk, 0.0) + 1 / (RRF_CONSTANT + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ordered[:k]]


def hybrid_search(
    connection: psycopg.Connection[tuple[Any, ...]],
    question: str,
    query_vector: Sequence[float],
    embedding_model: str,
    k: int = 5,
) -> list[SearchResult]:
    """Combine la recherche vectorielle et le plein texte par fusion RRF.

    Args:
        connection: connexion ouverte (voir storage.database.connect).
        question: question en langage courant, pour le plein texte.
        query_vector: vecteur de la question (voir embedding.embed_query).
        embedding_model: modèle qui a produit `query_vector`.
        k: nombre maximal de résultats.

    Returns:
        Résultats fusionnés, du meilleur au moins bon.
    """
    by_vector = search(connection, query_vector, embedding_model, k=HYBRID_CANDIDATES)
    by_text = full_text_search(connection, question, embedding_model, k=HYBRID_CANDIDATES)
    return reciprocal_rank_fusion([by_vector, by_text], k=k)

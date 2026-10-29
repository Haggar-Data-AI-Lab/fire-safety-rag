"""Vectorisation des chunks et des questions avec un modèle d'embedding local.

Le reste du projet n'utilise que `embed_passages` et `embed_query` : pour passer
plus tard à une API d'embedding, seul ce fichier est à réécrire.

Le modèle par défaut (intfloat/multilingual-e5-base) impose deux préfixes,
propres à la famille e5 : "passage: " pour les textes indexés et "query: "
pour les questions. Les oublier dégrade la qualité de la recherche.

La bibliothèque sentence-transformers est une dépendance optionnelle
(extra "embeddings") : elle n'est importée qu'au chargement du modèle, pour
que le reste du projet fonctionne sans elle.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from fire_safety_rag.config import get_settings
from fire_safety_rag.ingestion.chunker import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


def format_passage(chunk: Chunk) -> str:
    """Construit le texte à vectoriser pour un chunk.

    Format : préfixe e5, en-tête de l'article, puis le texte du chunk.
    Exemples d'en-tête : "Article GN 1", "Annexe I".

    Args:
        chunk: chunk à vectoriser.

    Returns:
        Texte prêt à être passé au modèle.
    """
    is_annex = chunk.article_id.startswith("Annexe")
    heading = chunk.article_id if is_annex else f"Article {chunk.article_id}"
    return f"{_PASSAGE_PREFIX}{heading}\n{chunk.text}"


def format_query(question: str) -> str:
    """Construit le texte à vectoriser pour une question.

    Args:
        question: question posée par l'utilisateur.

    Returns:
        Question précédée du préfixe e5 "query: ".
    """
    return f"{_QUERY_PREFIX}{question}"


@lru_cache
def _load_model(model_name: str) -> "SentenceTransformer":
    """Charge le modèle d'embedding, une seule fois par nom de modèle.

    Le premier appel télécharge le modèle depuis Hugging Face s'il n'est pas
    déjà dans le cache local (~/.cache/huggingface).

    Args:
        model_name: identifiant Hugging Face du modèle.

    Returns:
        Modèle chargé.
    """
    # Import différé : sentence-transformers est une dépendance optionnelle
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_passages(chunks: Sequence[Chunk], batch_size: int = 32) -> list[list[float]]:
    """Calcule les vecteurs normalisés d'une liste de chunks.

    Args:
        chunks: chunks à vectoriser.
        batch_size: nombre de textes traités ensemble par le modèle.

    Returns:
        Un vecteur par chunk, dans le même ordre.
    """
    model = _load_model(get_settings().embedding_model)
    embeddings = model.encode(
        [format_passage(chunk) for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    vectors: list[list[float]] = embeddings.tolist()
    return vectors


def embed_query(question: str) -> list[float]:
    """Calcule le vecteur normalisé d'une question.

    Args:
        question: question posée par l'utilisateur.

    Returns:
        Vecteur de la question.
    """
    model = _load_model(get_settings().embedding_model)
    embedding = model.encode(format_query(question), normalize_embeddings=True)
    vector: list[float] = embedding.tolist()
    return vector

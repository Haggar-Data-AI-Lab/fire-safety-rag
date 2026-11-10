"""Reclassement des passages retrouvés avec un cross-encodeur local.

La recherche vectorielle (bi-encodeur) compare deux vecteurs calculés
séparément : rapide, mais la question et le passage ne sont jamais lus
ensemble. Le cross-encodeur reçoit le couple (question, passage) en une seule
entrée et donne un score de pertinence : plus lent, mais plus précis.

Usage : la recherche vectorielle fournit une présélection (ex. 30 candidats),
le reranker les reclasse et on garde les meilleurs (ex. 5).

Comme pour les embeddings, sentence-transformers (extra "embeddings") n'est
importé qu'au chargement du modèle.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

# Note des couples (question, passage) : un score par couple, plus haut = plus pertinent.
Scorer = Callable[[list[tuple[str, str]]], list[float]]


def format_candidate(chunk: Chunk, document_title: str) -> str:
    """Texte d'un passage tel que lu par le reranker : en-tête puis contenu.

    L'en-tête (titre court du document + article) aide à distinguer des
    articles proches issus de textes différents (ex. Code du travail / ERP).

    Args:
        chunk: passage à présenter.
        document_title: titre court du document.

    Returns:
        "<titre> — Article X" puis le texte du passage.
    """
    is_annex = chunk.article_id.startswith("Annexe")
    heading = chunk.article_id if is_annex else f"Article {chunk.article_id}"
    return f"{document_title} — {heading}\n{chunk.text}"


@lru_cache
def _load_model(model_name: str) -> "CrossEncoder":
    """Charge le cross-encodeur, une seule fois par nom de modèle.

    Le premier appel télécharge le modèle dans le cache Hugging Face local.
    """
    # Import différé : sentence-transformers est une dépendance optionnelle
    from sentence_transformers import CrossEncoder

    model: CrossEncoder = CrossEncoder(model_name, max_length=512)
    return model


def model_scorer(model_name: str, batch_size: int = 8) -> Scorer:
    """Construit la fonction de notation qui utilise un vrai cross-encodeur.

    Args:
        model_name: identifiant Hugging Face du modèle.
        batch_size: nombre de couples traités ensemble.

    Returns:
        Fonction couples -> scores.
    """

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        model = _load_model(model_name)
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(value) for value in scores]

    return score


def rerank(
    question: str,
    results: Sequence[SearchResult],
    titles: Mapping[str, str],
    score_fn: Scorer,
    top_k: int = 5,
) -> list[SearchResult]:
    """Reclasse les passages selon le cross-encodeur et garde les meilleurs.

    Args:
        question: question de l'utilisateur.
        results: candidats de la recherche vectorielle.
        titles: titre court de chaque document, indexé par document_id ;
            un document absent est présenté avec son document_id.
        score_fn: notation des couples (model_scorer en usage réel).
        top_k: nombre de passages gardés.

    Returns:
        Les top_k passages, du plus pertinent au moins pertinent, avec le
        score du reranker à la place de la similarité cosinus. À score égal,
        l'ordre de la recherche vectorielle est conservé.
    """
    if not results:
        return []
    pairs = [
        (question, format_candidate(r.chunk, titles.get(r.chunk.document_id, r.chunk.document_id)))
        for r in results
    ]
    scores = score_fn(pairs)
    if len(scores) != len(results):
        raise ValueError(f"{len(scores)} scores reçus pour {len(results)} passages.")
    order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    return [SearchResult(chunk=results[i].chunk, score=scores[i]) for i in order[:top_k]]

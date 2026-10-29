"""Métriques de recherche : Hit@k et rang réciproque (pour le MRR).

Les résultats sont des clés "document_id:article_id", dans l'ordre renvoyé par
la recherche. Plusieurs chunks d'un même article peuvent apparaître : le rang
compté est celui du premier chunk qui correspond à un article attendu.
"""

from collections.abc import Collection, Sequence


def first_relevant_rank(retrieved: Sequence[str], expected: Collection[str]) -> int | None:
    """Renvoie le rang (à partir de 1) du premier résultat attendu.

    Args:
        retrieved: clés des résultats, du plus proche au moins proche.
        expected: clés acceptées comme bonne réponse.

    Returns:
        Rang du premier résultat présent dans `expected`, ou None si aucun.
    """
    for rank, key in enumerate(retrieved, start=1):
        if key in expected:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    """Indique si la bonne réponse apparaît dans les k premiers résultats.

    Args:
        rank: rang du premier résultat attendu (voir first_relevant_rank).
        k: nombre de résultats regardés.

    Returns:
        True si `rank` existe et vaut au plus `k`.
    """
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    """Renvoie 1 / rang, ou 0 si la bonne réponse n'a pas été trouvée.

    La moyenne de cette valeur sur toutes les questions est le MRR
    (Mean Reciprocal Rank) : 1 si la bonne réponse est toujours en tête.

    Args:
        rank: rang du premier résultat attendu (voir first_relevant_rank).

    Returns:
        Rang réciproque, entre 0 et 1.
    """
    return 0.0 if rank is None else 1.0 / rank

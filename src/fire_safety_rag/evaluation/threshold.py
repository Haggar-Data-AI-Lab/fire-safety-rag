"""Aide au choix du seuil de refus (score du meilleur passage).

Le RAG refuse de répondre, sans appeler le LLM, quand le meilleur passage
retrouvé a un score sous un seuil. Pour choisir ce seuil, on compare trois
groupes de questions :
- "trouvée" : question du corpus dont le bon article est dans le top k ;
- "manquée" : question du corpus dont le bon article n'est pas dans le top k ;
- "hors corpus" : question sans réponse dans le corpus
  (data/eval/hors_corpus.csv).

Un bon seuil refuse très peu de questions "trouvées" et beaucoup de questions
"hors corpus". Recherche évaluée : mode vecteurs (mode retenu par défaut).

Utilisation (base indexée, extra "embeddings" installé) :
    python -m fire_safety_rag.evaluation.threshold
"""

import csv
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fire_safety_rag.config import get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.evaluation.evaluate import (
    DEFAULT_QUESTIONS,
    EvalQuestion,
    Retriever,
    load_questions,
    make_retriever,
)
from fire_safety_rag.evaluation.metrics import first_relevant_rank
from fire_safety_rag.ingestion.pipeline import PROJECT_ROOT
from fire_safety_rag.storage.database import connect

DEFAULT_OFF_CORPUS = PROJECT_ROOT / "data" / "eval" / "hors_corpus.csv"

FOUND, MISSED, OFF_CORPUS = "trouvée", "manquée", "hors corpus"
GROUPS = (FOUND, MISSED, OFF_CORPUS)


@dataclass(frozen=True)
class ScoredQuestion:
    """Score du meilleur passage retrouvé pour une question.

    Attributes:
        id: identifiant de la question (ex. "q01", "h01").
        group: une des valeurs de GROUPS.
        top_score: similarité cosinus du meilleur passage (0 si aucun passage).
    """

    id: str
    group: str
    top_score: float


def load_off_corpus(path: Path) -> list[EvalQuestion]:
    """Lit les questions hors corpus (CSV : id, question, note).

    Args:
        path: chemin du fichier CSV.

    Returns:
        Questions dans l'ordre du fichier, sans article attendu.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            EvalQuestion(
                id=row["id"], question=row["question"], expected=frozenset(), note=row["note"]
            )
            for row in csv.DictReader(f)
        ]


def score_questions(
    questions: Sequence[EvalQuestion], retrieve: Retriever, k: int = 5
) -> list[ScoredQuestion]:
    """Interroge la recherche et classe chaque question dans un groupe.

    Une question sans article attendu est "hors corpus" ; sinon elle est
    "trouvée" si un article attendu est dans le top k, "manquée" sinon.

    Args:
        questions: questions du corpus et hors corpus.
        retrieve: recherche à évaluer (voir make_retriever).
        k: nombre de résultats regardés par question.

    Returns:
        Un résultat par question, dans le même ordre.
    """
    scored: list[ScoredQuestion] = []
    for question in questions:
        hits = retrieve(question.question, k)
        top_score = hits[0].score if hits else 0.0
        if not question.expected:
            group = OFF_CORPUS
        else:
            retrieved = [f"{hit.chunk.document_id}:{hit.chunk.article_id}" for hit in hits]
            rank = first_relevant_rank(retrieved, question.expected)
            group = FOUND if rank is not None else MISSED
        scored.append(ScoredQuestion(question.id, group, top_score))
    return scored


def refusal_counts(
    scored: Sequence[ScoredQuestion], threshold: float
) -> dict[str, tuple[int, int]]:
    """Compte, par groupe, les questions refusées avec un seuil donné.

    Une question est refusée quand son meilleur score est strictement
    inférieur au seuil.

    Args:
        scored: questions notées (voir score_questions).
        threshold: seuil de refus.

    Returns:
        Pour chaque groupe de GROUPS : (nombre refusé, nombre total).
    """
    counts: dict[str, tuple[int, int]] = {}
    for group in GROUPS:
        in_group = [s for s in scored if s.group == group]
        refused = sum(1 for s in in_group if s.top_score < threshold)
        counts[group] = (refused, len(in_group))
    return counts


def candidate_thresholds(scored: Sequence[ScoredQuestion], step: float = 0.01) -> list[float]:
    """Seuils candidats, au pas `step`, couvrant tous les scores observés.

    Args:
        scored: questions notées.
        step: écart entre deux seuils.

    Returns:
        Seuils croissants, du plus bas score (arrondi en dessous) au plus
        haut (arrondi au-dessus) ; liste vide si aucune question.
    """
    if not scored:
        return []
    low = math.floor(min(s.top_score for s in scored) / step)
    high = math.ceil(max(s.top_score for s in scored) / step)
    return [round(i * step, 10) for i in range(low, high + 1)]


def main() -> None:
    """Affiche les scores par question puis le tableau des refus par seuil."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    questions = load_questions(DEFAULT_QUESTIONS) + load_off_corpus(DEFAULT_OFF_CORPUS)
    with connect(settings) as connection:
        retrieve = make_retriever(connection, "vecteurs", embed_query, settings.embedding_model)
        scored = score_questions(questions, retrieve)

    print(f"\nScore du meilleur passage ({settings.embedding_model}, mode vecteurs) :")
    for s in sorted(scored, key=lambda s: s.top_score, reverse=True):
        print(f"  {s.top_score:.4f}  {s.id:4}  {s.group}")

    print("\nRefus par seuil (refusées / total) :")
    print(f"  {'seuil':>6}  " + "  ".join(f"{group:>12}" for group in GROUPS))
    for threshold in candidate_thresholds(scored):
        counts = refusal_counts(scored, threshold)
        cells = "  ".join(f"{f'{r} / {n}':>12}" for r, n in counts.values())
        print(f"  {threshold:6.2f}  {cells}")


if __name__ == "__main__":
    main()

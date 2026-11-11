"""Simulation hors ligne du reranker avec moins de candidats.

Lit le fichier produit par `evaluate --mode vecteurs+reranker --save-candidates`
(notes du reranker pour les N candidats vectoriels de chaque question).

Le reranker note chaque couple (question, passage) indépendamment des autres :
le résultat obtenu avec M < N candidats se recalcule donc exactement en gardant
les M premiers par rang vectoriel, puis en les triant par note du reranker.
Seul le temps est estimé (proportionnel au nombre de candidats), pas mesuré.

Utilisation :
    python -m fire_safety_rag.evaluation.candidates notes/candidats-reranker-AAAA-MM-JJ-HHMM.jsonl
"""

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fire_safety_rag.evaluation.evaluate import EvalQuestion, QuestionResult, summarize
from fire_safety_rag.evaluation.metrics import first_relevant_rank

DEFAULT_BUDGETS = (5, 10, 15, 20, 30)


@dataclass(frozen=True)
class Candidate:
    """Un candidat vectoriel et sa note du reranker.

    Attributes:
        vector_rank: rang dans la recherche vectorielle (à partir de 1).
        key: clé "document_id:article_id".
        rerank_score: note du reranker (plus haut = plus pertinent).
    """

    vector_rank: int
    key: str
    rerank_score: float


@dataclass(frozen=True)
class CandidateLog:
    """Candidats enregistrés pour une question.

    Attributes:
        id: identifiant de la question (ex. "q01").
        expected: articles acceptés comme bonne réponse.
        seconds: durée mesurée du reclassement de tous les candidats.
        candidates: candidats dans l'ordre vectoriel.
    """

    id: str
    expected: frozenset[str]
    seconds: float
    candidates: list[Candidate]


def load_logs(path: Path) -> list[CandidateLog]:
    """Relit le fichier JSONL écrit par evaluate.write_rerank_logs.

    Args:
        path: fichier notes/candidats-reranker-….jsonl.

    Returns:
        Une entrée par question, dans l'ordre du fichier.
    """
    logs: list[CandidateLog] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            logs.append(
                CandidateLog(
                    id=row["id"],
                    expected=frozenset(row["attendus"]),
                    seconds=row["duree_reranker_s"],
                    candidates=[
                        Candidate(c["rang_vecteur"], c["cle"], c["score_reranker"])
                        for c in row["candidats"]
                    ],
                )
            )
    return logs


def simulate(candidates: Sequence[Candidate], budget: int, k: int = 5) -> list[str]:
    """Résultat qu'aurait donné le reranker avec seulement `budget` candidats.

    Args:
        candidates: tous les candidats enregistrés pour une question.
        budget: nombre de candidats vectoriels relus par le reranker.
        k: nombre de résultats gardés.

    Returns:
        Les k clés retenues, de la meilleure note à la moins bonne ; à note
        égale, l'ordre vectoriel est conservé (comme reranker.rerank).

    Raises:
        ValueError: budget supérieur au nombre de candidats enregistrés.
    """
    if budget > len(candidates):
        raise ValueError(f"{budget} candidats demandés, {len(candidates)} enregistrés.")
    kept = sorted(
        (c for c in candidates if c.vector_rank <= budget),
        key=lambda c: (-c.rerank_score, c.vector_rank),
    )
    return [c.key for c in kept[:k]]


def summarize_budget(logs: Sequence[CandidateLog], budget: int, k: int = 5) -> dict[str, float]:
    """Métriques simulées pour un nombre de candidats, plus le temps estimé.

    Args:
        logs: candidats enregistrés (voir load_logs).
        budget: nombre de candidats relus par le reranker.
        k: nombre de résultats regardés.

    Returns:
        Hit@1, Hit@3, Hit@k, MRR (voir evaluate.summarize) et "Temps estimé"
        en secondes par question : durée mesurée × budget / candidats enregistrés.
    """
    results: list[QuestionResult] = []
    for log in logs:
        retrieved = simulate(log.candidates, budget, k)
        question = EvalQuestion(log.id, "", log.expected, "")
        results.append(
            QuestionResult(question, retrieved, first_relevant_rank(retrieved, log.expected))
        )
    summary = summarize(results, k)
    summary["Temps estimé"] = statistics.mean(
        log.seconds * budget / len(log.candidates) for log in logs
    )
    return summary


def best_vector_rank(log: CandidateLog) -> int | None:
    """Meilleur rang vectoriel d'un article attendu parmi les candidats.

    Returns:
        Rang (à partir de 1), ou None si aucun article attendu n'est parmi
        les candidats : le reranker ne pouvait alors pas le retrouver.
    """
    return min((c.vector_rank for c in log.candidates if c.key in log.expected), default=None)


def main() -> None:
    """Affiche les métriques simulées pour plusieurs nombres de candidats."""
    parser = argparse.ArgumentParser(
        description="Simule le reranker avec moins de candidats, à partir des notes enregistrées."
    )
    parser.add_argument("path", type=Path, help="fichier notes/candidats-reranker-….jsonl")
    args = parser.parse_args()

    logs = load_logs(args.path)
    recorded = min(len(log.candidates) for log in logs)
    budgets = [b for b in DEFAULT_BUDGETS if b <= recorded]

    print(f"{len(logs)} questions, {recorded} candidats enregistrés ({args.path.name})")
    print("Temps estimé = durée mesurée × candidats / enregistrés (non mesuré)\n")
    print(f"  {'Cand.':>5}  {'Hit@1':>5}  {'Hit@3':>5}  {'Hit@5':>5}  {'MRR':>5}  Temps estimé")
    for budget in budgets:
        s = summarize_budget(logs, budget)
        print(
            f"  {budget:>5}  {s['Hit@1']:>5.2f}  {s['Hit@3']:>5.2f}  {s['Hit@5']:>5.2f}"
            f"  {s['MRR']:>5.2f}  {s['Temps estimé']:.1f} s"
        )

    print(f"\nQuestions absentes du top 5 avec {recorded} candidats :")
    for log in logs:
        if first_relevant_rank(simulate(log.candidates, recorded), log.expected) is None:
            rank = best_vector_rank(log)
            where = f"> {recorded}" if rank is None else str(rank)
            print(f"  {log.id}  meilleur rang vectoriel d'un attendu : {where}")


if __name__ == "__main__":
    main()

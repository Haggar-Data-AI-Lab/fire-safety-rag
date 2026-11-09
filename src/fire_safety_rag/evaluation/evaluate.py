"""Évaluation de la recherche sur le jeu de questions de data/eval/questions.csv.

Pour chaque question : vectorisation, recherche des k chunks les plus proches,
puis rang du premier article attendu. Affiche Hit@1, Hit@3, Hit@k, le MRR et
le détail des questions dont la bonne réponse n'est pas en tête.

Utilisation (base indexée, extra "embeddings" installé) :
    python -m fire_safety_rag.evaluation.evaluate [--mode vecteurs|plein-texte|hybride]
"""

import argparse
import csv
import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from fire_safety_rag.config import get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.evaluation.metrics import first_relevant_rank, hit_at_k, reciprocal_rank
from fire_safety_rag.ingestion.pipeline import PROJECT_ROOT
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.store import SearchResult, full_text_search, hybrid_search, search

DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "questions.csv"

RETRIEVAL_MODES = ("vecteurs", "plein-texte", "hybride")

# Recherche à évaluer : (question, k) -> résultats du meilleur au moins bon.
Retriever = Callable[[str, int], list[SearchResult]]


@dataclass(frozen=True)
class EvalQuestion:
    """Une question du jeu d'évaluation.

    Attributes:
        id: identifiant court (ex. "q01").
        question: question en langage courant.
        expected: articles acceptés, sous forme "document_id:article_id".
        note: ce que l'article attendu répond (aide à la relecture).
    """

    id: str
    question: str
    expected: frozenset[str]
    note: str


@dataclass(frozen=True)
class QuestionResult:
    """Résultat de la recherche pour une question.

    Attributes:
        question: question évaluée.
        retrieved: clés "document_id:article_id" renvoyées, dans l'ordre.
        rank: rang du premier article attendu, ou None s'il n'est pas trouvé.
    """

    question: EvalQuestion
    retrieved: list[str]
    rank: int | None


def load_questions(path: Path) -> list[EvalQuestion]:
    """Lit le jeu de questions (CSV : id, question, expected, note).

    Dans la colonne `expected`, plusieurs articles sont séparés par ";".

    Args:
        path: chemin du fichier CSV.

    Returns:
        Questions dans l'ordre du fichier.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            EvalQuestion(
                id=row["id"],
                question=row["question"],
                expected=frozenset(key.strip() for key in row["expected"].split(";")),
                note=row["note"],
            )
            for row in csv.DictReader(f)
        ]


def make_retriever(
    connection: psycopg.Connection[tuple[Any, ...]],
    mode: str,
    embed: Callable[[str], list[float]],
    embedding_model: str,
) -> Retriever:
    """Construit la fonction de recherche correspondant à un mode.

    Args:
        connection: connexion à une base indexée.
        mode: une des valeurs de RETRIEVAL_MODES.
        embed: fonction qui vectorise une question (embed_query en usage réel).
        embedding_model: modèle qui a produit les vecteurs en base.

    Returns:
        Fonction (question, k) -> résultats.
    """

    def by_vector(question: str, k: int) -> list[SearchResult]:
        return search(connection, embed(question), embedding_model, k=k)

    def by_text(question: str, k: int) -> list[SearchResult]:
        return full_text_search(connection, question, embedding_model, k=k)

    def hybrid(question: str, k: int) -> list[SearchResult]:
        return hybrid_search(connection, question, embed(question), embedding_model, k=k)

    return {"vecteurs": by_vector, "plein-texte": by_text, "hybride": hybrid}[mode]


def evaluate(
    questions: Sequence[EvalQuestion], retrieve: Retriever, k: int = 5
) -> list[QuestionResult]:
    """Interroge la recherche pour chaque question et repère le rang de la bonne réponse.

    Args:
        questions: questions à évaluer.
        retrieve: recherche à évaluer (voir make_retriever).
        k: nombre de résultats regardés par question.

    Returns:
        Un résultat par question, dans le même ordre.
    """
    results: list[QuestionResult] = []
    for question in questions:
        hits = retrieve(question.question, k)
        retrieved = [f"{hit.chunk.document_id}:{hit.chunk.article_id}" for hit in hits]
        results.append(
            QuestionResult(question, retrieved, first_relevant_rank(retrieved, question.expected))
        )
    return results


def summarize(results: Sequence[QuestionResult], k: int = 5) -> dict[str, float]:
    """Calcule les métriques globales.

    Args:
        results: résultats par question (voir evaluate).
        k: nombre de résultats regardés (pour Hit@k).

    Returns:
        Dictionnaire {"Hit@1", "Hit@3", "Hit@k", "MRR"} -> valeur entre 0 et 1.
    """
    ranks = [result.rank for result in results]
    return {
        "Hit@1": statistics.mean(hit_at_k(rank, 1) for rank in ranks),
        "Hit@3": statistics.mean(hit_at_k(rank, 3) for rank in ranks),
        f"Hit@{k}": statistics.mean(hit_at_k(rank, k) for rank in ranks),
        "MRR": statistics.mean(reciprocal_rank(rank) for rank in ranks),
    }


def main() -> None:
    """Évalue la recherche sur le jeu de questions par défaut et affiche le bilan."""
    parser = argparse.ArgumentParser(description="Évalue la recherche sur le jeu de questions.")
    parser.add_argument(
        "--mode", choices=RETRIEVAL_MODES, default="vecteurs", help="méthode de recherche évaluée"
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    # Les requêtes de vérification de Hugging Face ne sont que du bruit ici
    logging.getLogger("httpx").setLevel(logging.WARNING)

    questions = load_questions(DEFAULT_QUESTIONS)
    with connect(settings) as connection:
        retrieve = make_retriever(connection, args.mode, embed_query, settings.embedding_model)
        results = evaluate(questions, retrieve)

    print(f"\n{len(results)} questions évaluées ({settings.embedding_model}, mode {args.mode})")
    for name, value in summarize(results).items():
        print(f"  {name:6} {value:.2f}")

    print("\nQuestions dont la bonne réponse n'est pas en tête :")
    for result in results:
        if result.rank != 1:
            rank = "absent du top 5" if result.rank is None else f"rang {result.rank}"
            print(f"  {result.question.id} ({rank}) {result.question.question}")
            print(f"      attendu : {', '.join(sorted(result.question.expected))}")
            print(f"      obtenu  : {', '.join(result.retrieved[:3])}")


if __name__ == "__main__":
    main()

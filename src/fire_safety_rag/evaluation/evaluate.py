"""Évaluation de la recherche sur le jeu de questions de data/eval/questions.csv.

Pour chaque question : vectorisation, recherche des k chunks les plus proches,
puis rang du premier article attendu. Affiche Hit@1, Hit@3, Hit@k, le MRR et
le détail des questions dont la bonne réponse n'est pas en tête.

Utilisation (base indexée, extra "embeddings" installé) :
    python -m fire_safety_rag.evaluation.evaluate [--mode MODE]
    (MODE : vecteurs, plein-texte, hybride ou vecteurs+reranker ; défaut vecteurs)

Le mode vecteurs+reranker utilise RERANKER_MODEL (ou BAAI/bge-reranker-v2-m3
si vide) sur RERANK_CANDIDATES candidats ; le premier lancement télécharge le modèle.
Avec --save-candidates, les notes du reranker de tous les candidats sont
enregistrées dans notes/candidats-reranker-AAAA-MM-JJ-HHMM.jsonl, pour simuler
hors ligne un nombre de candidats plus petit.
"""

import argparse
import csv
import json
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg

from fire_safety_rag.config import get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.evaluation.metrics import first_relevant_rank, hit_at_k, reciprocal_rank
from fire_safety_rag.ingestion.pipeline import DEFAULT_CATALOGUE, PROJECT_ROOT, load_catalogue
from fire_safety_rag.reranking.reranker import DEFAULT_RERANKER, model_scorer, rerank
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.store import SearchResult, full_text_search, hybrid_search, search

DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "questions.csv"
NOTES_DIR = PROJECT_ROOT / "notes"

RETRIEVAL_MODES = ("vecteurs", "plein-texte", "hybride", "vecteurs+reranker")

# Recherche à évaluer : (question, k) -> résultats du meilleur au moins bon.
Retriever = Callable[[str, int], list[SearchResult]]
# Reclassement : (question, candidats, k) -> k meilleurs passages.
Reranker = Callable[[str, list[SearchResult], int], list[SearchResult]]


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
    rerank_fn: Reranker | None = None,
    candidates: int = 30,
) -> Retriever:
    """Construit la fonction de recherche correspondant à un mode.

    Args:
        connection: connexion à une base indexée.
        mode: une des valeurs de RETRIEVAL_MODES.
        embed: fonction qui vectorise une question (embed_query en usage réel).
        embedding_model: modèle qui a produit les vecteurs en base.
        rerank_fn: reclassement, exigé par le mode "vecteurs+reranker".
        candidates: candidats vectoriels relus par le reranker.

    Returns:
        Fonction (question, k) -> résultats.

    Raises:
        ValueError: mode "vecteurs+reranker" sans rerank_fn.
    """
    if mode == "vecteurs+reranker" and rerank_fn is None:
        raise ValueError("Le mode vecteurs+reranker exige une fonction de reclassement.")

    def by_vector(question: str, k: int) -> list[SearchResult]:
        return search(connection, embed(question), embedding_model, k=k)

    def by_text(question: str, k: int) -> list[SearchResult]:
        return full_text_search(connection, question, embedding_model, k=k)

    def hybrid(question: str, k: int) -> list[SearchResult]:
        return hybrid_search(connection, question, embed(question), embedding_model, k=k)

    def reranked(question: str, k: int) -> list[SearchResult]:
        assert rerank_fn is not None  # vérifié plus haut
        return rerank_fn(question, by_vector(question, max(candidates, k)), k)

    modes: dict[str, Retriever] = {
        "vecteurs": by_vector,
        "plein-texte": by_text,
        "hybride": hybrid,
        "vecteurs+reranker": reranked,
    }
    return modes[mode]


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


@dataclass(frozen=True)
class RerankLog:
    """Notes du reranker pour tous les candidats d'une question.

    Attributes:
        question: texte de la question.
        seconds: durée du reclassement seul (sans la recherche vectorielle).
        candidates: un dictionnaire par candidat, dans l'ordre vectoriel :
            rang_vecteur, cle, score_vecteur, score_reranker.
    """

    question: str
    seconds: float
    candidates: list[dict[str, Any]]


def recording_reranker(rerank_fn: Reranker, logs: list[RerankLog]) -> Reranker:
    """Enveloppe un reclassement pour garder les notes de TOUS les candidats.

    Le reranker note chaque couple (question, passage) indépendamment des
    autres. Avec les notes des N candidats, on peut donc recalculer hors ligne
    le résultat obtenu avec moins de candidats : garder les M premiers par
    rang vectoriel, puis les trier par note du reranker.

    Args:
        rerank_fn: reclassement réel ; il doit renvoyer les mêmes objets
            Chunk que ceux reçus (c'est le cas de reranker.rerank).
        logs: liste complétée d'un RerankLog à chaque appel.

    Returns:
        Reclassement qui renvoie le même résultat que rerank_fn.
    """

    def wrapped(question: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
        start = time.perf_counter()
        ranked = rerank_fn(question, candidates, len(candidates))
        seconds = time.perf_counter() - start
        # Correspondance par identité d'objet : deux annexes ERP peuvent
        # partager document_id, article_id et chunk_index
        rerank_scores = {id(result.chunk): result.score for result in ranked}
        logs.append(
            RerankLog(
                question=question,
                seconds=seconds,
                candidates=[
                    {
                        "rang_vecteur": rank,
                        "cle": f"{c.chunk.document_id}:{c.chunk.article_id}",
                        "score_vecteur": c.score,
                        "score_reranker": rerank_scores[id(c.chunk)],
                    }
                    for rank, c in enumerate(candidates, start=1)
                ],
            )
        )
        return ranked[:k]

    return wrapped


def write_rerank_logs(
    path: Path, questions: Sequence[EvalQuestion], logs: Sequence[RerankLog]
) -> None:
    """Écrit les notes du reranker en JSONL, une ligne par question.

    Args:
        path: fichier à créer (refusé s'il existe déjà).
        questions: questions évaluées, pour retrouver id et articles attendus.
        logs: notes enregistrées par recording_reranker.

    Raises:
        FileExistsError: le fichier existe déjà (un lancement coûte ~20 min).
    """
    by_text = {question.question: question for question in questions}
    with path.open("x", encoding="utf-8") as f:
        for log in logs:
            question = by_text[log.question]
            line = {
                "id": question.id,
                "attendus": sorted(question.expected),
                "duree_reranker_s": round(log.seconds, 3),
                "candidats": log.candidates,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lit et vérifie les options de la ligne de commande.

    Args:
        argv: options à lire ; None = celles de la ligne de commande.

    Returns:
        Options mode, k et save_candidates.
    """
    parser = argparse.ArgumentParser(description="Évalue la recherche sur le jeu de questions.")
    parser.add_argument(
        "--mode", choices=RETRIEVAL_MODES, default="vecteurs", help="méthode de recherche évaluée"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="résultats regardés par question (5 par défaut ; plus grand pour situer un attendu)",
    )
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help="enregistre les notes du reranker de tous les candidats dans notes/ (JSONL)",
    )
    args = parser.parse_args(argv)
    if args.save_candidates and args.mode != "vecteurs+reranker":
        parser.error("--save-candidates n'a de sens qu'avec --mode vecteurs+reranker")
    if args.k < 1:
        parser.error("--k doit valoir au moins 1")
    return args


def main() -> None:
    """Évalue la recherche sur le jeu de questions par défaut et affiche le bilan."""
    args = parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    # Les requêtes de vérification de Hugging Face ne sont que du bruit ici
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rerank_fn: Reranker | None = None
    label = settings.embedding_model
    if args.mode == "vecteurs+reranker":
        reranker_model = settings.reranker_model or DEFAULT_RERANKER
        titles = {row["id"]: row["titre_court"] for row in load_catalogue(DEFAULT_CATALOGUE)}
        scorer = model_scorer(reranker_model)

        def rerank_fn(question: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
            return rerank(question, candidates, titles, scorer, top_k=k)

        label += f" + {reranker_model}, {settings.rerank_candidates} candidats"

    logs: list[RerankLog] = []
    if args.save_candidates and rerank_fn is not None:
        rerank_fn = recording_reranker(rerank_fn, logs)

    questions = load_questions(DEFAULT_QUESTIONS)
    with connect(settings) as connection:
        retrieve = make_retriever(
            connection,
            args.mode,
            embed_query,
            settings.embedding_model,
            rerank_fn=rerank_fn,
            candidates=settings.rerank_candidates,
        )
        # Premier appel hors chronométrage : il charge les modèles
        retrieve(questions[0].question, 5)
        logs.clear()  # le préchauffage ne compte pas
        start = time.perf_counter()
        results = evaluate(questions, retrieve, k=args.k)
        seconds_per_question = (time.perf_counter() - start) / len(questions)

    if args.save_candidates:
        NOTES_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        logs_path = NOTES_DIR / f"candidats-reranker-{stamp}.jsonl"
        write_rerank_logs(logs_path, questions, logs)
        print(f"\nNotes des candidats enregistrées : {logs_path}")

    print(f"\n{len(results)} questions évaluées ({label}, mode {args.mode})")
    if args.k != 5:
        print(f"  (top {args.k} : Hit@{args.k} et MRR non comparables aux résultats du top 5)")
    for name, value in summarize(results, k=args.k).items():
        print(f"  {name:6} {value:.2f}")
    print(f"  Temps moyen par question : {seconds_per_question:.2f} s")

    print("\nQuestions dont la bonne réponse n'est pas en tête :")
    for result in results:
        if result.rank != 1:
            rank = f"absent du top {args.k}" if result.rank is None else f"rang {result.rank}"
            print(f"  {result.question.id} ({rank}) {result.question.question}")
            print(f"      attendu : {', '.join(sorted(result.question.expected))}")
            print(f"      obtenu  : {', '.join(result.retrieved[:3])}")


if __name__ == "__main__":
    main()

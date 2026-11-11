"""Tests des métriques et du chargement du jeu de questions (sans base)."""

import json
from datetime import date
from pathlib import Path

import pytest

from fire_safety_rag.evaluation.evaluate import (
    DEFAULT_QUESTIONS,
    EvalQuestion,
    QuestionResult,
    RerankLog,
    evaluate,
    load_questions,
    parse_args,
    recording_reranker,
    summarize,
    write_rerank_logs,
)
from fire_safety_rag.evaluation.metrics import first_relevant_rank, hit_at_k, reciprocal_rank
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.reranking.reranker import rerank
from fire_safety_rag.storage.store import SearchResult


class TestMetrics:
    """Métriques calculées sur des exemples faits à la main."""

    def test_rang_du_premier_resultat_attendu(self) -> None:
        """Le rang compte à partir de 1 et s'arrête au premier article attendu."""
        retrieved = ["doc:A", "doc:B", "doc:C", "doc:B"]
        assert first_relevant_rank(retrieved, {"doc:B"}) == 2
        assert first_relevant_rank(retrieved, {"doc:C", "doc:A"}) == 1
        assert first_relevant_rank(retrieved, {"doc:Z"}) is None

    def test_hit_at_k(self) -> None:
        """Hit@k est vrai si le rang existe et vaut au plus k."""
        assert hit_at_k(3, 3)
        assert not hit_at_k(4, 3)
        assert not hit_at_k(None, 5)

    def test_reciprocal_rank(self) -> None:
        """1 / rang, et 0 si la bonne réponse est absente."""
        assert reciprocal_rank(1) == 1.0
        assert reciprocal_rank(4) == 0.25
        assert reciprocal_rank(None) == 0.0

    def test_summarize(self) -> None:
        """Trois questions aux rangs 1, 3 et absent : MRR = (1 + 1/3 + 0) / 3."""
        question = EvalQuestion("q", "?", frozenset({"doc:A"}), "")
        results = [QuestionResult(question, [], rank) for rank in (1, 3, None)]

        summary = summarize(results)

        assert summary["Hit@1"] == pytest.approx(1 / 3)
        assert summary["Hit@3"] == pytest.approx(2 / 3)
        assert summary["Hit@5"] == pytest.approx(2 / 3)
        assert summary["MRR"] == pytest.approx((1 + 1 / 3) / 3)


def _result(article_id: str) -> SearchResult:
    """Résultat de recherche minimal pour un article du document "doc"."""
    chunk = Chunk(
        document_id="doc",
        article_id=article_id,
        version="VERSION",
        valid_from=date(2020, 1, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text="Texte.",
        page_start=1,
        page_end=1,
    )
    return SearchResult(chunk=chunk, score=0.5)


def test_evaluate_repere_le_rang_de_l_article_attendu() -> None:
    """evaluate transmet k à la recherche et repère le rang de l'article attendu."""
    question = EvalQuestion("q01", "Question ?", frozenset({"doc:B"}), "")

    def fake_retrieve(text: str, k: int) -> list[SearchResult]:
        assert (text, k) == ("Question ?", 3)
        return [_result("A"), _result("B")]

    results = evaluate([question], fake_retrieve, k=3)

    assert results[0].retrieved == ["doc:A", "doc:B"]
    assert results[0].rank == 2


def test_load_questions_plusieurs_articles_attendus(tmp_path: Path) -> None:
    """Plusieurs articles attendus sont séparés par ";" dans la colonne expected."""
    path = tmp_path / "questions.csv"
    path.write_text(
        "id,question,expected,note\nq01,Question ?,doc:A;doc:B,une note\n", encoding="utf-8"
    )

    assert load_questions(path) == [
        EvalQuestion("q01", "Question ?", frozenset({"doc:A", "doc:B"}), "une note")
    ]


def test_jeu_de_questions_versionne_est_lisible() -> None:
    """Le fichier data/eval/questions.csv se charge et ses identifiants sont uniques."""
    questions = load_questions(DEFAULT_QUESTIONS)

    assert len(questions) >= 20
    assert len({question.id for question in questions}) == len(questions)
    assert all(":" in key for question in questions for key in question.expected)


def test_recording_reranker_garde_les_notes_de_tous_les_candidats(tmp_path: Path) -> None:
    """Toutes les notes sont gardées dans l'ordre vectoriel, même pour deux
    passages de même clé, et le résultat renvoyé est celui du vrai reclassement."""
    question = EvalQuestion("q01", "Question ?", frozenset({"doc:C"}), "")
    candidates = [_result("A"), _result("A"), _result("C")]  # 2 passages "doc:A"

    def fake_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1, 0.9, 0.5]

    def rerank_fn(text: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        return rerank(text, results, {}, fake_scorer, top_k=k)

    logs: list[RerankLog] = []
    ranked = recording_reranker(rerank_fn, logs)("Question ?", candidates, 2)

    assert [r.score for r in ranked] == [0.9, 0.5]
    assert [c["score_reranker"] for c in logs[0].candidates] == [0.1, 0.9, 0.5]
    assert [c["rang_vecteur"] for c in logs[0].candidates] == [1, 2, 3]
    assert logs[0].candidates[2]["cle"] == "doc:C"

    path = tmp_path / "candidats.jsonl"
    write_rerank_logs(path, [question], logs)
    line = json.loads(path.read_text(encoding="utf-8"))
    assert line["id"] == "q01"
    assert line["attendus"] == ["doc:C"]
    assert len(line["candidats"]) == 3

    with pytest.raises(FileExistsError):  # un fichier existant n'est jamais écrasé
        write_rerank_logs(path, [question], logs)


def test_parse_args_option_k() -> None:
    """--k vaut 5 par défaut, se règle, et doit valoir au moins 1."""
    assert parse_args([]).k == 5
    assert parse_args(["--mode", "plein-texte", "--k", "200"]).k == 200
    with pytest.raises(SystemExit):
        parse_args(["--k", "0"])

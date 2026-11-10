"""Tests unitaires de l'aide au choix du seuil de refus (sans base ni modèle)."""

from datetime import date
from pathlib import Path

from fire_safety_rag.evaluation.evaluate import EvalQuestion
from fire_safety_rag.evaluation.threshold import (
    DEFAULT_OFF_CORPUS,
    FOUND,
    MISSED,
    OFF_CORPUS,
    ScoredQuestion,
    candidate_thresholds,
    load_off_corpus,
    refusal_counts,
    score_questions,
)
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult


def _hit(key: str, score: float) -> SearchResult:
    """Construit un SearchResult minimal à partir d'une clé "document_id:article_id"."""
    document_id, article_id = key.split(":")
    chunk = Chunk(
        document_id=document_id,
        article_id=article_id,
        version="",
        valid_from=date(2020, 1, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text="texte",
        page_start=1,
        page_end=1,
    )
    return SearchResult(chunk=chunk, score=score)


def _question(id_: str, expected: frozenset[str]) -> EvalQuestion:
    """Construit une EvalQuestion dont le texte est son identifiant."""
    return EvalQuestion(id=id_, question=id_, expected=expected, note="")


class TestScoreQuestions:
    """Tests de score_questions avec une recherche simulée."""

    def test_groupes_et_meilleur_score(self) -> None:
        """Chaque question reçoit son groupe et le score du premier résultat."""
        answers = {
            "q1": [_hit("erp:GN 1", 0.9), _hit("erp:GN 2", 0.8)],
            "q2": [_hit("erp:MS 1", 0.7)],
            "h1": [_hit("erp:CO 1", 0.6)],
            "h2": [],
        }
        questions = [
            _question("q1", frozenset({"erp:GN 2"})),
            _question("q2", frozenset({"erp:GN 1"})),
            _question("h1", frozenset()),
            _question("h2", frozenset()),
        ]
        scored = score_questions(questions, lambda question, k: answers[question])
        assert scored == [
            ScoredQuestion("q1", FOUND, 0.9),
            ScoredQuestion("q2", MISSED, 0.7),
            ScoredQuestion("h1", OFF_CORPUS, 0.6),
            ScoredQuestion("h2", OFF_CORPUS, 0.0),
        ]


class TestRefusalCounts:
    """Tests de refusal_counts."""

    SCORED = [
        ScoredQuestion("q1", FOUND, 0.90),
        ScoredQuestion("q2", FOUND, 0.80),
        ScoredQuestion("q3", MISSED, 0.75),
        ScoredQuestion("h1", OFF_CORPUS, 0.70),
        ScoredQuestion("h2", OFF_CORPUS, 0.85),
    ]

    def test_comptage_par_groupe(self) -> None:
        """Refusée si score < seuil ; un score égal au seuil n'est pas refusé."""
        assert refusal_counts(self.SCORED, 0.80) == {
            FOUND: (0, 2),
            MISSED: (1, 1),
            OFF_CORPUS: (1, 2),
        }

    def test_groupe_vide(self) -> None:
        """Un groupe sans question donne (0, 0)."""
        counts = refusal_counts([ScoredQuestion("q1", FOUND, 0.9)], 0.5)
        assert counts[MISSED] == (0, 0)


class TestCandidateThresholds:
    """Tests de candidate_thresholds."""

    def test_couvre_les_scores(self) -> None:
        """Les seuils vont du plus bas score arrondi en dessous au plus haut arrondi au-dessus."""
        scored = [ScoredQuestion("a", FOUND, 0.812), ScoredQuestion("b", FOUND, 0.836)]
        assert candidate_thresholds(scored) == [0.81, 0.82, 0.83, 0.84]

    def test_aucune_question(self) -> None:
        """Sans question, aucun seuil."""
        assert candidate_thresholds([]) == []


class TestLoadOffCorpus:
    """Tests de load_off_corpus."""

    def test_lecture(self, tmp_path: Path) -> None:
        """Les questions sont lues sans article attendu."""
        path = tmp_path / "hors.csv"
        path.write_text("id,question,note\nh01,Question ?,éloigné\n", encoding="utf-8")
        assert load_off_corpus(path) == [
            EvalQuestion(id="h01", question="Question ?", expected=frozenset(), note="éloigné")
        ]

    def test_fichier_du_projet(self) -> None:
        """Le fichier du projet contient 10 questions aux identifiants uniques en "h"."""
        questions = load_off_corpus(DEFAULT_OFF_CORPUS)
        ids = [q.id for q in questions]
        assert len(ids) == 10
        assert len(set(ids)) == 10
        assert all(id_.startswith("h") for id_ in ids)

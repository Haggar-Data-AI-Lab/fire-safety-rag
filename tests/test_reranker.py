"""Tests unitaires du reranker, avec une fausse notation (aucun modèle téléchargé)."""

from datetime import date
from typing import Any, cast

import pytest

from fire_safety_rag.evaluation.evaluate import make_retriever
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.reranking.reranker import Scorer, format_candidate, rerank
from fire_safety_rag.storage.store import SearchResult

TITLES = {"erp": "Règlement ERP", "travail": "Code du travail"}


def _result(document_id: str, article_id: str, score: float, text: str = "texte") -> SearchResult:
    """Construit un SearchResult minimal."""
    chunk = Chunk(
        document_id=document_id,
        article_id=article_id,
        version="",
        valid_from=date(2020, 1, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text=text,
        page_start=1,
        page_end=1,
    )
    return SearchResult(chunk=chunk, score=score)


CANDIDATES = [
    _result("erp", "CTS 10", 0.88, "sorties des chapiteaux"),
    _result("erp", "PA 7", 0.87, "sorties des tribunes"),
    _result("travail", "R4227-5", 0.85, "sorties des lieux de travail"),
]


def _scores_by_keyword(
    keyword_scores: dict[str, float],
) -> tuple[Scorer, list[tuple[str, str]]]:
    """Fausse notation selon un mot-clé du passage ; enregistre les couples reçus."""
    seen: list[tuple[str, str]] = []

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        seen.extend(pairs)
        return [
            next((s for k, s in keyword_scores.items() if k in passage), 0.0)
            for _, passage in pairs
        ]

    return score, seen


class TestFormatCandidate:
    """Tests de format_candidate."""

    def test_article_et_annexe(self) -> None:
        """L'en-tête porte le titre et l'article ; une annexe garde son nom."""
        assert format_candidate(CANDIDATES[2].chunk, "Code du travail") == (
            "Code du travail — Article R4227-5\nsorties des lieux de travail"
        )
        annex = _result("ssiap", "Annexe I", 0.8, "contenu").chunk
        assert format_candidate(annex, "Arrêté SSIAP") == "Arrêté SSIAP — Annexe I\ncontenu"


class TestRerank:
    """Tests de rerank."""

    def test_reclasse_et_remplace_le_score(self) -> None:
        """Le meilleur selon le reranker passe en tête, avec son propre score."""
        score_fn, _ = _scores_by_keyword({"travail": 0.95, "chapiteaux": 0.10, "tribunes": 0.05})
        ranked = rerank("Sorties d'un bureau ?", CANDIDATES, TITLES, score_fn, top_k=2)
        assert [r.chunk.article_id for r in ranked] == ["R4227-5", "CTS 10"]
        assert [r.score for r in ranked] == [0.95, 0.10]

    def test_couples_envoyes(self) -> None:
        """Chaque couple associe la question au passage avec son en-tête."""
        score_fn, seen = _scores_by_keyword({})
        rerank("Q ?", CANDIDATES[:1], TITLES, score_fn)
        assert seen == [("Q ?", "Règlement ERP — Article CTS 10\nsorties des chapiteaux")]

    def test_egalite_garde_l_ordre_initial(self) -> None:
        """À score égal, l'ordre de la recherche vectorielle est conservé."""
        score_fn, _ = _scores_by_keyword({})
        ranked = rerank("Q ?", CANDIDATES, TITLES, score_fn, top_k=3)
        assert [r.chunk.article_id for r in ranked] == ["CTS 10", "PA 7", "R4227-5"]

    def test_titre_absent(self) -> None:
        """Un document sans titre connu est présenté avec son document_id."""
        score_fn, seen = _scores_by_keyword({})
        rerank("Q ?", [_result("inconnu", "1", 0.9)], TITLES, score_fn)
        assert seen[0][1].startswith("inconnu — Article 1")

    def test_aucun_candidat(self) -> None:
        """Sans candidat, aucune notation et liste vide."""
        score_fn, seen = _scores_by_keyword({})
        assert rerank("Q ?", [], TITLES, score_fn) == []
        assert seen == []

    def test_mode_evaluation_sans_reranker(self) -> None:
        """Le mode vecteurs+reranker de l'évaluation exige une fonction de reclassement."""
        with pytest.raises(ValueError, match="exige une fonction de reclassement"):
            make_retriever(cast(Any, None), "vecteurs+reranker", lambda q: [], "modele")

    def test_nombre_de_scores_incoherent(self) -> None:
        """Une notation qui ne renvoie pas un score par passage lève ValueError."""
        with pytest.raises(ValueError, match="2 scores reçus pour 3 passages"):
            rerank("Q ?", CANDIDATES, TITLES, lambda pairs: [0.1, 0.2])

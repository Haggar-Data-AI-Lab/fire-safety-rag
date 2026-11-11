"""Tests de la simulation hors ligne du reranker (sans modèle ni base)."""

from pathlib import Path

import pytest

from fire_safety_rag.evaluation.candidates import (
    Candidate,
    CandidateLog,
    best_vector_rank,
    load_logs,
    simulate,
    summarize_budget,
)
from fire_safety_rag.evaluation.evaluate import EvalQuestion, RerankLog, write_rerank_logs

# Rang vectoriel 1 à 4 ; le reranker préfère C (rang 3), puis D (rang 4)
CANDIDATES = [
    Candidate(1, "doc:A", 0.2),
    Candidate(2, "doc:B", 0.1),
    Candidate(3, "doc:C", 0.9),
    Candidate(4, "doc:D", 0.5),
]


def test_simulate_ne_garde_que_les_premiers_candidats_vectoriels() -> None:
    """Avec 2 candidats, C et D (rangs 3 et 4) ne sont pas relus par le reranker."""
    assert simulate(CANDIDATES, budget=2, k=5) == ["doc:A", "doc:B"]
    assert simulate(CANDIDATES, budget=4, k=2) == ["doc:C", "doc:D"]


def test_simulate_a_note_egale_garde_l_ordre_vectoriel() -> None:
    """Deux notes égales : le meilleur rang vectoriel passe devant."""
    tied = [Candidate(1, "doc:A", 0.5), Candidate(2, "doc:B", 0.5)]
    assert simulate(list(reversed(tied)), budget=2) == ["doc:A", "doc:B"]


def test_simulate_refuse_plus_de_candidats_qu_enregistres() -> None:
    """On ne peut pas simuler 10 candidats avec 4 notes enregistrées."""
    with pytest.raises(ValueError):
        simulate(CANDIDATES, budget=10)


def test_budget_egal_a_k_ne_change_pas_hit_at_k() -> None:
    """Avec k candidats, le reranker ne fait que réordonner le top k vectoriel :
    Hit@k est celui de la recherche vectorielle seule ; le temps est au prorata."""
    log = CandidateLog("q01", frozenset({"doc:B"}), seconds=40.0, candidates=CANDIDATES)

    summary = summarize_budget([log], budget=2, k=2)

    assert summary["Hit@2"] == 1.0  # B est dans le top 2 vectoriel
    assert summary["Hit@1"] == 0.0  # mais le reranker place A devant
    assert summary["Temps estimé"] == pytest.approx(20.0)  # 40 s × 2 / 4


def test_best_vector_rank() -> None:
    """Meilleur rang d'un attendu parmi les candidats, None s'il n'y est pas."""
    assert best_vector_rank(CandidateLog("q", frozenset({"doc:D", "doc:B"}), 1.0, CANDIDATES)) == 2
    assert best_vector_rank(CandidateLog("q", frozenset({"doc:Z"}), 1.0, CANDIDATES)) is None


def test_load_logs_relit_le_format_de_write_rerank_logs(tmp_path: Path) -> None:
    """load_logs relit exactement ce qu'écrit evaluate.write_rerank_logs."""
    question = EvalQuestion("q01", "Question ?", frozenset({"doc:C"}), "")
    rerank_log = RerankLog(
        question="Question ?",
        seconds=12.5,
        candidates=[
            {"rang_vecteur": 1, "cle": "doc:A", "score_vecteur": 0.86, "score_reranker": 0.2},
            {"rang_vecteur": 2, "cle": "doc:C", "score_vecteur": 0.85, "score_reranker": 0.9},
        ],
    )
    path = tmp_path / "candidats.jsonl"
    write_rerank_logs(path, [question], [rerank_log])

    assert load_logs(path) == [
        CandidateLog(
            "q01",
            frozenset({"doc:C"}),
            12.5,
            [Candidate(1, "doc:A", 0.2), Candidate(2, "doc:C", 0.9)],
        )
    ]

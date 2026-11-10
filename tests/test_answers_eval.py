"""Tests unitaires de l'évaluation des réponses (réponses simulées, aucun appel)."""

import csv
from dataclasses import replace
from datetime import date
from pathlib import Path

from fire_safety_rag.evaluation.answers import (
    AnswerRecord,
    append_record,
    read_records,
    record_answer,
    summarize,
    write_review_sheet,
)
from fire_safety_rag.evaluation.evaluate import EvalQuestion
from fire_safety_rag.evaluation.threshold import FOUND, MISSED, OFF_CORPUS
from fire_safety_rag.generation.answer import ANSWERED, REFUSED_BY_LLM, REFUSED_BY_SCORE, Answer
from fire_safety_rag.generation.citations import check_citations
from fire_safety_rag.generation.llm import LLMResponse
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult

TITLES = {"erp": "Règlement ERP"}


def _result(article_id: str, score: float = 0.88) -> SearchResult:
    """Construit un SearchResult minimal du document "erp"."""
    chunk = Chunk(
        document_id="erp",
        article_id=article_id,
        version="",
        valid_from=date(2020, 3, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text="texte",
        page_start=12,
        page_end=12,
    )
    return SearchResult(chunk=chunk, score=score)


RESULTS = (_result("MS 21"), _result("GN 1", 0.86))


def _answered(text: str) -> Answer:
    """Réponse du LLM sur RESULTS, contrôlée comme dans la vraie chaîne."""
    response = LLMResponse(text, "claude-haiku-5-5", "end_turn", 3000, 400)
    checked = check_citations(text, RESULTS, TITLES)
    return Answer("Q ?", ANSWERED, RESULTS, 0.84, response, checked)


def _question(expected: frozenset[str], id_: str = "q01") -> EvalQuestion:
    return EvalQuestion(id=id_, question="Q ?", expected=expected, note="")


def _record(group: str, status: str, **changes: object) -> AnswerRecord:
    """Ligne enregistrée minimale, modifiable champ par champ."""
    base = AnswerRecord(
        id="q",
        group=group,
        question="Q ?",
        expected=["erp:GN 1"] if group != OFF_CORPUS else [],
        retrieved=[],
        top_score=0.9,
        status=status,
        text="",
        cited=[],
        sources=[],
        invalid_numbers=[],
        uncited=False,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestRecordAnswer:
    """Tests de record_answer."""

    def test_reponse_trouvee_avec_bon_article(self) -> None:
        """Groupe "trouvée", articles cités et coût repris de la réponse."""
        record = record_answer(_question(frozenset({"erp:GN 1"})), _answered("Oui [2]."))
        assert record.group == FOUND
        assert record.retrieved == ["erp:MS 21", "erp:GN 1"]
        assert record.cited == ["erp:GN 1"]
        assert record.sources == [
            "[2] Règlement ERP — Article GN 1 (page 12, en vigueur depuis le 01/03/2020)"
        ]
        assert record.cost_usd == 0.0005

    def test_manquee(self) -> None:
        """Article attendu absent des passages : groupe "manquée"."""
        record = record_answer(_question(frozenset({"erp:CO 1"})), _answered("Oui [1]."))
        assert record.group == MISSED

    def test_refus_par_le_score_sans_appel(self) -> None:
        """Sans appel au LLM : texte vide, aucun token, aucun coût."""
        answer = Answer("Q ?", REFUSED_BY_SCORE, (_result("MS 21", 0.80),), 0.84)
        record = record_answer(_question(frozenset(), "h01"), answer)
        assert record.group == OFF_CORPUS
        assert (record.text, record.input_tokens, record.cost_usd) == ("", 0, 0.0)
        assert record.top_score == 0.80


class TestSummarize:
    """Tests de summarize."""

    def test_mesures(self) -> None:
        """Les mesures comptent les bons cas dans les bons groupes."""
        records = [
            _record(OFF_CORPUS, REFUSED_BY_SCORE),
            _record(OFF_CORPUS, REFUSED_BY_LLM, input_tokens=3000, cost_usd=0.0004),
            _record(OFF_CORPUS, ANSWERED, input_tokens=3000, cost_usd=0.0004),
            _record(FOUND, ANSWERED, cited=["erp:GN 1"], input_tokens=3000, cost_usd=0.0005),
            _record(FOUND, REFUSED_BY_LLM, input_tokens=3000, cost_usd=0.0003),
            _record(MISSED, ANSWERED, cited=["erp:MS 21"], invalid_numbers=[7], input_tokens=1),
        ]
        summary = summarize(records)
        assert summary["Refus justifiés (hors corpus refusées)"] == "2 / 3"
        assert summary["  dont refus par le score"] == "1 / 3"
        assert summary["Refus à tort (trouvées refusées)"] == "1 / 2"
        assert summary["Manquées refusées"] == "0 / 1"
        assert summary["Bon article cité (réponses du corpus)"] == "1 / 2"
        assert summary["Réponses avec citation inconnue"] == "1 / 6"
        assert summary["Appels au LLM"] == "5"
        assert summary["Coût total"] == "0.00160 $"
        assert summary["Coût moyen par appel"] == "0.00032 $"

    def test_aucun_appel(self) -> None:
        """Sans appel au LLM, le coût moyen n'est pas calculé."""
        assert summarize([_record(OFF_CORPUS, REFUSED_BY_SCORE)])["Coût moyen par appel"] == "—"


class TestFichiers:
    """Tests de l'enregistrement et de la fiche de relecture."""

    def test_aller_retour_jsonl(self, tmp_path: Path) -> None:
        """Une ligne écrite puis relue est identique (accents compris)."""
        path = tmp_path / "reponses.jsonl"
        records = [_record(FOUND, ANSWERED, text="Réponse [1]."), _record(MISSED, ANSWERED)]
        for record in records:
            append_record(path, record)
        assert read_records(path) == records

    def test_fiche_de_relecture(self, tmp_path: Path) -> None:
        """La fiche omet les refus par le score et prévoit les colonnes à remplir."""
        path = tmp_path / "relecture.csv"
        records = [
            _record(OFF_CORPUS, REFUSED_BY_SCORE, id="h10"),
            _record(FOUND, ANSWERED, id="q03", text="Oui [1].", sources=["[1] a", "[2] b"]),
        ]
        write_review_sheet(path, records)
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f, delimiter=";"))
        assert rows[0][-3:] == ["fidele", "repond_question", "commentaire"]
        assert len(rows) == 2
        assert rows[1][:3] == ["q03", FOUND, ANSWERED]
        assert rows[1][6] == "[1] a\n[2] b"
        assert rows[1][-3:] == ["", "", ""]

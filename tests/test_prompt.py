"""Tests unitaires de la construction du prompt (sans LLM ni base)."""

from datetime import date

import pytest

from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, build_prompt
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult


def _result(
    article_id: str,
    text: str,
    document_id: str = "erp",
    pages: tuple[int, int] = (12, 12),
    valid_until: date | None = None,
) -> SearchResult:
    """Construit un SearchResult minimal pour les tests."""
    chunk = Chunk(
        document_id=document_id,
        article_id=article_id,
        version="VERSION EN VIGUEUR DEPUIS LE 01/03/2020",
        valid_from=date(2020, 3, 1),
        valid_until=valid_until,
        history=(),
        chunk_index=0,
        text=text,
        page_start=pages[0],
        page_end=pages[1],
    )
    return SearchResult(chunk=chunk, score=0.8)


TITLES = {"erp": "Règlement ERP", "ssiap": "Arrêté SSIAP"}


class TestBuildPrompt:
    """Tests de build_prompt."""

    def test_numerotation_dans_l_ordre(self) -> None:
        """Les passages sont numérotés [1]…[n] dans l'ordre du classement."""
        results = [_result("MS 21", "Texte A"), _result("GN 1", "Texte B")]
        user = build_prompt("Question ?", results, TITLES).user
        assert user.index("[1] Règlement ERP — Article MS 21") < user.index(
            "[2] Règlement ERP — Article GN 1"
        )
        assert "[3]" not in user

    def test_en_tete_complet(self) -> None:
        """L'en-tête contient titre, article, pages et dates de vigueur."""
        results = [_result("MS 21", "Texte", pages=(412, 413), valid_until=date(2027, 6, 1))]
        user = build_prompt("Question ?", results, TITLES).user
        assert (
            "[1] Règlement ERP — Article MS 21 (pages 412-413, "
            "en vigueur depuis le 01/03/2020, jusqu'au 01/06/2027)\nTexte"
        ) in user

    def test_page_unique_et_annexe(self) -> None:
        """Une page seule s'écrit "page N" ; une annexe garde son titre sans "Article"."""
        results = [_result("Annexe I", "Contenu", document_id="ssiap")]
        user = build_prompt("Question ?", results, TITLES).user
        assert "[1] Arrêté SSIAP — Annexe I (page 12, en vigueur depuis le 01/03/2020)" in user

    def test_passages_entre_balises_puis_question(self) -> None:
        """Les passages sont entre balises, suivis de la question nettoyée."""
        user = build_prompt("  Quelle largeur ?  ", [_result("CO 1", "Texte")], TITLES).user
        assert user.startswith("<passages>\n[1]")
        assert user.endswith("</passages>\n\nQuestion : Quelle largeur ?")

    def test_titre_absent_repli_sur_document_id(self) -> None:
        """Un document sans titre connu est affiché avec son document_id."""
        user = build_prompt("Question ?", [_result("R4227-28", "Texte", "travail")], TITLES).user
        assert "[1] travail — Article R4227-28" in user

    def test_consigne_systeme_contient_la_phrase_de_refus(self) -> None:
        """La consigne système impose la phrase de refus exacte."""
        prompt = build_prompt("Question ?", [_result("CO 1", "Texte")], TITLES)
        assert REFUSAL_SENTENCE in prompt.system
        assert REFUSAL_SENTENCE not in prompt.user

    def test_aucun_passage(self) -> None:
        """Sans passage, on refuse avant le prompt : ValueError."""
        with pytest.raises(ValueError, match="Aucun passage"):
            build_prompt("Question ?", [], TITLES)

    def test_question_vide(self) -> None:
        """Une question vide ou blanche lève ValueError."""
        with pytest.raises(ValueError, match="question est vide"):
            build_prompt("   ", [_result("CO 1", "Texte")], TITLES)

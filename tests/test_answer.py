"""Tests unitaires du parcours question → réponse (fausses recherche et LLM, aucun coût)."""

from datetime import date

from fire_safety_rag.generation.answer import (
    ANSWERED,
    DISCLAIMER,
    INCOMPLETE,
    REFUSED_BY_LLM,
    REFUSED_BY_SCORE,
    Answer,
    answer_question,
    format_answer,
)
from fire_safety_rag.generation.llm import LLMResponse
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, Prompt
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult

TITLES = {"erp": "Règlement ERP"}


def _result(article_id: str, score: float) -> SearchResult:
    """Construit un SearchResult minimal (page 12, en vigueur depuis le 01/03/2020)."""
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


RESULTS = [_result("MS 21", 0.88), _result("GN 1", 0.86)]


class FakeLLM:
    """Faux LLM : renvoie un texte fixé et compte les appels."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.prompts: list[Prompt] = []

    def __call__(self, prompt: Prompt) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(self.text, "claude-haiku-5-5", self.stop_reason, 3000, 400)


def _ask(llm: FakeLLM, results: list[SearchResult] = RESULTS, threshold: float = 0.84) -> Answer:
    """Pose une question avec une recherche simulée qui renvoie `results`."""
    return answer_question("Question ?", lambda question, k: results, TITLES, llm, threshold)


class TestAnswerQuestion:
    """Tests de answer_question."""

    def test_refus_par_le_score_sans_appel(self) -> None:
        """Meilleur score sous le seuil : refus, et le LLM n'est pas appelé."""
        llm = FakeLLM("ne doit pas servir")
        answer = _ask(llm, [_result("MS 21", 0.83)])
        assert answer.status == REFUSED_BY_SCORE
        assert llm.prompts == []
        assert answer.response is None

    def test_aucun_passage(self) -> None:
        """Sans aucun passage : refus par le score, sans appel."""
        llm = FakeLLM("ne doit pas servir")
        assert _ask(llm, []).status == REFUSED_BY_SCORE
        assert llm.prompts == []

    def test_score_egal_au_seuil_appelle_le_llm(self) -> None:
        """Un score égal au seuil n'est pas refusé."""
        llm = FakeLLM("Oui [1].")
        assert _ask(llm, [_result("MS 21", 0.84)]).status == ANSWERED
        assert len(llm.prompts) == 1

    def test_reponse_normale(self) -> None:
        """Réponse complète : sources retrouvées ; les passages sont dans le prompt."""
        llm = FakeLLM("Deux sorties [2].")
        answer = _ask(llm)
        assert answer.status == ANSWERED
        assert answer.checked is not None
        assert [s.number for s in answer.checked.sources] == [2]
        assert "[2] Règlement ERP — Article GN 1" in llm.prompts[0].user

    def test_refus_par_le_llm(self) -> None:
        """Le LLM répond par la phrase de refus : statut refus_llm."""
        assert _ask(FakeLLM(REFUSAL_SENTENCE)).status == REFUSED_BY_LLM

    def test_reponse_coupee(self) -> None:
        """Une réponse coupée est incomplète, même si elle cite des passages."""
        assert _ask(FakeLLM("Début [1]…", "max_tokens")).status == INCOMPLETE


class TestFormatAnswer:
    """Tests de format_answer."""

    def test_refus_par_le_score(self) -> None:
        """Le refus par le score donne la raison et le seuil."""
        text = format_answer(_ask(FakeLLM(""), [_result("MS 21", 0.8031)]))
        assert text.startswith(REFUSAL_SENTENCE)
        assert "meilleur score 0.8031 < seuil 0.84" in text
        assert text.endswith(DISCLAIMER)

    def test_reponse_avec_sources(self) -> None:
        """La réponse est suivie de ses sources, puis de l'avertissement juridique."""
        text = format_answer(_ask(FakeLLM("Deux sorties [2].")))
        assert text == (
            "Deux sorties [2].\n\nSources :\n"
            "  [2] Règlement ERP — Article GN 1 (page 12, en vigueur depuis le 01/03/2020)\n\n"
            + DISCLAIMER
        )

    def test_avertissements(self) -> None:
        """Citation inconnue, réponse sans source et réponse coupée sont signalées."""
        assert "⚠ Citation [7] inconnue" in format_answer(_ask(FakeLLM("Oui [7].")))
        assert "⚠ Aucune source citée" in format_answer(_ask(FakeLLM("Oui.")))
        text = format_answer(_ask(FakeLLM("Début [1]", "max_tokens")))
        assert "⚠ Réponse incomplète (coupée par la limite de longueur)." in text

    def test_refus_par_le_llm_sans_avertissement(self) -> None:
        """Un refus du LLM ne déclenche pas l'avertissement « aucune source »."""
        text = format_answer(_ask(FakeLLM(REFUSAL_SENTENCE)))
        assert "⚠" not in text
        assert text.endswith(DISCLAIMER)

    def test_details(self) -> None:
        """L'option détails liste les passages et leur score."""
        text = format_answer(_ask(FakeLLM("Oui [1].")), details=True)
        assert (
            "Passages retrouvés (score) :\n  [1] 0.8800  erp:MS 21\n  [2] 0.8600  erp:GN 1" in text
        )

"""Tests de l'API HTTP (fausses recherche et LLM : ni base, ni modèle, ni coût)."""

from collections.abc import Callable, Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient

from fire_safety_rag.api.main import Answerer, app, get_answerer
from fire_safety_rag.generation.answer import DISCLAIMER, Answer, answer_question
from fire_safety_rag.generation.llm import LLMError, LLMResponse
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, Prompt
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult

TITLES = {"erp": "Règlement ERP"}


def _result(article_id: str, score: float) -> SearchResult:
    """SearchResult minimal du document "erp"."""
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


def _answerer(llm_text: str, score: float = 0.88, stop_reason: str = "end_turn") -> Answerer:
    """Vraie chaîne answer_question, avec recherche et LLM simulés."""

    def llm(prompt: Prompt) -> LLMResponse:
        return LLMResponse(llm_text, "claude-haiku-5-5", stop_reason, 3000, 400)

    def answer(question: str) -> Answer:
        results = [_result("MS 21", score), _result("GN 1", score - 0.02)]
        return answer_question(question, lambda q, k: results, TITLES, llm, threshold=0.84)

    return answer


@pytest.fixture
def client() -> Iterator[Callable[[Answerer], TestClient]]:
    """Client de test ; la fonction de réponse est choisie par chaque test."""

    def make(answerer: Answerer) -> TestClient:
        app.dependency_overrides[get_answerer] = lambda: answerer
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_health() -> None:
    """GET /health répond sans dépendance."""
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"statut": "ok"}


def test_reponse_avec_sources(client: Callable[[Answerer], TestClient]) -> None:
    """Une réponse qui cite [1] renvoie la source 1 et l'avertissement juridique."""
    response = client(_answerer("Il faut un extincteur [1]."))
    body = response.post("/questions", json={"question": "Combien d'extincteurs ?"}).json()

    assert body["statut"] == "reponse"
    assert body["reponse"] == "Il faut un extincteur [1]."
    assert [source["numero"] for source in body["sources"]] == [1]
    assert "MS 21" in body["sources"][0]["reference"]
    assert body["avertissements"] == []
    assert body["meilleur_score"] == pytest.approx(0.88)
    assert body["avertissement_juridique"] == DISCLAIMER


def test_refus_par_le_score(client: Callable[[Answerer], TestClient]) -> None:
    """Score sous le seuil : phrase de refus, aucune source."""
    response = client(_answerer("ne doit pas servir", score=0.80))
    body = response.post("/questions", json={"question": "Le permis de conduire ?"}).json()

    assert body["statut"] == "refus_score"
    assert body["reponse"] == REFUSAL_SENTENCE
    assert body["sources"] == []


def test_avertissement_citation_inconnue(client: Callable[[Answerer], TestClient]) -> None:
    """Un numéro cité qui ne correspond à aucun passage est signalé."""
    response = client(_answerer("Voir [1] et [7]."))
    body = response.post("/questions", json={"question": "Question ?"}).json()

    assert body["avertissements"] == ["Citation [7] inconnue : réponse à vérifier."]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_question_invalide_refusee(client: Callable[[Answerer], TestClient], question: str) -> None:
    """Question vide, blanche ou trop longue : 422, sans appeler la chaîne."""

    def never(question: str) -> Answer:
        raise AssertionError("la chaîne ne doit pas être appelée")

    response = client(never).post("/questions", json={"question": question})
    assert response.status_code == 422


def test_echec_du_llm_donne_503(client: Callable[[Answerer], TestClient]) -> None:
    """Une LLMError devient une 503 au message générique (pas de détail interne)."""

    def failing(question: str) -> Answer:
        raise LLMError("clé ANTHROPIC_API_KEY absente")

    response = client(failing).post("/questions", json={"question": "Question ?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Le service de génération est indisponible."}


def test_chaine_non_branchee_donne_503() -> None:
    """Sans remplacement, l'étape 1 répond 503 : la chaîne n'est pas encore branchée."""
    response = TestClient(app).post("/questions", json={"question": "Question ?"})
    assert response.status_code == 503

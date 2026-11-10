"""Tests unitaires de l'appel au LLM, avec un faux client (aucun appel réseau, aucun coût)."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage
from pydantic import SecretStr

from fire_safety_rag.config import Settings
from fire_safety_rag.generation.llm import LLMError, estimate_cost, generate, make_client
from fire_safety_rag.generation.prompt import Prompt

PROMPT = Prompt(system="consigne", user="<passages>…</passages>\n\nQuestion : ?")
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Configuration isolée : ni .env du projet, ni variable d'environnement LLM."""
    monkeypatch.chdir(tmp_path)
    for name in ("ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_EFFORT", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    yield Settings()


def _message(
    content: list[Any], stop_reason: str = "end_turn", model: str = "claude-haiku-5-5"
) -> Message:
    """Construit une réponse de l'API telle que le SDK la renverrait."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=3000, output_tokens=400),
    )


class FakeMessages:
    """Remplace client.messages : enregistre l'appel, renvoie ou lève ce qu'on lui donne."""

    def __init__(self, outcome: Message | Exception) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Message:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    """Faux client Anthropic : seul l'attribut messages est utilisé."""

    def __init__(self, outcome: Message | Exception) -> None:
        self.messages = FakeMessages(outcome)


def _client(fake: FakeClient) -> anthropic.Anthropic:
    """Présente le faux client avec le type attendu par generate."""
    return cast(anthropic.Anthropic, fake)


class TestGenerate:
    """Tests de generate."""

    def test_reponse_complete_et_parametres_envoyes(self, settings: Settings) -> None:
        """Le texte, les tokens et les paramètres de l'appel sont corrects."""
        fake = FakeClient(_message([TextBlock(type="text", text="Réponse [1].", citations=None)]))
        response = generate(PROMPT, _client(fake), settings)

        assert response.text == "Réponse [1]."
        assert response.complete
        assert (response.input_tokens, response.output_tokens) == (3000, 400)
        assert fake.messages.kwargs == {
            "model": "claude-haiku-5-5",
            "max_tokens": 2000,
            "system": "consigne",
            "messages": [{"role": "user", "content": PROMPT.user}],
            "output_config": {"effort": "low"},
        }

    def test_reflexion_ignoree_et_blocs_concatenes(self, settings: Settings) -> None:
        """Les blocs de réflexion sont ignorés ; les blocs de texte sont mis bout à bout."""
        content = [
            ThinkingBlock(type="thinking", thinking="", signature="sig"),
            TextBlock(type="text", text="Début ", citations=None),
            TextBlock(type="text", text="fin [2].", citations=None),
        ]
        response = generate(PROMPT, _client(FakeClient(_message(content))), settings)
        assert response.text == "Début fin [2]."

    def test_reponse_coupee_incomplete(self, settings: Settings) -> None:
        """Une réponse arrêtée par max_tokens n'est pas complète."""
        message = _message([TextBlock(type="text", text="Début…", citations=None)], "max_tokens")
        response = generate(PROMPT, _client(FakeClient(message)), settings)
        assert not response.complete
        assert response.stop_reason == "max_tokens"

    def test_journal_des_couts(self, settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
        """Chaque appel écrit une ligne avec les tokens et le coût estimé."""
        message = _message([TextBlock(type="text", text="ok", citations=None)])
        with caplog.at_level(logging.INFO, logger="fire_safety_rag.generation.llm"):
            generate(PROMPT, _client(FakeClient(message)), settings)
        assert "entrée=3000 tokens, sortie=400 tokens, coût estimé=0.00050 $" in caplog.text

    def test_cle_refusee(self, settings: Settings) -> None:
        """Une clé invalide (401) donne un message clair."""
        error = anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=_REQUEST), body=None
        )
        with pytest.raises(LLMError, match="Clé API refusée"):
            generate(PROMPT, _client(FakeClient(error)), settings)

    def test_delai_depasse(self, settings: Settings) -> None:
        """Un délai dépassé (après les nouvelles tentatives du SDK) donne un message clair."""
        error = anthropic.APITimeoutError(request=_REQUEST)
        with pytest.raises(LLMError, match="délai dépassé"):
            generate(PROMPT, _client(FakeClient(error)), settings)


class TestCost:
    """Tests de estimate_cost."""

    def test_cout_haiku(self) -> None:
        """3000 tokens en entrée et 400 en sortie avec Haiku 5.5 : 0,0005 $."""
        assert estimate_cost("claude-haiku-5-5", 3000, 400) == pytest.approx(0.0005)

    def test_modele_inconnu(self) -> None:
        """Un modèle absent de la grille donne None."""
        assert estimate_cost("modele-inconnu", 3000, 400) is None


class TestMakeClient:
    """Tests de make_client."""

    def test_sans_cle(self, settings: Settings) -> None:
        """Sans clé, on obtient un message clair, sans appel réseau."""
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            make_client(settings)

    def test_cle_vide(self, settings: Settings) -> None:
        """Une clé vide (ligne ANTHROPIC_API_KEY= recopiée telle quelle) compte comme absente."""
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            make_client(settings.model_copy(update={"anthropic_api_key": SecretStr("  ")}))

    def test_reglages_transmis(self, settings: Settings) -> None:
        """Le client reçoit le délai d'expiration et le nombre de nouvelles tentatives."""
        configured = settings.model_copy(
            update={
                "anthropic_api_key": SecretStr("cle-de-test"),
                "llm_timeout_seconds": 30.0,
                "llm_max_retries": 1,
            }
        )
        client = make_client(configured)
        assert client.timeout == 30.0
        assert client.max_retries == 1

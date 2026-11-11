"""Tests du client de l'API (faux serveur httpx.MockTransport : ni API, ni coût)."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from fire_safety_rag.ui.client import ApiError, Source, ask_api

API_URL = "http://api.test"

ANSWER_BODY: dict[str, Any] = {
    "statut": "reponse",
    "reponse": "Il faut un extincteur [1].",
    "sources": [{"numero": 1, "reference": "Code du travail — Article R4227-29"}],
    "avertissements": [],
    "meilleur_score": 0.857,
    "avertissement_juridique": "Ces informations ne constituent pas un avis juridique.",
}


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Faux serveur : `handler` reçoit chaque requête et fabrique la réponse."""
    return httpx.MockTransport(handler)


def test_reponse_normale() -> None:
    """La question est envoyée en JSON sur POST /questions ; la réponse est lue."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=ANSWER_BODY)

    answer = ask_api("Combien d'extincteurs ?", API_URL, transport=_transport(handler))

    assert sent[0].method == "POST"
    assert sent[0].url == f"{API_URL}/questions"
    assert sent[0].read() == b'{"question":"Combien d\'extincteurs ?"}'
    assert answer.statut == "reponse"
    assert answer.sources == (Source(1, "Code du travail — Article R4227-29"),)
    assert not answer.is_refusal


def test_refus_reconnu() -> None:
    """Un refus (par le score) est signalé par is_refusal."""
    body = {**ANSWER_BODY, "statut": "refus_score", "sources": []}
    answer = ask_api("Q ?", API_URL, transport=_transport(lambda r: httpx.Response(200, json=body)))

    assert answer.is_refusal
    assert answer.sources == ()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(422, json={"detail": []}), "Question invalide"),
        (
            httpx.Response(503, json={"detail": "Base de données indisponible."}),
            "Service indisponible : Base de données indisponible.",
        ),
        (httpx.Response(500, text="erreur"), r"Erreur inattendue de l'API \(code 500\)"),
        (httpx.Response(200, json={"statut": "reponse"}), "Réponse inattendue"),
        (httpx.Response(200, text="pas du JSON"), "Réponse inattendue"),
    ],
)
def test_erreurs_http(response: httpx.Response, message: str) -> None:
    """Chaque réponse d'erreur devient un message clair en français."""
    with pytest.raises(ApiError, match=message):
        ask_api("Q ?", API_URL, transport=_transport(lambda r: response))


def test_api_eteinte() -> None:
    """API injoignable : le message rappelle la commande pour la lancer."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusée", request=request)

    with pytest.raises(ApiError, match="uvicorn fire_safety_rag.api.main:app"):
        ask_api("Q ?", API_URL, transport=_transport(handler))


def test_delai_depasse() -> None:
    """Délai dépassé : message dédié, distinct de l'API éteinte."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("trop long", request=request)

    with pytest.raises(ApiError, match="pas répondu à temps"):
        ask_api("Q ?", API_URL, transport=_transport(handler))

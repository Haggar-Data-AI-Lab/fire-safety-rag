"""Client HTTP de l'API, utilisé par l'interface Streamlit.

L'interface n'importe pas la chaîne RAG : elle passe par l'API, comme
n'importe quel autre client. Ce module ne dépend pas de Streamlit, pour être
testé sans lui (faux serveur httpx.MockTransport, ni API ni coût).
"""

from dataclasses import dataclass
from typing import Any

import httpx

# Délai d'attente de la réponse : l'API peut attendre le LLM jusqu'à 60 s,
# et le SDK réessaie sur erreur passagère.
DEFAULT_TIMEOUT_SECONDS = 120.0

REFUSAL_STATUSES = ("refus_score", "refus_llm")


@dataclass(frozen=True)
class Source:
    """Passage cité, avec le numéro utilisé dans le texte de la réponse."""

    numero: int
    reference: str


@dataclass(frozen=True)
class ApiAnswer:
    """Réponse de POST /questions (voir api.main.AnswerResponse)."""

    statut: str
    reponse: str
    sources: tuple[Source, ...]
    avertissements: tuple[str, ...]
    meilleur_score: float
    avertissement_juridique: str

    @property
    def is_refusal(self) -> bool:
        """Vrai si l'assistant a refusé de répondre (score ou LLM)."""
        return self.statut in REFUSAL_STATUSES


class ApiError(Exception):
    """Échec de l'appel à l'API, avec un message à montrer à l'utilisateur."""


def _parse(body: Any) -> ApiAnswer:
    """Construit un ApiAnswer à partir du JSON reçu.

    Raises:
        ApiError: si le JSON n'a pas la forme attendue.
    """
    try:
        return ApiAnswer(
            statut=body["statut"],
            reponse=body["reponse"],
            sources=tuple(Source(s["numero"], s["reference"]) for s in body["sources"]),
            avertissements=tuple(body["avertissements"]),
            meilleur_score=float(body["meilleur_score"]),
            avertissement_juridique=body["avertissement_juridique"],
        )
    except (KeyError, TypeError, ValueError):
        raise ApiError("Réponse inattendue de l'API.") from None


def ask_api(
    question: str,
    api_url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> ApiAnswer:
    """Pose une question à l'API et renvoie sa réponse.

    Args:
        question: question de l'utilisateur.
        api_url: adresse de l'API (ex. "http://127.0.0.1:8000").
        timeout: délai d'attente maximal, en secondes.
        transport: transport httpx à utiliser ; les tests y passent un faux
            serveur. None = vrai réseau.

    Returns:
        La réponse de l'API.

    Raises:
        ApiError: API injoignable, délai dépassé, question refusée (422),
            service indisponible (503) ou réponse inattendue.
    """
    try:
        with httpx.Client(base_url=api_url, timeout=timeout, transport=transport) as client:
            response = client.post("/questions", json={"question": question})
    except httpx.TimeoutException:
        raise ApiError("L'API n'a pas répondu à temps. Réessayez dans un instant.") from None
    except httpx.ConnectError:
        raise ApiError(
            f"L'API ne répond pas à {api_url}. "
            "La lancer avec : uvicorn fire_safety_rag.api.main:app"
        ) from None

    if response.status_code == 422:
        raise ApiError("Question invalide : elle doit contenir entre 1 et 1000 caractères.")
    if response.status_code == 503:
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = "raison inconnue"
        raise ApiError(f"Service indisponible : {detail}")
    if response.status_code != 200:
        raise ApiError(f"Erreur inattendue de l'API (code {response.status_code}).")
    try:
        body = response.json()
    except ValueError:
        raise ApiError("Réponse inattendue de l'API.") from None
    return _parse(body)

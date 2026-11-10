"""Appel au LLM (API Anthropic) et journal des coûts.

C'est le seul module qui connaît le fournisseur du LLM : passer à un autre
accès (Amazon Bedrock, Google Vertex AI…) ne demande de modifier que lui.

Le SDK anthropic gère le délai d'expiration et les nouvelles tentatives sur
erreur passagère (429, 5xx, coupure réseau). Ce module ajoute :
- la lecture de la clé depuis la configuration ;
- des messages d'erreur clairs, en français ;
- le signalement d'une réponse incomplète (coupée ou refusée) ;
- une ligne de journal par appel : tokens consommés et coût estimé.
"""

import logging
from dataclasses import dataclass

import anthropic

from fire_safety_rag.config import Settings, get_settings
from fire_safety_rag.generation.prompt import Prompt

logger = logging.getLogger(__name__)

# Prix en dollars par million de tokens (entrée, sortie), relevés le
# 2026-11-03 sur la grille publique d'Anthropic. Haiku 5.5 : tarif des
# prompts de moins de 100 000 tokens (largement notre cas). À mettre à jour
# si les prix changent.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class LLMError(Exception):
    """Échec de l'appel au LLM (clé absente ou invalide, réseau, API…)."""


@dataclass(frozen=True)
class LLMResponse:
    """Réponse du LLM et consommation de l'appel.

    Attributes:
        text: texte de la réponse (blocs de texte mis bout à bout).
        model: modèle qui a répondu.
        stop_reason: raison de l'arrêt ("end_turn" si la réponse est complète).
        input_tokens: tokens envoyés.
        output_tokens: tokens générés (réflexion comprise).
    """

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int

    @property
    def complete(self) -> bool:
        """Vrai si le modèle a fini sa réponse normalement."""
        return self.stop_reason == "end_turn"

    @property
    def cost_usd(self) -> float | None:
        """Coût estimé de l'appel en dollars, ou None si le prix du modèle est inconnu."""
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estime le coût d'un appel à partir de PRICES_USD_PER_MTOK.

    Args:
        model: identifiant du modèle.
        input_tokens: tokens envoyés.
        output_tokens: tokens générés.

    Returns:
        Coût en dollars, ou None si le modèle n'est pas dans la grille.
    """
    prices = PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def make_client(settings: Settings) -> anthropic.Anthropic:
    """Crée le client Anthropic à partir de la configuration.

    Args:
        settings: configuration (clé, délai d'expiration, nouvelles tentatives).

    Returns:
        Client prêt à l'emploi.

    Raises:
        LLMError: si la clé ANTHROPIC_API_KEY est absente ou vide.
    """
    key = settings.anthropic_api_key
    if key is None or not key.get_secret_value().strip():
        raise LLMError("Clé API absente : renseigner ANTHROPIC_API_KEY dans le fichier .env.")
    return anthropic.Anthropic(
        api_key=key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def generate(
    prompt: Prompt,
    client: anthropic.Anthropic | None = None,
    settings: Settings | None = None,
) -> LLMResponse:
    """Envoie le prompt au LLM et renvoie sa réponse.

    Args:
        prompt: consigne système et message utilisateur (voir build_prompt).
        client: client Anthropic ; créé depuis la configuration si absent
            (les tests passent un faux client, sans appel réseau).
        settings: configuration ; get_settings() si absente.

    Returns:
        La réponse et sa consommation. Vérifier `complete` : une réponse
        coupée (max_tokens) ou refusée par le modèle n'est pas complète.

    Raises:
        LLMError: si l'appel échoue après les nouvelles tentatives du SDK.
    """
    settings = settings or get_settings()
    client = client or make_client(settings)
    try:
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"effort": settings.llm_effort},
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError("Clé API refusée : vérifier ANTHROPIC_API_KEY.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Trop de requêtes ou crédits épuisés : réessayer plus tard.") from exc
    except anthropic.APITimeoutError as exc:
        raise LLMError(
            f"Pas de réponse en {settings.llm_timeout_seconds:g} s (délai dépassé)."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Connexion à l'API impossible : vérifier le réseau.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Erreur de l'API ({exc.status_code}) : {exc.message}") from exc

    # Les blocs de réflexion ("thinking") sont ignorés : seul le texte compte.
    text = "".join(block.text for block in message.content if block.type == "text")
    response = LLMResponse(
        text=text,
        model=message.model,
        stop_reason=message.stop_reason,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
    cost = response.cost_usd
    logger.info(
        "Appel LLM : modèle=%s, arrêt=%s, entrée=%d tokens, sortie=%d tokens, coût estimé=%s",
        response.model,
        response.stop_reason,
        response.input_tokens,
        response.output_tokens,
        "inconnu" if cost is None else f"{cost:.5f} $",
    )
    return response

"""API HTTP de l'assistant : poser une question, recevoir la réponse et ses sources.

Routes :
- GET  /health     : l'API répond ;
- POST /questions  : {"question": "..."} -> réponse, statut, sources, avertissements.
- GET  /docs       : documentation interactive générée par FastAPI.

La route reçoit la fonction « répondre à une question » par injection de
dépendance (Depends) : les tests la remplacent par une fausse fonction
(app.dependency_overrides), sans base, sans modèle et sans appel payant.

Étape 1 : la couche HTTP seule. Le branchement réel (base, modèle
d'embedding, LLM) viendra à l'étape 2 ; d'ici là, POST /questions répond 503.
"""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from fire_safety_rag.generation.answer import (
    DISCLAIMER,
    REFUSED_BY_LLM,
    REFUSED_BY_SCORE,
    Answer,
    answer_warnings,
)
from fire_safety_rag.generation.llm import LLMError
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE

logger = logging.getLogger(__name__)

# Répondre à une question : texte de la question -> résultat complet.
Answerer = Callable[[str], Answer]

MAX_QUESTION_CHARS = 1000


class QuestionRequest(BaseModel):
    """Corps de POST /questions."""

    # Espaces de bord retirés avant le contrôle : "   " est refusée (422)
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


class SourceResponse(BaseModel):
    """Passage cité par la réponse."""

    numero: int
    reference: str


class AnswerResponse(BaseModel):
    """Réponse de POST /questions.

    statut : refus_score (aucun passage assez proche, LLM non appelé),
    refus_llm (le LLM n'a pas trouvé la réponse dans les passages), reponse,
    ou incomplete (réponse coupée).
    """

    statut: str
    reponse: str
    sources: list[SourceResponse]
    avertissements: list[str]
    meilleur_score: float
    avertissement_juridique: str


def to_response(answer: Answer) -> AnswerResponse:
    """Traduit le résultat de answer_question en réponse de l'API.

    Args:
        answer: résultat de answer_question.

    Returns:
        Réponse prête à être envoyée en JSON. Un refus ne porte aucune source.
    """
    if answer.status == REFUSED_BY_SCORE or answer.checked is None:
        text, sources = REFUSAL_SENTENCE, []
    else:
        text = answer.checked.text.strip()
        cited = () if answer.status == REFUSED_BY_LLM else answer.checked.sources
        sources = [SourceResponse(numero=s.number, reference=s.reference) for s in cited]
    return AnswerResponse(
        statut=answer.status,
        reponse=text,
        sources=sources,
        avertissements=answer_warnings(answer),
        meilleur_score=answer.top_score,
        avertissement_juridique=DISCLAIMER,
    )


def get_answerer() -> Answerer:
    """Fournit la fonction de réponse ; branchée sur la vraie chaîne à l'étape 2.

    Raises:
        HTTPException: 503 tant que la chaîne n'est pas branchée.
    """
    raise HTTPException(status_code=503, detail="Service non configuré.")


app = FastAPI(
    title="fire-safety-rag",
    description=(
        "Assistant sur la réglementation sécurité incendie et sécurité privée, "
        "avec citation des sources. " + DISCLAIMER
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Indique que l'API répond (ne vérifie ni la base ni le LLM)."""
    return {"statut": "ok"}


@app.post("/questions")
def ask_question(
    request: QuestionRequest, answerer: Annotated[Answerer, Depends(get_answerer)]
) -> AnswerResponse:
    """Répond à une question en citant les articles utilisés.

    Fonction synchrone : FastAPI l'exécute dans un fil séparé, ce qui convient
    aux appels bloquants (base, modèle, LLM).

    Raises:
        HTTPException: 503 si l'appel au LLM échoue (détail dans les journaux,
            pas dans la réponse : il peut mentionner la configuration).
    """
    try:
        answer = answerer(request.question)
    except LLMError:
        logger.exception("Échec de l'appel au LLM")
        raise HTTPException(
            status_code=503, detail="Le service de génération est indisponible."
        ) from None
    return to_response(answer)

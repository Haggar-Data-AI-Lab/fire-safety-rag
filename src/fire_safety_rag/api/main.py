"""API HTTP de l'assistant : poser une question, recevoir la réponse et ses sources.

Routes :
- GET  /           : redirige vers /docs ;
- GET  /health     : l'API répond ;
- POST /questions  : {"question": "..."} -> réponse, statut, sources, avertissements.
- GET  /docs       : documentation interactive générée par FastAPI.

La route reçoit la fonction « répondre à une question » par injection de
dépendance (Depends) : les tests la remplacent par une fausse fonction
(app.dependency_overrides), sans base, sans modèle et sans appel payant.

Au démarrage (lifespan) : lecture de la configuration, vérification du schéma
de la base (sans le modifier), chargement des titres, préchauffage du modèle
d'embedding. Chaque requête ouvre puis ferme sa propre connexion à la base.

Lancement (base Docker démarrée, extra "embeddings" installé, clé API dans .env) :
    uvicorn fire_safety_rag.api.main:app
"""

import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, StringConstraints

from fire_safety_rag.config import Settings, get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.generation.answer import (
    DISCLAIMER,
    REFUSED_BY_LLM,
    REFUSED_BY_SCORE,
    Answer,
    answer_question,
    answer_warnings,
    load_titles,
)
from fire_safety_rag.generation.llm import LLMError, LLMResponse, generate
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, Prompt
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.schema import check_schema
from fire_safety_rag.storage.store import SearchResult, search

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


Connection = psycopg.Connection[tuple[Any, ...]]

# Bibliothèques trop bavardes au niveau INFO (une ligne par requête HTTP ou par lot).
_NOISY_LOGGERS = ("httpx", "httpx2", "sentence_transformers")


def configure_logging(level: str) -> None:
    """Affiche les journaux du projet (dont le coût de chaque appel au LLM).

    uvicorn ne configure que ses propres journaux : sans cette fonction, les
    lignes INFO du projet seraient perdues. Les journaux d'uvicorn ne sont pas
    dupliqués (ils ne remontent pas jusqu'au journal racine).

    Args:
        level: niveau minimal (ex. "INFO"), lu dans la configuration.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Préparation au démarrage de l'API, une seule fois.

    Ordre : configuration, schéma (l'API refuse de démarrer s'il est en
    retard, plutôt que d'échouer à chaque question), titres des documents,
    préchauffage du modèle d'embedding (sinon la première question attend
    son chargement).

    Raises:
        SchemaError: si le schéma de la base est en retard sur le code.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    with connect(settings) as connection:
        check_schema(connection)
    app.state.settings = settings
    app.state.titles = load_titles()
    embed_query("préchauffage du modèle")
    logger.info("API prête (modèle %s)", settings.embedding_model)
    yield


def get_connection(request: Request) -> Iterator[Connection]:
    """Ouvre une connexion pour la durée d'une requête, puis la ferme.

    Une connexion par requête : plus simple et plus sûr qu'une connexion
    partagée entre requêtes simultanées ; son ouverture (quelques ms) est
    négligeable devant l'appel au LLM.

    Raises:
        HTTPException: 503 si l'API n'a pas été initialisée ou si la base est
            injoignable.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Service non initialisé.")
    try:
        connection = connect(settings)
    except psycopg.OperationalError:
        logger.exception("Base de données injoignable")
        raise HTTPException(status_code=503, detail="Base de données indisponible.") from None
    with connection:
        yield connection


def get_answerer(
    request: Request, connection: Annotated[Connection, Depends(get_connection)]
) -> Answerer:
    """Assemble la vraie chaîne : recherche vectorielle, seuil de refus, LLM."""
    settings: Settings = request.app.state.settings
    titles: dict[str, str] = request.app.state.titles

    def retrieve(question: str, k: int) -> list[SearchResult]:
        return search(connection, embed_query(question), settings.embedding_model, k=k)

    def llm(prompt: Prompt) -> LLMResponse:
        return generate(prompt, settings=settings)

    def answer(question: str) -> Answer:
        return answer_question(question, retrieve, titles, llm, settings.refusal_min_score)

    return answer


app = FastAPI(
    title="fire-safety-rag",
    description=(
        "Assistant sur la réglementation sécurité incendie et sécurité privée, "
        "avec citation des sources. " + DISCLAIMER
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirige la racine vers la documentation interactive."""
    return RedirectResponse(url="/docs")


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

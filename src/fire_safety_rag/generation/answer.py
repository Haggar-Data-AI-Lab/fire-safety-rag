"""Réponse à une question : recherche, refus éventuel, appel au LLM, citations.

Parcours d'une question :
1. recherche des passages les plus proches (mode vecteurs) ;
2. si le meilleur score est sous le seuil (refusal_min_score), refus
   immédiat, sans appel au LLM ;
3. sinon construction du prompt, appel au LLM, contrôle des citations.

La recherche et l'appel au LLM sont passés en paramètres : les tests les
remplacent par de fausses fonctions (ni base, ni modèle, ni coût).

Utilisation (base indexée, extra "embeddings" installé, clé API dans .env) :
    python -m fire_safety_rag.generation.answer "Ma question ?" [--details]
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from fire_safety_rag.config import Settings, get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.generation.citations import CheckedAnswer, check_citations
from fire_safety_rag.generation.llm import LLMError, LLMResponse, generate
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, Prompt, build_prompt
from fire_safety_rag.ingestion.pipeline import DEFAULT_CATALOGUE, load_catalogue
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.store import SearchResult, search

REFUSED_BY_SCORE = "refus_score"
REFUSED_BY_LLM = "refus_llm"
ANSWERED = "reponse"
INCOMPLETE = "incomplete"

DISCLAIMER = (
    "Ces informations ne constituent pas un avis juridique. Vérifiez le texte officiel en vigueur."
)

# Recherche : (question, k) -> passages du meilleur au moins bon.
Retriever = Callable[[str, int], list[SearchResult]]
# Appel au LLM : prompt -> réponse.
Generator = Callable[[Prompt], LLMResponse]

_STOP_REASONS = {
    "max_tokens": "coupée par la limite de longueur",
    "refusal": "refusée par le modèle",
}


@dataclass(frozen=True)
class Answer:
    """Résultat complet pour une question.

    Attributes:
        question: question posée.
        status: REFUSED_BY_SCORE, REFUSED_BY_LLM, ANSWERED ou INCOMPLETE.
        results: passages retrouvés (envoyés au LLM s'il a été appelé).
        threshold: seuil de refus utilisé.
        response: réponse brute du LLM, ou None s'il n'a pas été appelé.
        checked: réponse après contrôle des citations, ou None.
    """

    question: str
    status: str
    results: tuple[SearchResult, ...]
    threshold: float
    response: LLMResponse | None = None
    checked: CheckedAnswer | None = None

    @property
    def top_score(self) -> float:
        """Score du meilleur passage (0 si aucun passage)."""
        return self.results[0].score if self.results else 0.0


def answer_question(
    question: str,
    retrieve: Retriever,
    titles: Mapping[str, str],
    generate_fn: Generator,
    threshold: float,
    k: int = 5,
) -> Answer:
    """Répond à une question, ou refuse si les passages ne sont pas assez proches.

    Args:
        question: question de l'utilisateur.
        retrieve: recherche des passages.
        titles: titre court de chaque document, indexé par document_id.
        generate_fn: appel au LLM (generate en usage réel).
        threshold: score minimal du meilleur passage pour appeler le LLM.
        k: nombre de passages envoyés au LLM.

    Returns:
        Le résultat ; le LLM n'est appelé que si le meilleur score atteint le seuil.

    Raises:
        LLMError: si l'appel au LLM échoue.
    """
    results = tuple(retrieve(question, k))
    if not results or results[0].score < threshold:
        return Answer(question, REFUSED_BY_SCORE, results, threshold)

    response = generate_fn(build_prompt(question, results, titles))
    checked = check_citations(response.text, results, titles)
    if not response.complete:
        status = INCOMPLETE
    elif checked.is_refusal:
        status = REFUSED_BY_LLM
    else:
        status = ANSWERED
    return Answer(question, status, results, threshold, response, checked)


def format_answer(answer: Answer, details: bool = False) -> str:
    """Met en forme le résultat pour l'affichage en ligne de commande.

    Args:
        answer: résultat de answer_question.
        details: ajoute les passages retrouvés et leur score.

    Returns:
        Texte à afficher, terminé par l'avertissement juridique.
    """
    lines: list[str] = []
    if answer.status == REFUSED_BY_SCORE:
        lines.append(REFUSAL_SENTENCE)
        lines.append(
            f"(aucun passage assez proche : meilleur score {answer.top_score:.4f}"
            f" < seuil {answer.threshold:g} ; le LLM n'a pas été appelé)"
        )
    elif answer.checked is not None and answer.response is not None:
        checked = answer.checked
        lines.append(checked.text.strip())
        if answer.status == INCOMPLETE:
            reason = answer.response.stop_reason or "inconnue"
            lines.append(
                f"⚠ Réponse incomplète ({_STOP_REASONS.get(reason, f'arrêt : {reason}')})."
            )
        if answer.status != REFUSED_BY_LLM:
            for number in checked.invalid_numbers:
                lines.append(f"⚠ Citation [{number}] inconnue : réponse à vérifier.")
            if checked.uncited:
                lines.append("⚠ Aucune source citée : réponse à vérifier.")
            if checked.sources:
                lines.append("")
                lines.append("Sources :")
                lines.extend(f"  [{s.number}] {s.reference}" for s in checked.sources)

    if details and answer.results:
        lines.append("")
        lines.append("Passages retrouvés (score) :")
        for number, result in enumerate(answer.results, start=1):
            chunk = result.chunk
            lines.append(f"  [{number}] {result.score:.4f}  {chunk.document_id}:{chunk.article_id}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def load_titles() -> dict[str, str]:
    """Titres courts des documents du catalogue, indexés par identifiant."""
    return {row["id"]: row["titre_court"] for row in load_catalogue(DEFAULT_CATALOGUE)}


def main(argv: Sequence[str] | None = None) -> int:
    """Répond à la question passée en argument et affiche le résultat.

    Args:
        argv: arguments de la ligne de commande (sys.argv si absent).

    Returns:
        Code de sortie : 0 si tout s'est bien passé, 1 si l'appel au LLM a échoué.
    """
    parser = argparse.ArgumentParser(description="Pose une question sur la réglementation.")
    parser.add_argument("question", help="question en langage courant")
    parser.add_argument(
        "--details", action="store_true", help="affiche les passages retrouvés et leur score"
    )
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    # Bruit des bibliothèques : seules nos lignes (dont le coût) restent visibles
    for noisy in ("httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    titles = load_titles()
    with connect(settings) as connection:

        def retrieve(question: str, k: int) -> list[SearchResult]:
            return search(connection, embed_query(question), settings.embedding_model, k=k)

        try:
            answer = answer_question(
                args.question,
                retrieve,
                titles,
                lambda prompt: generate(prompt, settings=settings),
                settings.refusal_min_score,
            )
        except LLMError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 1

    print()
    print(format_answer(answer, details=args.details))
    return 0


if __name__ == "__main__":
    sys.exit(main())

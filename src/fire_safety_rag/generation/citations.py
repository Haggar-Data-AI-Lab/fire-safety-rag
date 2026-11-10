"""Contrôle des citations d'une réponse du LLM.

Le LLM cite les passages par leur numéro ([1], [2][3] ou [1, 2]). Ce module
vérifie que chaque numéro correspond à un passage réellement envoyé, puis
associe à chaque numéro valide sa référence, tirée de nos données (jamais du
texte du modèle).

Le contrôle signale les problèmes sans rien décider : l'affichage choisit
ensuite comment présenter une citation invalide ou une réponse sans citation.
Ce module est pur : pas de réseau, pas de base de données.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE, format_reference
from fire_safety_rag.storage.store import SearchResult

# Un groupe entre crochets : "[1]" ou "[1, 2]". Les plages ("[1-3]") ne sont
# volontairement pas reconnues.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Source:
    """Passage cité par la réponse.

    Attributes:
        number: numéro du passage dans le prompt (à partir de 1).
        reference: titre, article, pages et vigueur du passage.
    """

    number: int
    reference: str


@dataclass(frozen=True)
class CheckedAnswer:
    """Réponse du LLM après contrôle des citations.

    Attributes:
        text: réponse telle que renvoyée par le LLM.
        is_refusal: la réponse contient la phrase de refus imposée.
        sources: passages valides cités, par numéro croissant, sans doublon.
        invalid_numbers: numéros cités qui ne correspondent à aucun passage.
        uncited: la réponse n'est pas un refus et ne cite aucun passage valide.
    """

    text: str
    is_refusal: bool
    sources: tuple[Source, ...]
    invalid_numbers: tuple[int, ...]
    uncited: bool


def cited_numbers(text: str) -> set[int]:
    """Extrait les numéros cités dans un texte.

    Args:
        text: réponse du LLM.

    Returns:
        Ensemble des numéros trouvés entre crochets.
    """
    return {int(number) for group in _CITATION.findall(text) for number in group.split(",")}


def check_citations(
    answer: str,
    results: Sequence[SearchResult],
    titles: Mapping[str, str],
) -> CheckedAnswer:
    """Contrôle les citations d'une réponse et retrouve leurs références.

    Args:
        answer: réponse du LLM.
        results: passages envoyés dans le prompt, dans le même ordre
            (le passage [1] est results[0]).
        titles: titre court de chaque document, indexé par document_id ;
            un document absent est affiché avec son document_id.

    Returns:
        La réponse contrôlée. Pour un refus, aucune source n'est retenue.
    """
    if REFUSAL_SENTENCE in answer:
        return CheckedAnswer(answer, True, (), (), False)

    numbers = cited_numbers(answer)
    valid = sorted(n for n in numbers if 1 <= n <= len(results))
    invalid = tuple(sorted(n for n in numbers if not 1 <= n <= len(results)))

    sources = []
    for number in valid:
        chunk = results[number - 1].chunk
        title = titles.get(chunk.document_id, chunk.document_id)
        sources.append(Source(number, format_reference(chunk, title)))

    return CheckedAnswer(answer, False, tuple(sources), invalid, uncited=not sources)

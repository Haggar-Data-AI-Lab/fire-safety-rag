"""Découpage du texte des documents du corpus en articles réglementaires."""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Expressions régulières pour la détection des titres d'articles
# ---------------------------------------------------------------------------

# Ligne contenant uniquement un titre d'article, avec un identifiant d'un à
# trois mots.
# Exemples observés :
#   "Article R4227-4"     (Code du travail)
#   "Article R*141-2"     (CCH)
#   "Article GN 1"        (arrêté ERP)
#   "Article CTS 31 bis"  (arrêté ERP)
# Ce motif seul ne suffit pas : "Article 3 :" (modèle de convention en annexe
# de l'arrêté SSIAP) le respecte aussi. Voir _VERSION_PREFIX.
_RE_ARTICLE_HEADING: re.Pattern[str] = re.compile(r"^Article\s+(\S+(?: \S+){0,2})$")

# Début de la ligne qui suit TOUJOURS un vrai titre d'article Légifrance.
# Exemples observés :
#   "VERSION EN VIGUEUR DEPUIS LE 01/05/2008"
#   "VERSION EN VIGUEUR DU 12/08/1982 AU 07/0..."  (version non actuelle)
_VERSION_PREFIX = "VERSION EN VIGUEUR"


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Article:
    """Représente un article réglementaire extrait d'un document du corpus.

    Attributes:
        document_id: identifiant unique du document dans le corpus.
        article_id: identifiant de l'article (ex. "R4227-4", "CTS 31 bis").
        version: ligne de version Légifrance (ex. "VERSION EN VIGUEUR DEPUIS
            LE 01/05/2008"), conservée pour distinguer plus tard les versions
            actuelles des versions abrogées.
        text: contenu de l'article, sans la ligne de titre ni la ligne de
            version.
    """

    document_id: str
    article_id: str
    version: str
    text: str


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def _next_non_empty(lines: list[str], start: int) -> int | None:
    """Renvoie l'indice de la première ligne non vide à partir de `start`.

    Args:
        lines: lignes du texte, déjà débarrassées des espaces de bord.
        start: indice à partir duquel chercher.

    Returns:
        Indice de la ligne trouvée, ou None si toutes les lignes restantes
        sont vides.
    """
    for index in range(start, len(lines)):
        if lines[index]:
            return index
    return None


def split_into_articles(text: str, document_id: str) -> list[Article]:
    """Découpe le texte d'un document en articles réglementaires.

    Une ligne est un titre d'article si elle correspond à "Article <identifiant>"
    ET si la ligne non vide suivante commence par "VERSION EN VIGUEUR".
    Le contenu d'un article va de la ligne qui suit sa version jusqu'au titre
    de l'article suivant (ou jusqu'à la fin du texte).

    Le texte situé avant le premier article (titre du document, sommaire)
    est ignoré.

    Args:
        text: texte complet et nettoyé d'un document.
        document_id: identifiant unique du document dans le corpus.

    Returns:
        Articles dans l'ordre du document ; liste vide si aucun titre
        d'article n'est détecté.
    """
    lines = [line.strip() for line in text.splitlines()]

    # Repérage des titres : (indice du titre, identifiant, indice de la version)
    headings: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        match = _RE_ARTICLE_HEADING.match(line)
        if match is None:
            continue
        version_index = _next_non_empty(lines, index + 1)
        if version_index is not None and lines[version_index].startswith(_VERSION_PREFIX):
            headings.append((index, match.group(1), version_index))

    # Construction des articles : le contenu s'arrête au titre suivant
    articles: list[Article] = []
    for position, (_, article_id, version_index) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[version_index + 1 : end]).strip()
        articles.append(
            Article(
                document_id=document_id,
                article_id=article_id,
                version=lines[version_index],
                text=body,
            )
        )
    return articles

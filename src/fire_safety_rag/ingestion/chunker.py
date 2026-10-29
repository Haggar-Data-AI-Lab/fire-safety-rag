"""Découpage du texte des documents du corpus en articles réglementaires."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from fire_safety_rag.ingestion.pdf_extractor import PageText

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

# Ligne contenant uniquement un titre d'annexe. L'identifiant retenu est la
# ligne entière.
# Exemples observés :
#   "Annexe"                    (arrêté ERP, annexe de T 52)
#   "Annexe I", "Annexe XIII"   (arrêté SSIAP)
#   "Annexe à l'article PO 11"  (arrêté ERP)
# Le sommaire "Annexes (Articles Annexe I à Annexe XIII)" n'est pas reconnu
# ("Annexes" au pluriel), et la vérification de la ligne VERSION s'applique
# comme pour les articles.
_RE_ANNEX_HEADING: re.Pattern[str] = re.compile(r"^Annexe(?: .{1,40})?$")

# Début de la ligne qui suit TOUJOURS un vrai titre d'article Légifrance.
_VERSION_PREFIX = "VERSION EN VIGUEUR"

# Ligne de version complète : seuls deux formats ont été observés sur le corpus.
# Exemples :
#   "VERSION EN VIGUEUR DEPUIS LE 01/05/2008"         (sans date de fin)
#   "VERSION EN VIGUEUR DU 15/08/1980 AU 01/06/2027"  (date de fin, passée
#                                                      = abrogé, future = fin
#                                                      programmée)
_RE_VERSION: re.Pattern[str] = re.compile(
    r"^VERSION EN VIGUEUR (?:DEPUIS LE (\d{2}/\d{2}/\d{4})"
    r"|DU (\d{2}/\d{2}/\d{4}) AU (\d{2}/\d{2}/\d{4}))$"
)


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
            LE 01/05/2008").
        valid_from: date de début de vigueur, lue dans `version`.
        valid_until: date de fin de vigueur, lue dans `version` ; None si
            l'article est en vigueur sans date de fin.
        text: contenu de l'article, sans la ligne de titre ni la ligne de
            version.
        page_start: page du PDF où se trouve le titre de l'article.
        page_end: page du PDF où se trouve la dernière ligne non vide de
            l'article.
        line_pages: page de chaque ligne de `text`, dans l'ordre (un élément
            par ligne de `text.splitlines()`). Sert à donner leurs vraies
            pages aux morceaux d'un article long.
    """

    document_id: str
    article_id: str
    version: str
    valid_from: date
    valid_until: date | None
    text: str
    page_start: int
    page_end: int
    line_pages: tuple[int, ...]


@dataclass(frozen=True)
class Chunk:
    """Représente un morceau de texte prêt à être vectorisé (embedding).

    Un article court donne un seul morceau ; un article long en donne
    plusieurs, qui se chevauchent légèrement.

    Attributes:
        document_id: identifiant unique du document dans le corpus.
        article_id: identifiant de l'article d'origine.
        version: ligne de version Légifrance de l'article d'origine.
        valid_from: date de début de vigueur de l'article d'origine.
        valid_until: date de fin de vigueur de l'article d'origine, ou None.
        chunk_index: position du morceau dans l'article, en commençant à 0.
        text: contenu du morceau.
        page_start: page du PDF de la première ligne du morceau.
        page_end: page du PDF de la dernière ligne du morceau.
    """

    document_id: str
    article_id: str
    version: str
    valid_from: date
    valid_until: date | None
    chunk_index: int
    text: str
    page_start: int
    page_end: int


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    """Convertit une date au format JJ/MM/AAAA en objet date."""
    return datetime.strptime(text, "%d/%m/%Y").date()


def parse_version(line: str) -> tuple[date, date | None]:
    """Lit les dates de vigueur dans une ligne de version Légifrance.

    Args:
        line: ligne de version, ex. "VERSION EN VIGUEUR DEPUIS LE 01/05/2008"
            ou "VERSION EN VIGUEUR DU 15/08/1980 AU 01/06/2027".

    Returns:
        Couple (date de début, date de fin) ; la date de fin vaut None pour
        le format "DEPUIS LE".

    Raises:
        ValueError: si la ligne ne correspond à aucun format connu (on
            préfère échouer plutôt que d'indexer un article mal daté).
    """
    match = _RE_VERSION.match(line)
    if match is None:
        raise ValueError(f"Ligne de version non reconnue : {line!r}")
    since, start, end = match.groups()
    if since is not None:
        return _parse_date(since), None
    return _parse_date(start), _parse_date(end)


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


def _heading_id(line: str) -> str | None:
    """Renvoie l'identifiant d'un titre d'article ou d'annexe.

    Args:
        line: ligne de texte, débarrassée des espaces de bord.

    Returns:
        "R4227-4" pour "Article R4227-4", la ligne entière pour une annexe
        (ex. "Annexe I"), ou None si la ligne n'a pas la forme d'un titre.
    """
    article_match = _RE_ARTICLE_HEADING.match(line)
    if article_match is not None:
        return article_match.group(1)
    if _RE_ANNEX_HEADING.match(line):
        return line
    return None


def split_into_articles(pages: Sequence[PageText]) -> list[Article]:
    """Découpe les pages d'un document en articles réglementaires.

    Une ligne est un titre si elle correspond à "Article <identifiant>" ou
    "Annexe ..." ET si la ligne non vide suivante commence par
    "VERSION EN VIGUEUR". Les annexes sont traitées comme des articles.
    Le contenu d'un article va de la ligne qui suit sa version jusqu'au titre
    de l'article suivant (ou jusqu'à la fin du document), même s'il s'étend
    sur plusieurs pages.

    Le texte situé avant le premier article (titre du document, sommaire)
    est ignoré.

    Args:
        pages: pages nettoyées d'un SEUL document, dans l'ordre. Le
            document_id des articles est lu sur la première page.

    Returns:
        Articles dans l'ordre du document ; liste vide si aucun titre
        d'article n'est détecté.
    """
    if not pages:
        return []
    document_id = pages[0].document_id

    # Listes parallèles : chaque ligne et le numéro de la page d'où elle vient
    lines: list[str] = []
    line_pages: list[int] = []
    for page in pages:
        for line in page.text.splitlines():
            lines.append(line.strip())
            line_pages.append(page.page_number)

    # Repérage des titres : (indice du titre, identifiant, indice de la version)
    headings: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        article_id = _heading_id(line)
        if article_id is None:
            continue
        version_index = _next_non_empty(lines, index + 1)
        if version_index is not None and lines[version_index].startswith(_VERSION_PREFIX):
            headings.append((index, article_id, version_index))

    # Construction des articles : le contenu s'arrête au titre suivant
    articles: list[Article] = []
    for position, (heading_index, article_id, version_index) in enumerate(headings):
        start = version_index + 1
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)

        # Retrait des lignes vides au début et à la fin du contenu, par indices
        # (et non par strip()) pour que line_pages reste aligné sur le texte
        while start < end and not lines[start]:
            start += 1
        while end > start and not lines[end - 1]:
            end -= 1

        # Dernière ligne non vide de l'article (au minimum, sa ligne de version)
        last_index = end - 1 if end > start else version_index
        valid_from, valid_until = parse_version(lines[version_index])

        articles.append(
            Article(
                document_id=document_id,
                article_id=article_id,
                version=lines[version_index],
                valid_from=valid_from,
                valid_until=valid_until,
                text="\n".join(lines[start:end]),
                page_start=line_pages[heading_index],
                page_end=line_pages[last_index],
                line_pages=tuple(line_pages[start:end]),
            )
        )
    return articles


def _ends_sentence(line: str) -> bool:
    """Indique si une ligne termine une phrase ou un élément d'énumération.

    Args:
        line: ligne de texte.

    Returns:
        True si la ligne se termine par ".", ";" ou ":".
    """
    return line.endswith((".", ";", ":"))


def split_article(article: Article, max_chars: int = 1500, overlap_chars: int = 200) -> list[Chunk]:
    """Découpe un article en morceaux d'au plus `max_chars` caractères.

    Le PDF ne marque pas les paragraphes : la découpe se fait donc entre deux
    lignes. Pour un article trop long :
    - on remplit le morceau ligne par ligne tant qu'il tient dans `max_chars` ;
    - on recule la coupe jusqu'à la dernière ligne qui termine une phrase
      (".", ";", ":"), si cela laisse un morceau d'au moins `max_chars // 2`
      caractères ; sinon on coupe à la dernière ligne qui tient ;
    - le morceau suivant reprend les dernières lignes du précédent, dans la
      limite de `overlap_chars` caractères (chevauchement).

    Une ligne seule plus longue que `max_chars` forme un morceau à elle seule.

    Args:
        article: article à découper.
        max_chars: taille maximale visée d'un morceau, en caractères.
        overlap_chars: taille maximale du chevauchement, en caractères.

    Returns:
        Morceaux de l'article, dans l'ordre, numérotés à partir de 0.
    """
    if len(article.text) <= max_chars:
        return [
            Chunk(
                document_id=article.document_id,
                article_id=article.article_id,
                version=article.version,
                valid_from=article.valid_from,
                valid_until=article.valid_until,
                chunk_index=0,
                text=article.text,
                page_start=article.page_start,
                page_end=article.page_end,
            )
        ]

    lines = article.text.splitlines()
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        # Remplissage : on ajoute des lignes tant que le morceau tient
        end = start
        size = 0
        while end < len(lines):
            added = len(lines[end]) + (1 if end > start else 0)  # +1 pour "\n"
            if size + added > max_chars:
                break
            size += added
            end += 1
        if end == start:
            end = start + 1  # ligne seule trop longue

        # Coupe de préférence après une fin de phrase, si le morceau reste assez long
        if end < len(lines):
            for cut in range(end, start, -1):
                if _ends_sentence(lines[cut - 1]):
                    if len("\n".join(lines[start:cut])) >= max_chars // 2:
                        end = cut
                    break

        chunks.append(
            Chunk(
                document_id=article.document_id,
                article_id=article.article_id,
                version=article.version,
                valid_from=article.valid_from,
                valid_until=article.valid_until,
                chunk_index=len(chunks),
                text="\n".join(lines[start:end]),
                page_start=article.line_pages[start],
                page_end=article.line_pages[end - 1],
            )
        )
        if end >= len(lines):
            break

        # Chevauchement : on recule tant que les lignes reprises tiennent dans
        # overlap_chars, sans revenir au début du morceau (pour toujours avancer)
        next_start = end
        overlap = 0
        while next_start - 1 > start and overlap + len(lines[next_start - 1]) <= overlap_chars:
            next_start -= 1
            overlap += len(lines[next_start]) + 1
        start = next_start
    return chunks

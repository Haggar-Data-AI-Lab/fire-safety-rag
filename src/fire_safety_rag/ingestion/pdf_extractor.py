"""Extraction et nettoyage du texte brut des fichiers PDF du corpus."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Expressions régulières pour les lignes de bruit d'impression Légifrance
# ---------------------------------------------------------------------------

# Ligne d'horodatage + titre d'onglet, présente en avant-dernière position sur
# chaque page imprimée depuis Légifrance.
# Exemples observés :
#   "19/10/2026 20:35 Chapitre VII : Risques d'incendies ... - Légifrance"
#   "19/10/2026 20:32 Arrêté du 2 mai 2005 ... ince…"
# La ligne se termine TOUJOURS par "- Légifrance" (titre complet) ou "…" (titre
# tronqué). Ce critère de fin évite de supprimer une ligne réglementaire qui
# commencerait fortuitement par une date et une heure.
_RE_TIMESTAMP_LINE: re.Pattern[str] = re.compile(
    r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2} .+(?:- Légifrance|…)$"
)

# Ligne URL Légifrance + ratio de pagination, toujours en dernière position.
# Exemples observés :
#   "https://www.legifrance.gouv.fr/codes/id/LEGISCTA000018532149 9/20"
#   "https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000448223 1/147"
_RE_URL_LINE: re.Pattern[str] = re.compile(r"^https://www\.legifrance\.gouv\.fr/\S+ \d+/\d+$")


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageText:
    """Représente le texte nettoyé d'une page d'un document du corpus.

    Attributes:
        document_id: identifiant unique du document dans le corpus.
        page_number: numéro de la page, en commençant à 1.
        text: texte nettoyé extrait de la page.
    """

    document_id: str
    page_number: int
    text: str


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def clean_page_text(text: str) -> str:
    """Supprime les lignes de bruit d'impression Légifrance et normalise le texte.

    Supprime uniquement les lignes entières correspondant aux motifs suivants :
    - ligne d'horodatage + titre d'onglet (ex. "19/10/2026 20:35 ... - Légifrance")
    - ligne URL + pagination (ex. "https://www.legifrance.gouv.fr/... 9/20")

    Conserve sans modification :
    - les lignes "VERSION EN VIGUEUR..."
    - les lignes "Création..." / "Modifié par..."
    - tout autre contenu réglementaire

    Normalise ensuite :
    - espaces en fin de ligne supprimés
    - séquences de plus de 2 lignes vides consécutives réduites à 1
    - espaces en début et fin de texte global retirés

    Args:
        text: texte brut extrait d'une page PDF.

    Returns:
        Texte nettoyé et normalisé.
    """
    lines = text.splitlines()

    # Suppression des lignes de bruit
    filtered: list[str] = []
    for line in lines:
        rstripped = line.rstrip()
        if _RE_TIMESTAMP_LINE.match(rstripped) or _RE_URL_LINE.match(rstripped):
            continue
        filtered.append(rstripped)

    # Réduction des séquences de plus de 2 lignes vides consécutives à 1
    normalized: list[str] = []
    blank_run = 0
    for line in filtered:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                normalized.append(line)
        else:
            blank_run = 0
            normalized.append(line)

    return "\n".join(normalized).strip()


def extract_pages(pdf_path: Path, document_id: str) -> Iterator[PageText]:
    """Lit un fichier PDF et produit une PageText nettoyée pour chaque page.

    Args:
        pdf_path: chemin vers le fichier PDF à lire.
        document_id: identifiant unique du document dans le corpus.

    Yields:
        PageText nettoyé pour chaque page du PDF, numérotées à partir de 1.
    """
    reader = PdfReader(pdf_path)
    for index, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        yield PageText(
            document_id=document_id,
            page_number=index + 1,
            text=clean_page_text(raw_text),
        )

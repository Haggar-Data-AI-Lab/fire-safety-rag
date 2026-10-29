"""Pipeline d'ingestion : du catalogue des sources jusqu'au fichier de chunks.

Enchaîne, pour chaque document du catalogue : extraction des pages PDF,
découpage en articles, puis découpage des articles en chunks. Le résultat
est écrit dans un fichier JSONL (un chunk par ligne).

Utilisation :
    python -m fire_safety_rag.ingestion.pipeline
"""

import csv
import json
import logging
import statistics
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fire_safety_rag.config import get_settings
from fire_safety_rag.ingestion.chunker import (
    Article,
    Chunk,
    split_article,
    split_into_articles,
)
from fire_safety_rag.ingestion.pdf_extractor import extract_pages

logger = logging.getLogger(__name__)

# Racine du projet : src/fire_safety_rag/ingestion/pipeline.py -> 3 niveaux au-dessus
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOGUE = PROJECT_ROOT / "data" / "sources.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_catalogue(path: Path) -> list[dict[str, str]]:
    """Lit le catalogue des sources (CSV) et renvoie une ligne par document.

    Args:
        path: chemin du fichier CSV du catalogue.

    Returns:
        Lignes du catalogue, sous forme de dictionnaires colonne -> valeur.
    """
    # utf-8-sig : accepte un éventuel BOM ajouté par Excel
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_in_force(article: Article, reference_date: date) -> bool:
    """Indique si un article est en vigueur à une date donnée.

    Un article "DU X AU Y" cesse d'être en vigueur le jour Y (date à laquelle
    il est abrogé ou remplacé). Un article sans date de fin est en vigueur.

    Args:
        article: article à tester.
        reference_date: date à laquelle on évalue la vigueur.

    Returns:
        True si l'article est en vigueur à `reference_date`.
    """
    return article.valid_until is None or reference_date < article.valid_until


def chunk_corpus(catalogue_path: Path, raw_dir: Path, reference_date: date) -> list[Chunk]:
    """Découpe en chunks tous les documents listés dans le catalogue.

    Pour chaque document : extraction des pages, découpage en articles,
    exclusion des articles qui ne sont plus en vigueur à `reference_date`,
    puis découpage de chaque article restant en chunks. Des statistiques par
    document sont journalisées au niveau INFO.

    Args:
        catalogue_path: chemin du catalogue des sources (CSV).
        raw_dir: dossier contenant les fichiers PDF du corpus.
        reference_date: date à laquelle on évalue la vigueur des articles.

    Returns:
        Chunks de tout le corpus, dans l'ordre du catalogue puis du document.
    """
    chunks: list[Chunk] = []
    for entry in load_catalogue(catalogue_path):
        document_id = entry["id"]
        pages = list(extract_pages(raw_dir / entry["fichier"], document_id))
        articles = split_into_articles(pages)
        in_force = [article for article in articles if is_in_force(article, reference_date)]
        document_chunks = [chunk for article in in_force for chunk in split_article(article)]
        sizes = [len(chunk.text) for chunk in document_chunks]
        logger.info(
            "%s : %d pages, %d articles (%d exclus car plus en vigueur), "
            "%d chunks (taille médiane %d, max %d)",
            document_id,
            len(pages),
            len(articles),
            len(articles) - len(in_force),
            len(document_chunks),
            statistics.median(sizes) if sizes else 0,
            max(sizes, default=0),
        )
        chunks.extend(document_chunks)
    return chunks


def _json_default(value: object) -> str:
    """Convertit les valeurs que json ne sait pas écrire seul.

    Seules les dates sont acceptées (format ISO, ex. "2027-06-01") ; tout
    autre type lève une erreur plutôt que d'être converti silencieusement.

    Args:
        value: valeur rencontrée par json.dumps.

    Returns:
        Représentation ISO de la date.

    Raises:
        TypeError: si la valeur n'est pas une date.
    """
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Type non sérialisable en JSON : {type(value).__name__}")


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Écrit les chunks dans un fichier JSONL, un objet JSON par ligne.

    Le dossier parent est créé s'il n'existe pas ; un fichier existant est
    remplacé.

    Args:
        chunks: chunks à écrire.
        output_path: chemin du fichier JSONL à produire.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # ensure_ascii=False : garde les accents lisibles dans le fichier
            line = json.dumps(asdict(chunk), ensure_ascii=False, default=_json_default)
            f.write(line + "\n")


def main() -> None:
    """Exécute le pipeline sur le corpus avec les chemins par défaut."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    chunks = chunk_corpus(DEFAULT_CATALOGUE, DEFAULT_RAW_DIR, reference_date=date.today())
    write_chunks_jsonl(chunks, DEFAULT_OUTPUT)
    logger.info("%d chunks écrits dans %s", len(chunks), DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()

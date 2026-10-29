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
from pathlib import Path

from fire_safety_rag.config import get_settings
from fire_safety_rag.ingestion.chunker import Chunk, split_article, split_into_articles
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


def chunk_corpus(catalogue_path: Path, raw_dir: Path) -> list[Chunk]:
    """Découpe en chunks tous les documents listés dans le catalogue.

    Pour chaque document : extraction des pages, découpage en articles,
    puis découpage de chaque article en chunks. Des statistiques par
    document sont journalisées au niveau INFO.

    Args:
        catalogue_path: chemin du catalogue des sources (CSV).
        raw_dir: dossier contenant les fichiers PDF du corpus.

    Returns:
        Chunks de tout le corpus, dans l'ordre du catalogue puis du document.
    """
    chunks: list[Chunk] = []
    for entry in load_catalogue(catalogue_path):
        document_id = entry["id"]
        pages = list(extract_pages(raw_dir / entry["fichier"], document_id))
        articles = split_into_articles(pages)
        document_chunks = [chunk for article in articles for chunk in split_article(article)]
        sizes = [len(chunk.text) for chunk in document_chunks]
        logger.info(
            "%s : %d pages, %d articles, %d chunks (taille médiane %d, max %d)",
            document_id,
            len(pages),
            len(articles),
            len(document_chunks),
            statistics.median(sizes) if sizes else 0,
            max(sizes, default=0),
        )
        chunks.extend(document_chunks)
    return chunks


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
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main() -> None:
    """Exécute le pipeline sur le corpus avec les chemins par défaut."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    chunks = chunk_corpus(DEFAULT_CATALOGUE, DEFAULT_RAW_DIR)
    write_chunks_jsonl(chunks, DEFAULT_OUTPUT)
    logger.info("%d chunks écrits dans %s", len(chunks), DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()

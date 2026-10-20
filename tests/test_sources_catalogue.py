"""Tests de cohérence du catalogue des sources (data/sources.csv)."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"

EXPECTED_COLUMNS = [
    "id",
    "titre",
    "type",
    "theme",
    "url",
    "format",
    "pages",
    "date_version",
    "date_telechargement",
    "licence",
    "fichier",
    "notes",
]
ALLOWED_TYPES = {"arrete", "code", "convention", "guide"}
ALLOWED_THEMES = {"incendie", "securite-privee", "social", "vulgarisation"}


def read_catalogue() -> list[dict[str, str]]:
    """Lit le catalogue et renvoie une liste de lignes (une par document)."""
    with CATALOGUE.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_columns_match_schema() -> None:
    """Le catalogue respecte exactement le schéma de colonnes attendu."""
    with CATALOGUE.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))

    assert header == EXPECTED_COLUMNS


def test_ids_are_unique() -> None:
    """Chaque document a un identifiant unique."""
    ids = [row["id"] for row in read_catalogue()]

    assert len(ids) == len(set(ids))


def test_file_name_matches_id() -> None:
    """Le nom du fichier est l'identifiant suivi de l'extension du format."""
    for row in read_catalogue():
        assert row["fichier"] == f"{row['id']}.{row['format']}", row["id"]


def test_every_file_exists() -> None:
    """Chaque fichier référencé existe bien dans data/raw/."""
    for row in read_catalogue():
        assert (RAW_DIR / row["fichier"]).is_file(), row["fichier"]


def test_controlled_vocabularies() -> None:
    """Les colonnes type et theme n'utilisent que les valeurs autorisées."""
    for row in read_catalogue():
        assert row["type"] in ALLOWED_TYPES, row["id"]
        assert row["theme"] in ALLOWED_THEMES, row["id"]

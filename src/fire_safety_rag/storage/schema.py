"""Schéma de la base : tables des documents indexés et de leurs chunks.

Le SQL est exécuté depuis Python, et non depuis db/init/ : Docker n'exécute
les scripts de db/init/ qu'à la toute première création du volume de données.

Grâce à IF NOT EXISTS, init_schema est idempotente : elle peut être appelée à
chaque lancement. Limite assumée : CREATE TABLE IF NOT EXISTS ne modifie pas
une table existante. Une colonne ajoutée après coup l'est par
ALTER TABLE ... ADD COLUMN IF NOT EXISTS (ex. section) ; pour un changement
plus profond, supprimer les tables puis tout réindexer (pas d'outil de
migration tant qu'il n'y en a pas besoin).
"""

from typing import Any

import psycopg

# Dimension des vecteurs de intfloat/multilingual-e5-base. Changer de modèle
# impose de recréer les tables (et de tout revectoriser).
EMBEDDING_DIMENSION = 768

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS documents (
    document_id     text PRIMARY KEY,
    content_hash    text NOT NULL,
    embedding_model text NOT NULL,
    indexed_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    document_id  text NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    position     integer NOT NULL,
    article_id   text NOT NULL,
    chunk_index  integer NOT NULL,
    version      text NOT NULL,
    valid_from   date NOT NULL,
    valid_until  date,
    history      text[] NOT NULL,
    text         text NOT NULL,
    page_start   integer NOT NULL,
    page_end     integer NOT NULL,
    embedding    vector({EMBEDDING_DIMENSION}) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('french', text)) STORED,
    PRIMARY KEY (document_id, position)
);

-- Ajoutée après coup : ADD COLUMN IF NOT EXISTS met à jour une base existante
-- sans supprimer les tables. Les anciennes lignes reçoivent '' jusqu'à la
-- réindexation.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section text NOT NULL DEFAULT '';
"""


def init_schema(connection: psycopg.Connection[tuple[Any, ...]]) -> None:
    """Crée les tables `documents` et `chunks` si elles n'existent pas.

    - documents : une ligne par document indexé (empreinte du PDF, modèle
      d'embedding utilisé, date d'indexation).
    - chunks : une ligne par chunk, identifiée par (document_id, position),
      car article_id n'est pas unique (ex. "Annexe I" deux fois dans l'ERP).
      Supprimer un document supprime ses chunks (ON DELETE CASCADE).
      search_vector est calculée par PostgreSQL à partir de text (plein texte).

    Args:
        connection: connexion ouverte (voir storage.database.connect).
    """
    connection.execute(SCHEMA_SQL)
    connection.commit()


# Colonnes dont le code a besoin, par table (à tenir à jour avec SCHEMA_SQL).
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "documents": ("document_id", "content_hash", "embedding_model", "indexed_at"),
    "chunks": (
        "document_id",
        "position",
        "article_id",
        "chunk_index",
        "version",
        "valid_from",
        "valid_until",
        "history",
        "text",
        "page_start",
        "page_end",
        "embedding",
        "search_vector",
        "section",
    ),
}


class SchemaError(RuntimeError):
    """Le schéma de la base est en retard sur le code (colonne ou table manquante)."""


def missing_columns(existing: set[tuple[str, str]]) -> list[str]:
    """Liste les colonnes attendues absentes de la base.

    Args:
        existing: couples (table, colonne) présents dans la base.

    Returns:
        Colonnes manquantes, sous la forme "table.colonne", dans l'ordre de
        EXPECTED_COLUMNS ; liste vide si le schéma est complet.
    """
    return [
        f"{table}.{column}"
        for table, columns in EXPECTED_COLUMNS.items()
        for column in columns
        if (table, column) not in existing
    ]


def check_schema(connection: psycopg.Connection[tuple[Any, ...]]) -> None:
    """Vérifie que la base contient toutes les colonnes attendues, sans la modifier.

    À appeler au démarrage d'un service qui ne fait que lire (l'API) : mettre
    le schéma à jour reste le rôle de l'indexeur, qui appelle init_schema.

    Args:
        connection: connexion ouverte (voir storage.database.connect).

    Raises:
        SchemaError: s'il manque au moins une colonne (ou une table).
    """
    rows = connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = ANY(%s)",
        (list(EXPECTED_COLUMNS),),
    ).fetchall()
    missing = missing_columns({(table, column) for table, column in rows})
    if missing:
        raise SchemaError(
            f"Schéma de la base en retard sur le code, colonnes manquantes : {', '.join(missing)}."
            " Lancer l'indexeur pour le mettre à jour : python -m fire_safety_rag.indexing.indexer"
        )

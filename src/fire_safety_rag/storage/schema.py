"""Schéma de la base : tables des documents indexés et de leurs chunks.

Le SQL est exécuté depuis Python, et non depuis db/init/ : Docker n'exécute
les scripts de db/init/ qu'à la toute première création du volume de données.

Grâce à IF NOT EXISTS, init_schema est idempotente : elle peut être appelée à
chaque lancement. Limite assumée : elle ne modifie pas une table existante.
Si le schéma change, supprimer les tables puis tout réindexer (pas d'outil de
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
    PRIMARY KEY (document_id, position)
);
"""


def init_schema(connection: psycopg.Connection[tuple[Any, ...]]) -> None:
    """Crée les tables `documents` et `chunks` si elles n'existent pas.

    - documents : une ligne par document indexé (empreinte du PDF, modèle
      d'embedding utilisé, date d'indexation).
    - chunks : une ligne par chunk, identifiée par (document_id, position),
      car article_id n'est pas unique (ex. "Annexe I" deux fois dans l'ERP).
      Supprimer un document supprime ses chunks (ON DELETE CASCADE).

    Args:
        connection: connexion ouverte (voir storage.database.connect).
    """
    connection.execute(SCHEMA_SQL)
    connection.commit()

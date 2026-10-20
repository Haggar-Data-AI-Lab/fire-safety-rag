-- Active l'extension pgvector pour le stockage et la recherche de vecteurs.
-- Ce script est exécuté automatiquement par PostgreSQL au premier démarrage du conteneur.
CREATE EXTENSION IF NOT EXISTS vector;

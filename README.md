# fire-safety-rag
[![CI](https://github.com/Haggar-Data-AI-Lab/fire-safety-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Haggar-Data-AI-Lab/fire-safety-rag/actions/workflows/ci.yml)

Assistant IA (RAG) qui répond en français à des questions sur la réglementation
sécurité incendie et sécurité privée, en s'appuyant uniquement sur des documents
publics et en citant ses sources.

> ⚠️ Outil d'aide à la recherche documentaire. Ne constitue pas un avis juridique.

## Statut

🚧 En cours de développement : Phase 1 (ingestion), corpus initial et base de données en place.

## Démarrage rapide

Prérequis : Python 3.11+, Docker Desktop.

```bash
# Environnement Python
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev]"

# Configuration : copier le modèle puis définir un vrai mot de passe
cp .env.example .env

# Base de données PostgreSQL + pgvector
docker compose up -d
docker compose ps                 # STATUS doit afficher "healthy"

# Vérifications
pytest
```

Si le port 5432 est déjà utilisé sur votre machine, changez `POSTGRES_PORT` dans `.env`.

## Licence

MIT. Voir [LICENSE](LICENSE).

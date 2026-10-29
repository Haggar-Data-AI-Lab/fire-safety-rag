# fire-safety-rag
[![CI](https://github.com/Haggar-Data-AI-Lab/fire-safety-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Haggar-Data-AI-Lab/fire-safety-rag/actions/workflows/ci.yml)

Assistant IA (RAG) qui répond en français à des questions sur la réglementation
sécurité incendie et sécurité privée, en s'appuyant uniquement sur des documents
publics et en citant ses sources.

> ⚠️ Outil d'aide à la recherche documentaire. Ne constitue pas un avis juridique.

## Statut

🚧 En cours de développement.

- ✅ **Ingestion** : extraction des PDF, découpage par article (avec annexes, pages,
  dates de vigueur et historique), exclusion des articles abrogés.
- ✅ **Indexation** : vectorisation locale (`intfloat/multilingual-e5-base`) et stockage
  dans PostgreSQL + pgvector, avec saut des documents inchangés (empreinte SHA-256).
- ✅ **Évaluation de la recherche** : 29 questions, mesure de référence ci-dessous.
- ⏳ À venir : amélioration de la recherche, génération des réponses avec citations
  (LLM), API et interface.

## Résultats actuels (recherche seule)

Mesure de référence sur 29 questions ([`data/eval/questions.csv`](data/eval/questions.csv)) :

| Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|
| 0,34 | 0,52 | 0,59 | 0,44 |

Hit@k : part des questions dont l'article attendu figure dans les k premiers résultats.
MRR : moyenne de 1 / rang du premier bon résultat. Ces chiffres servent de point de
départ pour mesurer chaque amélioration ; le jeu de questions est encore petit
(une question = 3,4 points).

## Démarrage rapide

Prérequis : Python 3.11+, Docker Desktop.

```bash
# Environnement Python
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev,embeddings]"   # "embeddings" installe PyTorch (lourd)

# Configuration : copier le modèle puis définir un vrai mot de passe
cp .env.example .env

# Base de données PostgreSQL + pgvector
docker compose up -d
docker compose ps                 # STATUS doit afficher "healthy"

# Vérifications
pytest
```

Si le port 5432 est déjà utilisé sur votre machine, changez `POSTGRES_PORT` dans `.env`.

Sans l'extra `embeddings` (cas de la CI), les tests qui demandent le modèle sont
ignorés. Sans base démarrée, les tests qui en dépendent sont ignorés aussi.

## Utilisation

Le corpus doit être présent dans `data/raw/` (voir ci-dessous) et la base démarrée.

```bash
# 1. Découper le corpus en chunks -> data/processed/chunks.jsonl (inspection)
python -m fire_safety_rag.ingestion.pipeline

# 2. Indexer dans PostgreSQL (~10 min sur CPU la première fois ;
#    les documents inchangés sont ensuite sautés, --force pour tout refaire)
python -m fire_safety_rag.indexing.indexer

# 3. Mesurer la qualité de la recherche
python -m fire_safety_rag.evaluation.evaluate
```

Le premier lancement télécharge le modèle d'embedding (~1 Go) dans le cache
Hugging Face de l'utilisateur, hors du dépôt.

## Organisation du code

| Module | Rôle |
|---|---|
| `ingestion/` | Extraction des PDF, découpage en articles et en chunks, pipeline JSONL |
| `embedding/` | Vectorisation des chunks et des questions (modèle local) |
| `storage/` | Connexion PostgreSQL, schéma des tables, écriture et recherche |
| `indexing/` | Indexation du corpus dans la base |
| `evaluation/` | Métriques (Hit@k, MRR) et script d'évaluation |

## Obtenir le corpus

Les documents sources ne sont pas versionnés (volumineux, disponibles à leur source
officielle). Le catalogue [`data/sources.csv`](data/sources.csv) liste chaque document
avec son URL, sa date de version et sa licence.

Pour reconstituer le corpus, pour chaque ligne du catalogue :

1. ouvrir l'URL indiquée dans la colonne `url` (Légifrance) ;
2. imprimer la page en PDF depuis le navigateur ;
3. enregistrer le fichier dans `data/raw/` sous le nom indiqué dans la colonne `fichier`.

Sans corpus, les tests qui en dépendent sont ignorés (`SKIPPED`) ; les autres s'exécutent normalement.

## Licence

MIT. Voir [LICENSE](LICENSE).

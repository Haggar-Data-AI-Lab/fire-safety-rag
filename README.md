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
- ✅ **Évaluation de la recherche** : 29 questions, résultats ci-dessous.
- 🔄 **Amélioration de la recherche** : une piste à la fois, mesurée à chaque fois.
- ✅ **Recherche hybride évaluée** (plein texte PostgreSQL + vecteurs, fusion RRF) :
  non retenue, voir ci-dessous.
- ⏳ À venir : génération des réponses avec citations (LLM), API et interface.

## Résultats actuels (recherche seule)

Mesures sur 29 questions ([`data/eval/questions.csv`](data/eval/questions.csv)),
une seule modification par ligne :

| Version | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Référence : en-tête « Article X » | 0,34 | 0,52 | 0,59 | 0,44 |
| **+ titre court du document dans l'en-tête** | **0,41** | **0,62** | **0,66** | **0,52** |

Hit@k : part des questions dont l'article attendu figure dans les k premiers résultats.
MRR : moyenne de 1 / rang du premier bon résultat. Le jeu de questions est encore
petit (une question = 3,4 points) : seuls des écarts nets et cohérents entre les
métriques sont retenus.

### Comparaison des méthodes de recherche

| Méthode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| **Vecteurs seuls (retenue)** | **0,41** | **0,62** | **0,66** | **0,52** |
| Plein texte seul | 0,34 | 0,45 | 0,55 | 0,41 |
| Hybride (fusion RRF) | 0,41 | 0,59 | 0,62 | 0,49 |

L'hybride gagne 5 questions et en perd 5 par rapport aux vecteurs seuls : sans gain
mesuré, il n'est pas retenu. Le plein texte utilise `ts_rank` normalisé par le
logarithme de la longueur, sans IDF ; BM25 n'a pas été testé.

Principaux échecs restants : questions sur le Code du travail ou le Code de la
construction captées par le règlement ERP (83 % des chunks), et articles SSIAP
généraux (4, 5, 6) classés devant l'article ou l'annexe attendus.

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
#    les documents inchangés sont ensuite sautés ; --force pour tout refaire,
#    par exemple après une modification du découpage ou de data/sources.csv)
python -m fire_safety_rag.indexing.indexer

# 3. Mesurer la qualité de la recherche (--mode vecteurs, plein-texte ou hybride)
python -m fire_safety_rag.evaluation.evaluate --mode vecteurs
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

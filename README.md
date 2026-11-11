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
- ✅ **Recherche hybride évaluée** (plein texte PostgreSQL + vecteurs, fusion RRF) :
  non retenue, voir ci-dessous.
- ✅ **Réponses avec citations** (Claude Haiku 5.5) : refus si aucun passage n'est
  assez proche, contrôle des numéros cités, coût journalisé.
- ✅ **Évaluation de la chaîne complète** : 39 questions, voir
  [`docs/evaluation.md`](docs/evaluation.md).
- ✅ **Amélioration de la recherche** : reranker évalué (non retenu, trop lent sur
  processeur), titres de partie rattachés aux articles (retenu), voir
  [`docs/evaluation.md`](docs/evaluation.md) section 6.
- ✅ **API HTTP** (FastAPI) : `POST /questions`, schéma de la base vérifié au démarrage.
- ✅ **Interface web** (Streamlit) : question, réponse, sources numérotées ; elle passe
  par l'API.
- ✅ **Conteneurisation** : base, API et interface lancées ensemble par Docker Compose.
- ⏳ À venir : démo en ligne.

## Résultats de la chaîne complète

Sur 39 questions (29 du corpus, 10 hors corpus), détail dans
[`docs/evaluation.md`](docs/evaluation.md) :

- **10 / 10** questions hors corpus refusées, **0 / 19** refus à tort ;
- aucune réponse n'invente une règle absente des passages ;
- quand la recherche fournit le bon article, la réponse est correcte dans 17 cas sur 19 ;
- point faible : la recherche (Hit@5 = 0,72). Sans le bon article, le LLM répond
  à partir d'un texte voisin, le plus souvent du règlement ERP.

Coût du lancement complet : 0,017 $ (environ 0,0005 $ par question).

## Résultats de la recherche seule

Mesures sur 29 questions ([`data/eval/questions.csv`](data/eval/questions.csv)),
une seule modification par ligne, avec les attendus d'origine :

| Version | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Référence : en-tête « Article X » | 0,34 | 0,52 | 0,59 | 0,44 |
| **+ titre court du document dans l'en-tête** | **0,41** | **0,62** | **0,66** | **0,52** |

Hit@k : part des questions dont l'article attendu figure dans les k premiers résultats.
MRR : moyenne de 1 / rang du premier bon résultat. Le jeu de questions est encore
petit (une question = 3,4 points) : seuls des écarts nets et cohérents entre les
métriques sont retenus.

### Comparaison des méthodes de recherche

Après correction de deux attendus (q10, q12 : un autre article de l'arrêté SSIAP
répond directement, voir [`docs/evaluation.md`](docs/evaluation.md)) :

| Méthode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| **Vecteurs seuls (retenue)** | **0,45** | **0,69** | **0,72** | **0,57** |
| Plein texte seul | 0,34 | 0,45 | 0,59 | 0,41 |
| Hybride (fusion RRF) | 0,45 | 0,62 | 0,66 | 0,53 |

Depuis, avec les titres de partie (Livre, Chapitre, Section…) dans l'en-tête des
passages et un attendu élargi (q02) : **Hit@1 0,55, Hit@3 0,72, Hit@5 0,72,
MRR 0,62** (comparaison à attendus identiques dans
[`docs/evaluation.md`](docs/evaluation.md), section 6.3).

L'hybride n'apporte pas de gain mesuré par rapport aux vecteurs seuls : il n'est
pas retenu. Le plein texte utilise `ts_rank` normalisé par le
logarithme de la longueur, sans IDF ; BM25 n'a pas été testé.

Principaux échecs restants : questions sur le Code du travail ou le Code de la
construction captées par le règlement ERP (83 % des chunks), et articles SSIAP
généraux (4, 5, 6) classés devant l'article ou l'annexe attendus.

## Lancer avec Docker (base + API + interface)

Prérequis : Docker Desktop, le corpus déjà indexé dans la base (voir « Utilisation »)
et un fichier `.env` complet (mot de passe de la base, clé `ANTHROPIC_API_KEY`).

```bash
docker compose up -d --build     # construit l'image puis démarre db, api et ui
docker compose ps                # api "healthy", ui "Up"
docker compose logs -f api       # journaux de l'API (dont le coût de chaque appel)
```

- Interface : http://localhost:8501 — API : http://127.0.0.1:8000/docs
- Premier lancement : quelques minutes (construction de l'image, puis téléchargement
  du modèle d'embedding, ~1 Go, gardé ensuite dans le volume `hf_cache`).
- Arrêt : `docker compose down`. Ne pas ajouter `-v` : cette option supprime les
  volumes, donc la base indexée et le modèle téléchargé.
- Après une modification du code : `docker compose up -d --build`.

Une seule image sert aux deux services ; seule la commande change. L'image ne
contient aucun secret : seul le service `api` reçoit `.env`, au lancement.
L'interface n'a besoin que de l'adresse de l'API (`API_URL=http://api:8000`).

## Démarrage rapide (développement, sans Docker pour l'application)

Prérequis : Python 3.11+, Docker Desktop.

```bash
# Environnement Python
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev,embeddings,ui]"   # "embeddings" : PyTorch ; "ui" : Streamlit (lourds)

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

# 4. Poser une question (clé ANTHROPIC_API_KEY dans .env ; appel payant)
python -m fire_safety_rag.generation.answer "Combien d'extincteurs dans des locaux de travail ?" --details

# 5. Évaluer la chaîne complète (39 questions, appels payants : ~0,02 $)
python -m fire_safety_rag.evaluation.answers --limit 3   # essai
python -m fire_safety_rag.evaluation.answers
```

### API et interface web

Deux terminaux (environnement activé, base démarrée, clé `ANTHROPIC_API_KEY` dans `.env`) :

```bash
# Terminal 1 : l'API (prête quand "Application startup complete." s'affiche)
uvicorn fire_safety_rag.api.main:app
# -> documentation interactive : http://127.0.0.1:8000/docs

# Terminal 2 : l'interface web, qui appelle l'API (adresse réglée par API_URL)
streamlit run app/streamlit_app.py
# -> http://localhost:8501
```

Au démarrage, l'API vérifie que le schéma de la base correspond au code (sans le
modifier) : s'il est en retard, elle s'arrête et indique de relancer l'indexeur.

```text
Navigateur -> Streamlit (8501) -> API FastAPI (8000) -> PostgreSQL + e5 + Claude
```

Les questions dont aucun passage n'atteint le seuil de refus (`REFUSAL_MIN_SCORE`)
sont refusées sans appel au LLM. Chaque appel journalise ses tokens et son coût estimé.

Le premier lancement télécharge le modèle d'embedding (~1 Go) dans le cache
Hugging Face de l'utilisateur, hors du dépôt.

## Organisation du code

| Module | Rôle |
|---|---|
| `ingestion/` | Extraction des PDF, découpage en articles et en chunks, pipeline JSONL |
| `embedding/` | Vectorisation des chunks et des questions (modèle local) |
| `storage/` | Connexion PostgreSQL, schéma des tables, écriture et recherche |
| `indexing/` | Indexation du corpus dans la base |
| `generation/` | Prompt, appel au LLM et journal des coûts, contrôle des citations, commande de réponse |
| `evaluation/` | Métriques (Hit@k, MRR), évaluation de la recherche, choix du seuil, évaluation des réponses |
| `reranking/` | Reclassement par cross-encodeur (évalué, désactivé par défaut) |
| `api/` | API FastAPI : routes, démarrage (schéma, préchauffage), connexion par requête |
| `ui/` | Client HTTP de l'API, utilisé par l'interface |

L'interface Streamlit elle-même est dans [`app/streamlit_app.py`](app/streamlit_app.py)
(affichage seulement, aucune logique de recherche).

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

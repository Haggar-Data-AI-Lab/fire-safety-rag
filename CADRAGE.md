# Projet : fire-safety-rag — Assistant IA sur la réglementation sécurité incendie et sécurité privée

## 1. Contexte

Projet conçu et développé par Haggar Senoussi, étudiant en Master 2 Data / Intelligence Artificielle et agent de sécurité privée. Le sujet vient du terrain : la réglementation sécurité incendie est dispersée dans de longs textes, et retrouver rapidement la bonne règle, avec sa source, est difficile.

Le projet a trois objectifs :

1. **Construire un assistant d'IA générative concret**, déployé et démontrable.
2. **Appliquer au RAG les pratiques du data engineering** : pipelines idempotents, journalisation, gestion des erreurs, tests et intégration continue.
3. **Mesurer la qualité des réponses** par une évaluation reproductible, plutôt que de la supposer.

## 2. Description du produit

Un assistant qui répond en français à des questions sur la réglementation et les bonnes pratiques en sécurité incendie et sécurité privée, en s'appuyant **uniquement** sur un corpus de documents identifiés, et en **citant ses sources** (document, article ou page).

Exemples de questions :
- « Quelles sont les obligations d'un agent SSIAP 1 lors d'une ronde ? »
- « Quelle est la fréquence de vérification des extincteurs ? »
- « Quelles règles s'appliquent à l'évacuation d'un ERP ? »

Comportement attendu :
- Répondre de façon claire et concise, avec les extraits sources.
- Dire « je ne trouve pas cette information dans les documents » plutôt qu'inventer.
- Afficher un avertissement : l'outil est une aide à la recherche, pas un avis juridique.

## 3. Architecture fonctionnelle

### Brique A — Pipeline d'ingestion (partie data engineering)

- Collecte des documents sources (PDF, HTML, texte) dans un dossier dédié.
- Extraction du texte et nettoyage.
- Découpage en passages (« chunks ») avec métadonnées : titre, source, date, article ou page, URL.
- Calcul d'une empreinte (hash) par document pour détecter les doublons et ne retraiter que les documents nouveaux ou modifiés.
- Génération des embeddings et stockage dans PostgreSQL + pgvector.
- Journalisation de chaque exécution (documents traités, ignorés, en erreur) et gestion des erreurs.
- Ré-exécution possible sans créer de doublons (idempotence).

### Brique B — Recherche et génération de réponse (RAG)

- Transformation de la question en embedding.
- Recherche des passages les plus pertinents (recherche vectorielle, puis éventuellement hybride avec mots-clés).
- Construction du prompt avec les passages retrouvés.
- Appel au LLM et génération d'une réponse avec citations.
- Refus de répondre lorsque les passages ne sont pas pertinents.

### Brique C — Évaluation

- Jeu de test de 30 à 50 questions avec réponses attendues et sources attendues.
- Mesures : taux de bonne source retrouvée, fidélité de la réponse aux sources, taux de refus justifiés.
- Script d'évaluation relançable après chaque modification.

### Brique D — Interfaces

- API avec FastAPI (endpoints : `/ask`, `/health`, `/ingest`).
- Interface de démonstration avec Streamlit.

### Brique E — Industrialisation

- Docker et docker-compose (API + base de données + interface).
- Tests automatisés et intégration continue avec GitHub Actions.
- README soigné : objectif, architecture (schéma), captures ou GIF de démo, résultats d'évaluation, instructions d'installation.
- Déploiement d'une démo en ligne si possible avec une offre gratuite ou très peu coûteuse.

## 4. Technologies envisagées

- Python 3.11+ avec environnement virtuel `.venv`.
- PostgreSQL + pgvector (via Docker).
- Modèle d'embeddings : `intfloat/multilingual-e5-base` (open source, licence MIT, multilingue, 768 dimensions), exécuté en local via `sentence-transformers`. Isolé derrière deux fonctions (`embed_passages`, `embed_query`) pour pouvoir le comparer plus tard à une API.
- LLM : API d'un fournisseur (par exemple Claude), modèle économique pour le développement.
- FastAPI, Streamlit, pytest.
- Orchestration (Prefect) seulement si utile, après la première version.
- Git et GitHub.

Ne pas tout installer d'un coup. Expliquer chaque dépendance avant de l'ajouter.

## 5. Corpus de documents

- Utiliser uniquement des documents publics et librement réutilisables : textes réglementaires officiels (Légifrance), guides publics des administrations, documents dont la licence le permet.
- **Ne pas intégrer** de normes payantes (NF, APSAD, etc.) ni de documents internes d'employeur.
- Enregistrer pour chaque document sa source, son URL, sa date et sa licence.
- Commencer avec un petit corpus (10 à 20 documents), puis l'élargir.

## 6. Organisation du dépôt

```
fire-safety-rag/
├── README.md, CADRAGE.md, CLAUDE.md, LICENSE
├── pyproject.toml          # dépendances et configuration des outils
├── .env.example            # modèle de configuration, sans secret
├── docker-compose.yml      # PostgreSQL + pgvector
├── .github/workflows/      # intégration continue (ruff, mypy, pytest)
├── db/init/                # scripts SQL exécutés au premier démarrage
├── data/
│   ├── sources.csv         # catalogue du corpus (versionné)
│   ├── eval/questions.csv  # jeu de questions d'évaluation (versionné)
│   ├── raw/                # documents sources (non versionnés)
│   └── processed/          # données produites (non versionnées)
├── src/fire_safety_rag/
│   ├── config.py
│   ├── ingestion/          # extraction, découpage en articles et en chunks
│   ├── embedding/          # vectorisation (modèle local)
│   ├── storage/            # PostgreSQL : connexion, schéma, écriture, recherche
│   ├── indexing/           # indexation du corpus avec empreinte par document
│   └── evaluation/         # métriques et script d'évaluation de la recherche
└── tests/
```

À venir, au fil des phases : modules `generation/` et `api/` dans `src/fire_safety_rag/`, interface Streamlit dans `app/`, journaux dans `logs/` (non versionnés).

## 7. Sécurité et bonnes pratiques

- Aucun secret dans le code ni dans Git. Fournir un `.env.example` sans clé.
- Plafond de dépenses sur le compte API du LLM.
- Journaliser les appels, les coûts estimés et les erreurs.
- Délais d'expiration et reprises contrôlées sur les appels API.
- Aucune donnée personnelle dans le corpus.

## 8. Budget

- Objectif : moins de 10 € par mois pendant le développement.
- Embeddings en local (gratuit), base de données en local via Docker (gratuit).
- Seuls les appels au LLM sont payants : utiliser un modèle économique et un petit corpus.

## 9. Plan de développement

- **Phase 0 — Socle :** dépôt, `.venv`, structure, configuration, premier test, GitHub Actions. ✅
- **Phase 1 — Ingestion :** extraction, découpage, métadonnées, hash, stockage pgvector, logs. ✅
- **Phase 2 — RAG minimal :** question → passages → réponse avec citations, en ligne de commande.
- **Phase 3 — Évaluation :** jeu de questions, script de mesure, premiers résultats chiffrés. 🔄 Évaluation de la *recherche* avancée avant la Phase 2 : la qualité des passages retrouvés conditionne celle des réponses, autant la mesurer et l'améliorer d'abord. Référence actuelle (29 questions) : Hit@5 = 0,59, MRR = 0,44. L'évaluation des *réponses* (fidélité, refus justifiés) viendra avec la Phase 2.
- **Phase 4 — Interfaces :** API FastAPI, interface Streamlit.
- **Phase 5 — Industrialisation :** Docker, README complet, démo en ligne.
- **Phase 6 — Valorisation :** ligne de CV, post LinkedIn, préparation de la démo pour les entretiens.

## 10. Environnement

- Windows 11, VS Code, Python 3.11 (environnement virtuel `.venv`), Git, Docker Desktop.
- Base de données : image `pgvector/pgvector:pg17` (PostgreSQL 17, extension pgvector).
- Dépôt GitHub public : `Haggar-Data-AI-Lab/fire-safety-rag`.

Les conventions de travail (Git, code, méthode) sont décrites dans `CLAUDE.md`.

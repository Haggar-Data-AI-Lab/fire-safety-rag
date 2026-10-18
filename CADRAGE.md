# Projet : fire-safety-rag — Assistant IA sur la réglementation sécurité incendie et sécurité privée

## 1. Contexte

Je suis Haggar Senoussi, étudiant en Master 2 Data / Intelligence Artificielle à SUPINFO Tours. Je recherche un stage de 6 mois en Data / IA à partir de janvier 2027.

Ce projet a trois objectifs :

1. **Renforcer mon CV et mon portfolio GitHub** avec un projet d'IA générative concret, déployé et démontrable en entretien.
2. **Développer mes compétences** en IA appliquée (LLM, RAG, bases vectorielles, API, déploiement) en m'appuyant sur mon expérience en data engineering (ETL/ELT, Snowflake, Bash, CRON, alertes).
3. **Servir de démonstration** pour de futures prestations d'intégration de l'IA en entreprise.

Le sujet s'appuie sur ma connaissance du terrain : je travaille comme agent de sécurité privée et je fonde une société de sécurité incendie et de gestion des risques au Tchad.

**Échéance visée :** une première version démontrable fin novembre 2026.

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
- Modèle d'embeddings : à évaluer, de préférence un modèle open source multilingue exécuté en local (gratuit).
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

## 6. Organisation du dépôt (indicative)

```
securag/
├── README.md
├── CADRAGE.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── data/
│   ├── raw/          # documents sources (non versionnés si volumineux)
│   └── processed/
├── src/
│   └── securag/
│       ├── config/
│       ├── ingestion/
│       ├── embeddings/
│       ├── storage/
│       ├── retrieval/
│       ├── generation/
│       ├── evaluation/
│       └── api/
├── app/              # interface Streamlit
├── tests/
├── eval/             # jeu de questions de test
├── logs/
└── docs/
```

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
- Aucun achat ni service payant sans mon accord explicite.

## 9. Plan de développement

- **Phase 0 — Socle :** dépôt, `.venv`, structure, configuration, premier test, GitHub Actions.
- **Phase 1 — Ingestion :** extraction, découpage, métadonnées, hash, stockage pgvector, logs.
- **Phase 2 — RAG minimal :** question → passages → réponse avec citations, en ligne de commande.
- **Phase 3 — Évaluation :** jeu de questions, script de mesure, premiers résultats chiffrés.
- **Phase 4 — Interfaces :** API FastAPI, interface Streamlit.
- **Phase 5 — Industrialisation :** Docker, README complet, démo en ligne.
- **Phase 6 — Valorisation :** ligne de CV, post LinkedIn, préparation de la démo pour les entretiens.

## 10. Règles de travail pour l'assistant

1. Travailler étape par étape, avec une approche pédagogique : expliquer l'objectif, le fonctionnement et les choix de chaque étape. Je veux comprendre et être capable d'expliquer le projet en entretien.
2. Examiner le dossier existant avant de créer ou modifier des fichiers. Ne rien supprimer sans vérifier.
3. Ne pas générer tout le projet d'un coup. Une étape limitée et vérifiable à la fois.
4. Expliquer chaque dépendance avant de l'ajouter.
5. Code lisible, typé, documenté et testé.
6. Ne jamais inventer de résultats (évaluations, réponses d'API).
7. Signaler les décisions qui nécessitent mes identifiants, mon accord ou un paiement.
8. Distinguer clairement ce qui fonctionne, ce qui est simulé et ce qui reste à faire.
9. Après chaque étape : fichiers créés ou modifiés, tests effectués, commandes pour vérifier moi-même, prochaine étape proposée.
10. Attendre ma validation avant de passer à l'étape suivante.

## 11. Environnement

- Windows, VS Code, Python, `.venv`, Git, Docker Desktop (à vérifier).
- Dépôt GitHub : `Haggar-Data-AI-Lab/fire-safety-rag` (nom provisoire).

## 12. Première action demandée

1. Vérifier l'environnement : versions de Python, Git, Docker.
2. Proposer la structure minimale de la Phase 0.
3. Lister les dépendances de la Phase 0 et leur rôle.
4. Attendre ma validation avant d'implémenter.

# CLAUDE.md — Conventions de travail du projet

Ce fichier guide l'assistant IA (Claude Code) sur ce dépôt.
Le cadrage complet du projet est dans `CADRAGE.md`.

## Projet

`fire-safety-rag` : assistant RAG en français sur la réglementation
sécurité incendie et sécurité privée, avec citation des sources.
Stack : Python 3.11, PostgreSQL + pgvector, FastAPI, Streamlit, pytest.

## Méthode de travail

Le développeur construit ce projet pour apprendre et doit pouvoir
en expliquer chaque partie. En conséquence :

1. Travailler par petites étapes vérifiables, une à la fois.
2. Expliquer l'objectif et le fonctionnement avant de proposer du code.
3. Privilégier les explications et la relecture ; ne pas générer
   plusieurs fichiers ou de longs blocs de code d'un coup.
4. Expliquer toute nouvelle dépendance avant de l'ajouter.
5. Attendre la validation explicite avant de passer à l'étape suivante.
6. Ne jamais inventer de résultats (tests, évaluations, réponses d'API).
7. Signaler toute action nécessitant des identifiants, un paiement
   ou un accord.
8. Examiner l'existant avant de créer ou modifier un fichier ;
   ne rien supprimer sans vérifier.
9. Distinguer clairement ce qui fonctionne, ce qui est simulé
   et ce qui reste à faire.
10. Après chaque étape : fichiers créés ou modifiés, tests effectués,
    commandes pour vérifier soi-même, prochaine étape proposée.

## Git

- Workflow GitHub Flow : `main` toujours stable, une branche par sujet,
  fusion via Pull Request.
- Branches : `feat/`, `fix/`, `docs/`, `chore/`, `test/` + nom court.
- Commits au format Conventional Commits, en français
  (ex. `feat: ajoute le chargement de la configuration`).
- Un commit = un changement cohérent.
- L'assistant n'exécute JAMAIS d'opération Git qui modifie le dépôt ou
  l'historique (`git add`, `commit`, `push`, `merge`, `rebase`, `switch -c`,
  création de PR). Le développeur les réalise lui-même, manuellement.
- L'assistant peut utiliser Git en lecture seule (`git status`, `git diff`,
  `git log`) pour comprendre l'état du dépôt.
- À la fin de chaque étape, l'assistant liste les fichiers créés ou modifiés
  et propose un message de commit, sans l'exécuter.

## Code

- Code typé (annotations), documenté (docstrings) et testé (pytest).
- Vérifications avant chaque commit : `ruff check`, `ruff format`,
  `mypy`, `pytest`.
- Organisation par responsabilité dans `src/fire_safety_rag/`.
- Pas de design pattern sans besoin réel et justifié.

## Sécurité

- Aucun secret dans le code ni dans Git : utiliser `.env`
  (ignoré par Git) et maintenir `.env.example` à jour.
- Aucune donnée personnelle dans le corpus.
- Corpus : uniquement des documents publics et librement réutilisables.

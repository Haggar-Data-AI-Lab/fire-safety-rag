# Évaluation de la chaîne complète (novembre 2026)

Mesures de l'assistant de bout en bout : recherche, seuil de refus, génération
de la réponse par le LLM, contrôle des citations. Configuration évaluée :

| Élément | Valeur |
|---|---|
| Recherche | vecteurs seuls, `intfloat/multilingual-e5-base`, 5 passages |
| Seuil de refus | 0,84 (similarité cosinus du meilleur passage) |
| LLM | Claude Haiku 5.5, effort `low` |
| Questions | 29 du corpus ([`questions.csv`](../data/eval/questions.csv)) + 10 hors corpus ([`hors_corpus.csv`](../data/eval/hors_corpus.csv)) |
| Coût du lancement complet | 0,017 $ pour 33 appels (0,0005 $ par appel) |

Commandes pour reproduire :

```bash
python -m fire_safety_rag.evaluation.threshold   # choix du seuil (gratuit)
python -m fire_safety_rag.evaluation.answers     # chaîne complète (appels payants)
```

## 1. Recherche

| Méthode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| **Vecteurs seuls (retenue)** | **0,45** | **0,69** | **0,72** | **0,57** |
| Hybride (fusion RRF) | 0,45 | 0,62 | 0,66 | 0,53 |
| Plein texte seul | 0,34 | 0,45 | 0,59 | 0,41 |

Ces chiffres tiennent compte de la correction de deux questions (q10, q12) :
la relecture des réponses a montré qu'un autre article que celui attendu y
répondait directement (articles 5 et 4 de l'arrêté SSIAP). Ces articles sont
désormais acceptés. Les mesures antérieures du README étaient calculées avec
les attendus d'origine et ne sont pas comparables ligne à ligne.

## 2. Seuil de refus

Le meilleur score de chaque question, dans trois groupes :

| Groupe | Score le plus bas | Score le plus haut |
|---|---|---|
| Trouvées (bon article dans le top 5) | 0,8438 | 0,8923 |
| Manquées (bon article absent du top 5) | 0,8521 | 0,8874 |
| Hors corpus | 0,8031 | 0,8488 |

Avec un seuil de 0,84 : aucune question trouvée refusée, 6 questions hors corpus
sur 10 refusées sans appel au LLM.

- La marge est faible : la question trouvée la plus basse (q23) n'est qu'à 0,004
  au-dessus du seuil.
- **Le score ne détecte pas les ratés de la recherche** : les questions manquées
  ont des scores aussi élevés que les trouvées. Le score mesure la ressemblance,
  pas la présence de la réponse.
- Le seuil dépend du modèle d'embedding : il est à remesurer si celui-ci change.

## 3. Comportement de la chaîne

| Mesure | Résultat |
|---|---|
| Questions hors corpus refusées | **10 / 10** (6 par le score, 4 par le LLM) |
| Questions trouvées refusées à tort | **0 / 19** |
| Questions manquées refusées | 0 / 10 |
| Réponses du corpus citant un article attendu | 19 / 29 (les 19 trouvées) |
| Citations inconnues, réponses sans citation, réponses coupées | 0 |

Les deux filets de refus se complètent : le score écarte les questions
éloignées (permis de conduire, TVA), le LLM refuse les questions proches mais
sans réponse (carte professionnelle du CNAPS, légionellose).

## 4. Relecture des réponses

Chaque réponse a été comparée au texte des articles cités, sur deux critères
distincts :

- **fidèle** : chaque affirmation est appuyée par les passages cités ;
- **répond à la question** : la réponse répond à la question posée, avec le bon texte.

Une réponse peut être fidèle sans répondre à la question : elle restitue
correctement un article qui ne s'applique pas au cas posé.

| Groupe | Fidèle (oui / partiel / non) | Répond à la question (oui / partiel / non) |
|---|---|---|
| Trouvées (19) | 18 / 1 / 0 | 17 / 2 / 0 |
| Manquées (10) | 7 / 3 / 0 | 2 / 6 / 2 |
| Refus du LLM (4) | 4 / 0 / 0 | 4 / 0 / 0 (refus justifiés) |

> La relecture a été faite par l'assistant de développement (Claude), à partir
> des textes du corpus, et non par un relecteur humain indépendant. Elle ne peut
> donc pas servir de référence pour valider un futur LLM-juge sans une seconde
> relecture humaine.

### Ce qui fonctionne

- Aucune réponse n'invente une règle absente des passages.
- Quand la recherche fournit le bon article, la réponse est correcte dans 17 cas
  sur 19 ; les 2 autres sont incomplètes, pas fausses.
- Le modèle signale souvent lui-même ses limites (« les passages ne précisent
  pas… »).

### Défauts observés

1. **Généralisation d'une règle propre à un type d'ERP** (défaut de fidélité le
   plus fréquent) : une règle des hôtels-restaurants d'altitude (OA 4) présentée
   comme valable pour tous les établissements, ou les règles du registre de
   sécurité des structures gonflables et chapiteaux appliquées à tout ERP.
2. **Réponse fondée sur le mauvais texte** quand la recherche a raté : la règle
   des chapiteaux (CTS 10) appliquée à un bureau, au lieu de l'article R4227-5
   du Code du travail. La réserve n'arrive qu'en fin de réponse.
3. **Débordement** : une question sur les locaux de travail reçoit aussi la
   règle des ERP (signalée comme telle).
4. **Variabilité** : la même question (q01) a reçu une réponse nuancée lors d'un
   essai et une réponse affirmative lors du lancement complet.

## 5. Conclusion

Le point faible est la **recherche**, pas la génération : environ un quart des
questions arrivent au LLM sans le bon article (Hit@5 = 0,72), et le LLM produit
alors une réponse plausible à partir d'un texte voisin, le plus souvent du
règlement ERP, qui représente 83 % des chunks.

Pistes, par ordre de priorité :

1. **Reranker** : relire la question et chaque passage ensemble pour mieux
   classer les candidats ; les scores cosinus actuels sont trop serrés pour
   distinguer le bon article d'un article voisin.
2. **Consigne** : demander au LLM de vérifier que le type d'établissement visé
   par un article correspond à la question, avant de l'appliquer.
3. **Corpus** : rééquilibrer en ajoutant des textes (objectif de 10 à 20 documents).

## Limites de cette évaluation

- Une seule passe : un LLM ne répond pas deux fois de la même façon.
- 39 questions : chaque question pèse environ 3 points de pourcentage.
- Seuil réglé sur les mêmes questions que celles qui servent à le mesurer.
- Relecture faite par l'assistant de développement, pas par un expert du domaine.

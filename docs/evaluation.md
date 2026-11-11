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

Ce tableau décrit l'index évalué avec la chaîne complète. La recherche actuelle
(sections de document dans l'en-tête, attendus de q02 élargis) est mesurée en
section 6.3 : Hit@1 0,55, Hit@5 0,72, MRR 0,62.

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
   distinguer le bon article d'un article voisin. *Testé depuis : voir la
   section 6.*
2. **Consigne** : demander au LLM de vérifier que le type d'établissement visé
   par un article correspond à la question, avant de l'appliquer.
3. **Corpus** : rééquilibrer en ajoutant des textes (objectif de 10 à 20 documents).

## 6. Améliorations de la recherche testées

Les sections 1 à 5 décrivent la chaîne telle qu'évaluée début novembre. Cette
section suit les pistes de la conclusion. Recherche seule, 29 questions.

### 6.1 Reranker (cross-encodeur local) : non retenu

Le reranker relit la question avec chacun des N premiers passages vectoriels et
les reclasse. Il note chaque couple indépendamment des autres : les notes des
30 candidats, enregistrées une fois (`evaluate --save-candidates`), permettent
de recalculer hors ligne le résultat pour 5 à 30 candidats
(`python -m fire_safety_rag.evaluation.candidates`). Seul le temps est alors
estimé (au prorata), pas mesuré. Processeur seul, 8 cœurs, sans carte graphique.

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Temps / question |
|---|---|---|---|---|---|
| Vecteurs seuls | 0,45 | 0,69 | 0,72 | 0,57 | 0,1 s |
| `BAAI/bge-reranker-v2-m3`, 30 candidats | 0,66 | 0,76 | 0,83 | 0,72 | 43,8 s (mesuré) |
| `BAAI/bge-reranker-v2-m3`, 5 candidats | 0,59 | 0,69 | 0,72 | 0,64 | ~7 s (estimé) |
| `mmarco-mMiniLMv2-L12-H384-v1`, 30 candidats | 0,55 | 0,72 | 0,76 | 0,63 | 4,2 s (mesuré) |
| `mmarco-mMiniLMv2-L12-H384-v1`, 5 candidats | 0,59 | 0,72 | 0,72 | 0,65 | ~0,7 s (estimé) |

- bge améliore nettement la recherche, mais 44 s par question le rend
  inutilisable en interactif sur processeur.
- mMiniLM est 10 fois plus rapide, mais ne profite pas de candidats
  supplémentaires : son meilleur réglage est 5 candidats.
- Le LLM reçoit 5 passages : avec 5 candidats, le reranker ne change que leur
  **ordre**, pas leur contenu. Seul bge à 30 candidats élargit le contexte
  (Hit@5 0,72 → 0,83).

Le reranker reste dans le code, désactivé par défaut (`RERANKER_MODEL` vide),
et pourra être réévalué avec une carte graphique ou une version ONNX.

### 6.2 Diagnostic du rappel

Pour 5 questions, aucun article attendu ne figurait parmi les 30 premiers
passages vectoriels : aucun reranker ne pouvait les retrouver. Rang de
l'article attendu jusqu'à 200 (`evaluate --k 200`) :

| Question | Vecteurs | Plein texte | Cause probable (relecture des passages) |
|---|---|---|---|
| q02 portes vers l'extérieur | 51 | > 200 | Vocabulaire : l'article dit « dans le sens de la sortie » ; et CO 45 (ERP) répond aussi |
| q19 catégorie d'un ERP | 40 | 5 | Article général du CCH noyé parmi les articles ERP |
| q20 qui autorise l'ouverture | > 200 | 17 | « Le maire autorise l'ouverture… » : ni « ERP » ni « établissement » dans l'article |
| q22 contenu du registre | > 200 | 37 | Articles généraux noyés parmi les nombreux articles ERP sur le registre |
| q26 fréquence des visites | 46 | 9 | La réponse (GE 4) est un tableau devenu illisible à l'extraction du PDF |

La relecture a aussi montré un défaut du découpage : les titres de partie
(« Section 3 : Registre de sécurité incendie (Articles R141-10 à R141-11) »)
étaient collés à la **fin de l'article précédent**, au lieu de donner leur
contexte aux articles suivants (plusieurs centaines d'articles concernés,
surtout dans le règlement ERP).

CO 45 a été ajouté aux attendus de q02 : la question ne précise pas s'il s'agit
d'un lieu de travail ou d'un ERP, et CO 45 donne la même règle pour les ERP.

### 6.3 Titres de partie rattachés aux articles : retenu

Les titres sont retirés de la fin des articles et forment la `section` des
articles suivants (ex. « Chapitre III : ÉTABLISSEMENTS RECEVANT DU PUBLIC ›
Section 2 : Classement des établissements »), ajoutée à l'en-tête du texte
vectorisé. Comparaison à attendus identiques (CO 45 inclus) :

| Index | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Avant (1907 passages) | 0,45 | 0,72 | 0,76 | 0,58 |
| **Avec les sections (1898 passages, retenu)** | **0,55** | **0,72** | **0,72** | **0,62** |

- Mieux classées : q01 (absente → rang 3), q02, q07, q18, q29 (→ rang 1),
  q14, q21.
- Moins bien classées : q23 (1 → absente), q27 (2 → absente), q12 (2 → 3).
- **Les trois questions visées (q19, q20, q22) restent absentes.** Explication
  probable, non vérifiée : les passages ERP reçoivent eux aussi un en-tête qui
  mentionne les établissements recevant du public (« Livre Ier : Dispositions
  applicables à tous les établissements recevant du public »).

Retenu pour le gain en tête de classement et parce que le texte des articles
est désormais propre, malgré une question de moins dans le top 5 (l'écart
reste dans le bruit : une question vaut 0,034).

Le seuil de refus, lié aux vecteurs, a été remesuré après la réindexation : le
seuil de 0,84 est conservé. Aucune question trouvée n'est refusée (0 / 21), et
7 questions hors corpus sur 10 sont refusées sans appel au LLM (6 avant). La
question trouvée la plus basse (q01, 0,8467) est à 0,007 au-dessus du seuil ;
la question hors corpus la plus haute (h01, 0,8446) passe juste au-dessus et
reste confiée au LLM. Un seuil à 0,845 refuserait h01, mais avec une marge de
0,002 sur q01 : trop fragile pour être retenu.

### Bilan

Le frein principal reste le **déséquilibre du corpus** (83 % de passages ERP)
face aux questions générales sur les ERP, dont les réponses sont dans le CCH.
Pistes non testées : ajouter des documents, filtrer ou réserver des places par
document, réparer l'extraction des tableaux.

## Limites de cette évaluation

- Une seule passe : un LLM ne répond pas deux fois de la même façon.
- 39 questions : chaque question pèse environ 3 points de pourcentage.
- Seuil réglé sur les mêmes questions que celles qui servent à le mesurer.
- Relecture faite par l'assistant de développement, pas par un expert du domaine.

# pln_proyecto2

```bash
proyecto/
  data/        NAP.xls, corpus/ (276 anciennes), nuevas_desc.csv
  nlp_utils.py     limpia_lematiza, conceptos
  grafo.py         Construye_Grafo, reducir_grafo, estadísticas
  modelos.py       centralités, PageRank, node2vec, RRF
  evaluacion.py    p@k, MRR, cobertura, tableau final
  main.ipynb       exécute tout, génère les tableaux
  README.md, requirements.txt
```

## Répartition en trois personnes
#Personne 1 : données, évaluation, rapport
Jour 1 : créer le dépôt, le dossier partagé et la structure ci-dessus. Définir le format nuevas_desc.csv (concepto, definicion, fuente, particion) et le partager.
Jour 1 : lancer la collecte des 150 descriptions (tout le monde y participe, voir plus bas).
Jour 1-2 : écrire evaluacion.py : evaluar(modelo, datos) qui retourne p@1/3/5, MRR et couverture (% de définitions avec au moins un lemme dans le graphe, nombre moyen de lemmes utiles). Corriger le lineas[1:] pour exclure le nom du concept. Livrer cette fonction en premier, les deux autres en ont besoin.
Jour 2 : exclure ou signaler les descriptions qui contiennent le mot cible. Vérifier que chaque concept cible existe comme nœud du graphe.
Jour 2-3 : statistiques du graphe : nœuds, arêtes, degré moyen, composantes connexes, coefficient de regroupement, graphique de la distribution des degrés.
Jour 4-5 : tableau comparatif unique (tous les modèles et pondérations + lignes du Tableau 1), puis rapport et structure de la présentation.

# Personne 2 : modèles de graphes + PageRank
Jour 1 : réécrire Construye_Grafo pour qu’il charge NAP.xls une seule fois et stocke deux attributs par arête : dist (pour les chemins : 1/frec, tiempo, 100-FA) et sim (la force, utilisée par PageRank et node2vec : frec, FA...). Justifier la transformation similarité → distance par écrit.
Jour 1-2 : nettoyer et refactoriser les 3 centralités sous la forme modelo(definicion), en graphe complet et réduit.
Jour 2-3 : PageRank personnalisé : nx.pagerank(G, personalization={lema: 1 ...}, weight="sim") sur le graphe complet. Régler alpha (probabilité de continuer à marcher, par exemple 0,5 à 0,9) sur le jeu de dév. Exclure les mots de la définition, comme dans conceptos().
Jour 3-4 : lancer toutes les évaluations sur dév puis test.
Jour 4 (si le temps le permet) : graphe orienté (nx.DiGraph, arête estimulo → respuesta), comparer avec le non orienté.
Jour 5 : répondre aux questions de discussion sur la pondération et sur « proximité contre intermédiarité ».

# Personne 3 : node2vec, fusion, démo
Jour 1-2 : refaire l’entraînement node2vec avec les poids sim (c’est un point critique : avec les distances, les associations fortes sont les moins parcourues). Fixer seed et workers=1 pour la reproductibilité.
Jour 2-3 : au moins 3 configurations (par exemple dimension 64/128/300, p et q valant 0,5/1/2, longueur de marche 30/80, nombre de marches 10/20), tableau des résultats sur dév. Comparer somme, moyenne et somme pondérée IDF des vecteurs.
Jour 3-4 : RRF : score[mot] += 1/(60+rang) sur la liste du meilleur modèle de graphe (probablement proximité, graphe réduit, association) et sur celle du meilleur node2vec. Tester aussi une fusion à trois (avec PageRank).
Jour 4-5 : analyse « quand gagne node2vec ou le graphe ? » (question 5).
Jour 5-6 : démo en direct (même simple : une cellule input() qui affiche le top 5). Gradio vaut +3 points si vous avez le temps.
Tout le monde
Jour 1-2 : ≈ 50 descriptions chacun, écrites par des proches qui ne connaissent pas le mot et sans le nommer. Pas de LLM pour les générer ni pour simuler des participants.
Jour 5 : chacun analyse 5 cas d’erreur, classés par cause (lemmatisation, hors vocabulaire, polysémie, nœuds très connectés, description ambiguë).
Jour 6 : répétition de la présentation (15 min). N’importe qui peut être interrogé sur n’importe quelle partie, donc relisez le code des autres. Déclarez l’usage de l’IA.

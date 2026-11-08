# pln_proyecto2


# Diccionario inverso con normas de asociación (NAP): PageRank personalizado y fusión de rankings

**Proyecto de la Unidad 2 · PLN · Modalidad C (extensión metodológica sobre NAP)**
Equipo: Celia (P1), Léa (P2), Max (P3)

> Convención de este borrador: `[PENDIENTE: nombre]` indica una sección que debe redactar esa persona. Las secciones sin marca ya están redactadas por Léa (P2).

---

## 1. Introducción
[PENDIENTE: Celia (P1)]
- Descripción del problema (diccionario inverso, fenómeno de la «punta de la lengua»).
- Modalidad elegida (C) y las dos extensiones: **PageRank personalizado** y **fusión de rankings (reciprocal rank fusion) entre la mejor medida de centralidad y node2vec**. Si el tiempo alcanza, grafo dirigido.

## 2. Datos
[PENDIENTE: Celia (P1)]
- NAP (normas de asociación del español de México): descripción del recurso.
- Conjunto de desarrollo: corpus de clase (276 descripciones de 56 conceptos).
- Conjunto de prueba: ≥ 150 descripciones nuevas recolectadas por el equipo (cómo se recolectaron, cuántas se descartaron por contener la palabra objetivo).
- Formato de los CSV y diccionario de datos.

## 3. Construcción y caracterización del grafo

### 3.1 Grafo
Cada par estímulo–respuesta de NAP es una arista no dirigida entre dos nodos (palabras). Se construyeron tres grafos con la misma estructura (**4 239 nodos y 12 872 aristas** en el grafo de asociación), que difieren únicamente en el peso de las aristas. Implementación en Python con `networkx`.

### 3.2 Esquemas de ponderación y transformación similitud → distancia
Los algoritmos de caminos más cortos (intermediación, cercanía) interpretan el peso de una arista como un **costo**: a mayor peso, mayor distancia. En cambio, PageRank y node2vec lo interpretan como una **fuerza**: a mayor peso, mayor probabilidad de recorrer esa arista. Por eso cada arista guarda dos atributos:

| Grafo | Valor original en NAP | `weight` (distancia, para caminos más cortos) | `sim` (fuerza, para PageRank y node2vec) |
|---|---|---|---|
| Asociación | fuerza de asociación FA | `100 − FA` | `FA` |
| Frecuencia | frecuencia f | `1 / f` | `f` |
| Tiempo | tiempo de respuesta t | `t` | `1 / (t + 1)` |

Justificación:
- **Asociación:** una FA alta indica una asociación fuerte, es decir, una distancia corta; `100 − FA` invierte la escala (FA = 80 → distancia 20; FA = 5 → distancia 95).
- **Frecuencia:** una frecuencia alta indica asociación fuerte; `1/f` la convierte en distancia pequeña.
- **Tiempo:** una respuesta rápida indica una asociación fuerte, por lo que el tiempo ya funciona como distancia y se usa directamente en `weight`. Para la fuerza se usa `1/(t+1)`: el `+1` evita la división entre cero, ya que NAP contiene tiempos iguales a 0.

### 3.3 Estadísticas del grafo
[PENDIENTE: Celia (P1)]
- Número de nodos y aristas (ya disponible: 4 239 y 12 872), grado medio, número y tamaño de componentes conexas, coeficiente de agrupamiento y gráfica de la distribución de grado.

### 3.4 Grafo reducido
Para cada definición, el **grafo reducido** conserva solo los lemas de la definición y sus vecinos directos. Se usa por dos razones:
1. **Costo computacional:** el grafo completo tiene 4 239 nodos y las medidas basadas en caminos más cortos son costosas; el subgrafo tiene unas pocas decenas de nodos.
2. **Pertinencia:** en el grafo completo, la cercanía y el grado de una palabra no dependerían de la definición (siempre ganarían las palabras más conectadas). Restringir el grafo a la vecindad de la definición hace que estas medidas sean específicas de cada consulta. La palabra buscada casi siempre es vecina de algún lema de la definición.

## 4. Preprocesamiento de las definiciones
[PENDIENTE: Celia (P1)]
- Pipeline: eliminación de signos de puntuación, palabras funcionales (NLTK) y lematización (spaCy `es_core_news_sm`).
- Manejo de palabras fuera del vocabulario del grafo.
- Cobertura: porcentaje de definiciones con al menos un lema en el grafo y número promedio de lemas útiles por definición.

## 5. Modelos

### 5.1 Modelos basados en centralidad (P2)
En todos los casos, los lemas de la definición se excluyen de la lista de candidatos y solo se conservan los candidatos con puntaje mayor que cero (función `conceptos()` de la clase).

| Modelo | Grafo | Qué mide | Peso usado |
|---|---|---|---|
| **Cercanía** (`diccionario_cc`) | reducido | distancia promedio de cada palabra al resto de la vecindad de la definición | `weight` |
| **Intermediación por subconjunto** (`diccionario_btc`) | completo | cuántos caminos más cortos entre los lemas de la definición pasan por cada palabra | `weight` |
| **Intermediación** (`diccionario_btc_reducido`) | reducido | lo mismo, pero calculado solo sobre la vecindad de la definición | `weight` |
| **Grado** (`diccionario_dc`) | reducido | número de vecinos de cada palabra en la vecindad de la definición | sin peso |

Se descartaron Katz, vector propio y comunicabilidad: en los resultados de referencia de la clase (Tabla 1) son las medidas con menor desempeño y las de mayor costo (Katz tardó cerca de 16 minutos en la evaluación de clase).

### 5.2 PageRank personalizado (extensión 1, P2)
Se imagina un caminante aleatorio que recorre el grafo siguiendo las aristas, con preferencia por las de mayor `sim`. En cada paso, con probabilidad `alpha` continúa por una arista y con probabilidad `1 − alpha` **reinicia en uno de los lemas de la definición**. La distribución personalizada asigna probabilidad de reinicio solo a esos lemas. El puntaje de cada palabra es la fracción de tiempo que el caminante pasa en ella; las palabras con mayor puntaje (excluyendo los lemas de la definición) son los candidatos. Se aplica sobre el grafo completo.

Implementación: `nx.pagerank(grafo, alpha=alpha, personalization=pers, weight="sim")`.

**Ajuste de `alpha` (conjunto de desarrollo, grafo de asociación):**

| alpha | p@1 | p@3 | p@5 |
|---|---|---|---|
| 0.2 | 0.420 | 0.598 | 0.696 |
| 0.35 | 0.420 | 0.601 | 0.696 |
| **0.5** | **0.424** | 0.609 | **0.707** |
| 0.7 | **0.424** | **0.612** | 0.696 |
| 0.85 | 0.409 | 0.598 | 0.685 |
| 0.9 | 0.370 | 0.580 | 0.667 |

Se fijó **alpha = 0.5**. Entre 0.2 y 0.7 las diferencias son mínimas (con 276 descripciones, un acierto más o menos cambia p@1 en 0.0036), por lo que el modelo es poco sensible a `alpha` en ese rango y solo se degrada con valores altos (≥ 0.85), cuando el caminante se aleja demasiado de la definición y ganan las palabras muy conectadas. El valor se eligió con el conjunto de desarrollo y **no se modificó** tras evaluar el conjunto de prueba.

### 5.3 node2vec
[PENDIENTE: Max (P3)]
- Entrenamiento sobre el grafo usando la fuerza `sim` como peso (no la distancia), semillas fijas, al menos 3 configuraciones de hiperparámetros (dimensión, p, q, longitud y número de caminatas), representación de la definición por suma vectorial (y variantes), similitud coseno.

### 5.4 Fusión de rankings (extensión 2)
[PENDIENTE: Max (P3)]
- Reciprocal rank fusion entre el mejor modelo de grafo (PageRank personalizado) y el mejor modelo node2vec: `score(palabra) = Σ 1 / (k + rango)`.

## 6. Evaluación y resultados

### 6.1 Métricas
[PENDIENTE: Celia (P1)]
- p@1, p@3, p@5, MRR y cobertura; separación entre desarrollo y prueba.

### 6.2 Resultados en el conjunto de desarrollo (corpus de clase, 276 descripciones), modelos de grafos
Resultados de Léa (P2). Se incluyen los de referencia de la Tabla 1 de la consigna para comparar.

| Modelo | Grafo | Ponderación | p@1 | p@3 | p@5 |
|---|---|---|---|---|---|
| Grado | reducido | Asociación | 0.116 | 0.250 | 0.330 |
| Cercanía | reducido | Asociación | 0.221 | 0.428 | 0.518 |
| Cercanía | reducido | Frecuencia | 0.196 | 0.355 | 0.514 |
| Cercanía | reducido | Tiempo | 0.076 | 0.214 | 0.286 |
| Intermediación por subconjunto | completo | Asociación | **0.264** | 0.406 | 0.446 |
| Intermediación | reducido | Asociación | 0.167 | 0.362 | 0.449 |
| Intermediación | reducido | Frecuencia | 0.138 | 0.301 | 0.362 |
| Intermediación | reducido | Tiempo | 0.083 | 0.207 | 0.261 |
| **PageRank personalizado (alpha 0.5)** | completo | Asociación | 0.424 | 0.609 | 0.707 |
| **PageRank personalizado (alpha 0.5)** | completo | Frecuencia | 0.424 | 0.605 | 0.696 |
| **PageRank personalizado (alpha 0.5)** | completo | Tiempo | 0.196 | 0.380 | 0.529 |

*Nota de verificación:* las filas de cercanía (asociación y frecuencia) e intermediación coinciden con la Tabla 1 de la consigna, lo que indica que la evaluación es equivalente a la de clase. En grado (0.116 / 0.250 / 0.330 frente a 0.109 / 0.239 / 0.362) y cercanía con tiempo (0.076 frente a 0.083 en p@1) hay pequeñas diferencias, que probablemente se deben al orden de nodos con puntajes empatados. Lo mismo ocurre con la intermediación en grafo reducido (0.167 / 0.362 / 0.449 frente a 0.159 / 0.344 / 0.431 en la Tabla 1).

### 6.2b Resultados en el conjunto de prueba (corpus nuevo), modelos de grafos
[PENDIENTE: Léa, cuando Celia entregue el corpus nuevo]
- Una sola ejecución, sin modificar nada: cercanía, grado, intermediación (completa y reducida) y PageRank personalizado con alpha = 0.5, grafo de asociación.

### 6.3 Tabla comparativa única (todos los modelos, desarrollo y prueba)
[PENDIENTE: Celia (P1)]
- Incluir modelos de grafos, node2vec, RRF, MRR y cobertura, y el conjunto de prueba (corpus nuevo).

## 7. Análisis de errores
Se analizarán al menos 15 casos (5 por integrante), clasificados por causa (lematización, palabras fuera del vocabulario, polisemia, sesgo hacia nodos muy conectados, descripciones ambiguas).
### 7.1 Casos de Léa (P2): PageRank personalizado, grafo de asociación, alpha = 0.5
Casos tomados del corpus de clase (conjunto de desarrollo). «Posición» es el lugar que ocupa el concepto correcto en la lista de candidatos (el modelo acierta p@5 si es ≤ 5).

| # | Concepto | Descripción | Lemas (fuera del grafo) | Posición | Top 5 | Causa |
|---|---|---|---|---|---|---|
| 1 | guajolote | «Se acompaña con mole» | él, acompañar, mo, él (acompañar, mo) | no aparece | botella, agua, vino, alcohol, vidrio | **Lematización y palabra funcional con mayúscula.** «mole» se lematiza como «mo» y «Se» no se filtra como palabra funcional, así que se convierte en el lema «él» (que sí es un nodo del grafo). La consulta queda casi sin contenido útil. |
| 2 | queso | «El producto que se saca de la leche de la vaca» | el, producto, saco, leche, vacar (el, vacar) | 11 | botón, dinero, tienda, vaca, mercado | **Lematización palabra por palabra.** «saca» se lematiza como «saco» (sustantivo) y «vaca» como «vacar»; al no reconocerse «vaca» como lema de la definición, **aparece entre los candidatos pese a estar en la descripción**. Además «producto» es un nodo genérico (dinero, tienda, mercado). |
| 3 | guajolote | «el animal que dice "gordogordogordo"» | animal, decir, gordogordogordo (decir, gordogordogordo) | 8 | hipopótamo, oso, jirafa, venado, cochino | **Palabra fuera del vocabulario** (onomatopeya y verbo «decir»). Solo queda «animal», un nodo muy general, y el modelo devuelve animales en general. |
| 4 | plancha | «el aparato con el que la ropa queda lisa» | aparato, ropa, quedar, lisa (quedar) | 10 | cara, cámara, televisión, ba_t, mano | **Sesgo hacia nodos muy conectados.** «aparato» está enlazado con muchos aparatos (cámara, televisión) y probablemente «lisa» se asocia con partes del cuerpo (cara, mano) más que con la ropa. |
| 5 | radio | «Programa» | programa | 12 | televisión, computador, entretenimiento, película, ver | **Descripción demasiado corta y ambigua.** Con un solo lema, PageRank devuelve vecinos plausibles de «programa» (medios de entretenimiento), pero no distingue «radio». |

Caso adicional: borrego, «animal que produce lana» (posición 27, top 5: hipopótamo, oso, jirafa, venado, cochino): «lana» se lematiza como «lán» y «producir» queda fuera del grafo, por lo que solo queda «animal» (como en el caso 3).

### 7.2 Hallazgos generales (P2)
1. **Palabras funcionales con mayúscula inicial.** `limpia_lematiza` compara cada palabra con la lista de palabras funcionales (en minúsculas) *antes* de pasar el texto a minúsculas, por lo que «La», «El» o «Se» no se eliminan y se convierten en los lemas «el» o «él», que son nodos del grafo y contaminan la consulta (casos 1 y 2, y también en «gelatina»).
2. **Lematización palabra por palabra, sin contexto.** Cada palabra se pasa sola a spaCy, lo que produce lemas incorrectos: mole → mo, saca → saco, vaca → vacar, lana → lán, alpaca → alpacar.
3. **Vocabulario.** Los verbos (acompañar, decir, quedar, producir) y algunos sustantivos (agujero, cuaja, similar) no están en el grafo, de modo que a menudo solo sobrevive un lema genérico.
4. **Nodos genéricos.** Palabras como «animal», «aparato» o «producto» arrastran a los mismos vecinos muy conectados (hipopótamo, oso, cámara, televisión) sin importar el concepto.
5. **Casi aciertos.** En varios errores el concepto aparece cerca del top 5 (posiciones 6 a 12), o el modelo devuelve una variante (en «aspiradora», «aspirador»), por lo que p@5 subestima en parte la utilidad del sistema.

Mejoras posibles, **no aplicadas** para mantener la comparabilidad con el pipeline de clase y la Tabla 1: pasar a minúsculas antes de filtrar palabras funcionales y lematizar la frase completa (con contexto) en lugar de palabra por palabra.
- Casos de Celia (P1): [PENDIENTE: Celia (P1)]
- Casos de Max (P3): [PENDIENTE: Max (P3)]

## 8. Discusión

### Pregunta 2. ¿Por qué la cercanía en el grafo reducido mejora p@5 respecto a la intermediación, pero no p@1?
Resultados: cercanía (reducido) 0.221 / 0.428 / 0.518 frente a intermediación (completo) 0.264 / 0.406 / 0.446. Si la intermediación también se calcula sobre el grafo reducido (0.167 / 0.362 / 0.449), la cercanía gana en los tres valores de k: la ventaja de la intermediación en p@1 aparece solo cuando se calcula sobre el grafo completo.
*Hipótesis (por verificar con los casos de error):* la intermediación por subconjunto asigna puntaje positivo solo a las palabras que están en caminos más cortos entre los lemas de la definición, y `conceptos()` descarta los puntajes nulos; por eso produce listas de candidatos cortas, y cuando acierta lo hace en primer lugar, pero con pocas alternativas si falla. La cercanía puntúa a todas las palabras de la vecindad, lo que da listas más largas y una buena palabra aparece entre los primeros cinco con más frecuencia, aunque no siempre ocupe el primer lugar. [Verificar contando la longitud promedio de las listas de cada modelo.]
¿Ocurre lo mismo con nuestro recurso? Sí: con el mismo NAP obtuvimos el mismo patrón que en clase (misma tabla), pero PageRank supera a ambos en p@1 y p@5.

### Pregunta 3. ¿Qué ponderación funciona mejor y por qué? ¿Cómo influye la transformación de similitud a distancia?
| Modelo | Asociación | Frecuencia | Tiempo |
|---|---|---|---|
| PageRank (alpha 0.5) | 0.424 / 0.609 / 0.707 | 0.424 / 0.605 / 0.696 | 0.196 / 0.380 / 0.529 |
| Cercanía (reducido) | 0.221 / 0.428 / 0.518 | 0.196 / 0.355 / 0.514 | 0.076 / 0.214 / 0.286 |

- **La asociación es la mejor ponderación** en ambos modelos, con la frecuencia muy cerca (en PageRank, p@1 idéntico, 0.424, y p@5 de 0.707 contra 0.696) y el **tiempo claramente la peor**, sobre todo en p@1 (0.196 con PageRank y 0.076 con cercanía). Una explicación plausible: la fuerza de asociación y la frecuencia miden directamente cuántos participantes dieron esa respuesta, mientras que el tiempo de respuesta es una señal más ruidosa (depende de cada participante y de cuántas palabras se le ocurren).
- PageRank supera a la cercanía con las tres ponderaciones en p@1, p@3 y p@5, y con el tiempo (p@5 = 0.529) alcanza un nivel parecido al de la mejor cercanía con asociación (0.518).
- **Influencia de la transformación:** en los modelos de caminos más cortos, `100 − FA` y `1/f` determinan qué caminos se consideran «cortos»; otra transformación (por ejemplo `−log p`) cambiaría los caminos y, por tanto, los puntajes. En PageRank y node2vec se usa directamente la fuerza `sim`, sin transformar.
- Todos los resultados de PageRank se obtuvieron con alpha = 0.5.

### Otras preguntas guía
[PENDIENTE: Celia (P1) o Max (P3), una pregunta más de la lista, por ejemplo la 5: ¿en qué definiciones gana el grafo y en cuáles node2vec?]

## 9. Conclusiones y limitaciones
[PENDIENTE: Celia (P1)]

## 10. Uso de IA generativa
Se declara el uso de Claude (Anthropic) como apoyo.
Las descripciones del conjunto de prueba fueron escritas por personas; no se usó IA para simular participantes ni para fabricar normas o respuestas.


# Répartition en trois personnes
## Célia : données, évaluation, rapport
Jour 1 : créer le dépôt, le dossier partagé et la structure ci-dessus. Définir le format nuevas_desc.csv (concepto, definicion, fuente, particion) et le partager.

Jour 1 : lancer la collecte des 150 descriptions (tout le monde y participe, voir plus bas).

Jour 1-2 : écrire evaluacion.py : evaluar(modelo, datos) qui retourne p@1/3/5, MRR et couverture (% de définitions avec au moins un lemme dans le graphe, nombre moyen de lemmes utiles). Corriger le lineas[1:] pour exclure le nom du concept. Livrer cette fonction en premier, les deux autres en ont besoin.

Jour 2 : exclure ou signaler les descriptions qui contiennent le mot cible. Vérifier que chaque concept cible existe comme nœud du graphe.

Jour 2-3 : statistiques du graphe : nœuds, arêtes, degré moyen, composantes connexes, coefficient de regroupement, graphique de la distribution des degrés.

Jour 4-5 : tableau comparatif unique (tous les modèles et pondérations + lignes du Tableau 1), puis rapport et structure de la présentation.

## Léa : modèles de graphes + PageRank
Jour 1 : réécrire Construye_Grafo pour qu’il charge NAP.xls une seule fois et stocke deux attributs par arête : dist (pour les chemins : 1/frec, tiempo, 100-FA) et sim (la force, utilisée par PageRank et node2vec : frec, FA...). Justifier la transformation similarité → distance par écrit.

Jour 1-2 : nettoyer et refactoriser les 3 centralités sous la forme modelo(definicion), en graphe complet et réduit.

Jour 2-3 : PageRank personnalisé : nx.pagerank(G, personalization={lema: 1 ...}, weight="sim") sur le graphe complet. Régler alpha (probabilité de continuer à marcher, par exemple 0,5 à 0,9) sur le jeu de dév. Exclure les mots de la définition, comme dans conceptos().

Jour 3-4 : lancer toutes les évaluations sur dév puis test.

Jour 4 (si le temps le permet) : graphe orienté (nx.DiGraph, arête estimulo → respuesta), comparer avec le non orienté.

Jour 5 : répondre aux questions de discussion sur la pondération et sur « proximité contre intermédiarité ».

## Maximilien : node2vec, fusion, démo

Jour 1-2 : refaire l’entraînement node2vec avec les poids sim (c’est un point critique : avec les distances, les associations fortes sont les moins parcourues). Fixer seed et workers=1 pour la reproductibilité.

Jour 2-3 : au moins 3 configurations (par exemple dimension 64/128/300, p et q valant 0,5/1/2, longueur de marche 30/80, nombre de marches 10/20), tableau des résultats sur dév. Comparer somme, moyenne et somme pondérée IDF des vecteurs.

Jour 3-4 : RRF : score[mot] += 1/(60+rang) sur la liste du meilleur modèle de graphe (probablement proximité, graphe réduit, association) et sur celle du meilleur node2vec. Tester aussi une fusion à trois (avec PageRank).

Jour 4-5 : analyse « quand gagne node2vec ou le graphe ? » (question 5).

Jour 5-6 : démo en direct (même simple : une cellule input() qui affiche le top 5). Gradio vaut +3 points si vous avez le temps.

Tout le monde
Jour 1-2 : ≈ 50 descriptions chacun, écrites par des proches qui ne connaissent pas le mot et sans le nommer. Pas de LLM pour les générer ni pour simuler des participants.

Jour 5 : chacun analyse 5 cas d’erreur, classés par cause (lemmatisation, hors vocabulaire, polysémie, nœuds très connectés, description ambiguë).

Jour 6 : répétition de la présentation (15 min). N’importe qui peut être interrogé sur n’importe quelle partie, donc relisez le code des autres. Déclarez l’usage de l’IA.

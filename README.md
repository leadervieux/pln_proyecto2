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
Corpus nuevo (`nuevas_desc.csv`): 174 descripciones de 29 conceptos, escritas por el equipo. Todos los conceptos existen como nodos del grafo. Se omitieron 6 descripciones que contienen la propia palabra objetivo (por ejemplo, «fruta de color naranja» para *naranja*), por lo que se evaluaron **168**. Una sola ejecución por modelo, sin modificar parámetros tras ver los resultados (alpha = 0.5, fijado en el conjunto de desarrollo). Grafo de asociación.

| Modelo | Grafo | p@1 | p@3 | p@5 |
|---|---|---|---|---|
| **PageRank personalizado (alpha 0.5)** | completo | 0.405 | 0.613 | 0.690 |
| Cercanía | reducido | 0.268 | 0.488 | 0.560 |
| Grado | reducido | 0.101 | 0.250 | 0.363 |
| Intermediación por subconjunto | completo | 0.280 | 0.458 | 0.542 |
| Intermediación | reducido | 0.155 | 0.399 | 0.554 |

PageRank mantiene en prueba un desempeño muy cercano al de desarrollo (0.424 / 0.609 / 0.707): la diferencia es de 0.017 como máximo, es decir, de unas 3 descripciones sobre 168, por lo que no hay señal de que el ajuste de alpha haya sobreajustado el conjunto de desarrollo.

PageRank personalizado es el mejor modelo de grafos en los tres valores de k también en el conjunto de prueba (p@5 = 0.690 frente a 0.560 de la mejor medida de centralidad, la cercanía). Las medidas de centralidad obtienen en prueba resultados parecidos o algo mayores que en desarrollo (por ejemplo, la cercanía pasa de 0.221 / 0.428 / 0.518 a 0.268 / 0.488 / 0.560), por lo que la ventaja de PageRank se reduce en prueba, aunque se mantiene. Los conceptos del corpus nuevo son distintos de los del corpus de clase, así que esta diferencia puede deberse en parte a los conceptos elegidos; no se probó.

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
**Explicación (verificada midiendo la longitud de las listas de candidatos).** La intermediación por subconjunto asigna puntaje positivo solo a las palabras que están en caminos más cortos entre los lemas de la definición, y `conceptos()` descarta los puntajes nulos. Por eso produce listas muy cortas: cuando el concepto aparece, suele hacerlo en los primeros lugares, pero casi nunca hay alternativas si no es el primero. La cercanía puntúa a todas las palabras de la vecindad, lo que da listas largas y hace más probable que el concepto aparezca entre los primeros cinco aunque no sea el primero.

| Modelo | Conjunto | Longitud media de la lista | Listas con menos de 5 candidatos | Listas vacías |
|---|---|---|---|---|
| Cercanía (reducido) | desarrollo | 54.8 | 12.0 % | 3.3 % |
| Intermediación por subconjunto (completo) | desarrollo | 4.3 | 67.0 % | 25.7 % |
| PageRank (completo) | desarrollo | 92.6 | 3.3 % | 3.3 % |
| Cercanía (reducido) | prueba | 56.0 | 2.9 % | 0.6 % |
| Intermediación por subconjunto (completo) | prueba | 5.5 | 52.3 % | 9.2 % |
| PageRank (completo) | prueba | 95.0 | 0.6 % | 0.6 % |

La intermediación devuelve en promedio solo 4 o 5 candidatos, y más de la mitad de sus listas tienen menos de 5, mientras que la cercanía devuelve más de 50. Además, la intermediación devuelve una lista vacía en el 25.7 % de las descripciones de desarrollo (9.2 % en prueba), probablemente porque, cuando solo un lema de la definición está en el grafo, no existen caminos entre lemas distintos y todos los puntajes son nulos; esta causa no se verificó por separado. Esto limita su p@3 y p@5. A pesar de ello, cuando la lista no está vacía, su p@1 es más alto: dividiendo p@1 entre la fracción de listas no vacías, resulta aproximadamente 0.36 para la intermediación frente a 0.23 para la cercanía en desarrollo (0.264 / 0.743 y 0.221 / 0.967), y 0.31 frente a 0.27 en prueba. PageRank puntúa casi todos los nodos del grafo, por lo que sus listas son las más largas (92 a 95 candidatos).
¿Ocurre lo mismo con nuestro recurso? Sí. Con el mismo NAP obtuvimos el mismo patrón que en clase (misma tabla) y, en el corpus nuevo de prueba, también: la intermediación por subconjunto (grafo completo) gana en p@1 (0.280 frente a 0.268 de la cercanía), pero la cercanía es mejor en p@3 (0.488 frente a 0.458) y en p@5 (0.560 frente a 0.542). Además, PageRank supera a ambas en p@1, p@3 y p@5 tanto en desarrollo como en prueba.

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
Se declara el uso de Claude (Anthropic) como apoyo. Las descripciones del conjunto de prueba fueron escritas por personas; no se usó IA para simular participantes ni para fabricar normas o respuestas.

Jour 5 : chacun analyse 5 cas d’erreur, classés par cause (lemmatisation, hors vocabulaire, polysémie, nœuds très connectés, description ambiguë).

Jour 6 : répétition de la présentation (15 min). N’importe qui peut être interrogé sur n’importe quelle partie, donc relisez le code des autres. Déclarez l’usage de l’IA.

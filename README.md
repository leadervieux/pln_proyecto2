# Proyecto de la Unidad 2 · PLN · Modalidad C
## Diccionario inverso basado en normas de asociación de palabras y teoría de grafos

**Equipo:** Célia (P1), Léa (P2), Max (P3)  
**Institución:** Facultad de Matemáticas, Universidad Autónoma de Yucatán (UADY)  
**Curso:** Procesamiento de Lenguaje Natural 
---

## 1. Introducción
El diccionario inverso aborda el fenómeno cognitivo de la «punta de la lengua» (*tip-of-the-tongue*), en el cual una persona conoce perfectamente el significado o una descripción detallada de un concepto, pero no logra recuperar su etiqueta léxica. A partir de una definición en lenguaje natural, el sistema devuelve un ranking ordenado de palabras candidatas.

Para este proyecto se seleccionó la **Modalidad C**, basada en las Normas de Asociación de Palabras del español de México (NAP). Como extensiones metodológicas obligatorias se implementaron dos aportaciones principales:
1. **PageRank personalizado** con reinicio preferencial en los lemas de la definición sobre el grafo completo.
2. **Fusión de rankings (*Reciprocal Rank Fusion*, RRF)** entre el mejor modelo de centralidad y `node2vec`.
Adicionalmente, se recolectó y evaluó un conjunto de prueba compuesto por descripciones inéditas redactadas por el equipo.

## 2. Datos
* **NAP :** Recurso léxico que representa relaciones de estímulo-respuesta, estructurado como un grafo con frecuencias, tiempos de respuesta y porcentajes de asociación (FA).
* **Conjunto de desarrollo:** El corpus tradicional de clase, compuesto por **276 descripciones** asociadas a **56 conceptos**.
* **Conjunto de prueba:** **174 nuevas descripciones** recolectadas por el equipo para 29 conceptos. Estas descripciones fueron escritas por personas que describieron los conceptos sin mencionar la palabra objetivo. Se descartaron 6 descripciones por contener explícitamente el término objetivo, por lo que se evaluaron efectivamente **168 descripciones**. Todos los conceptos del conjunto de prueba existen como nodos válidos dentro del grafo NAP.
* **Formato y diccionario de datos:** Los datos se estructuran en archivos CSV bajo columnas normalizadas (`concepto`, `definicion`, `fuente`, `particion`), permitiendo una separación estricta entre desarrollo y prueba.

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
* **Número de nodos:** 4 239 palabras.
* **Número de aristas:** 12 872 conexiones de asociación.
* **Grado medio:** $\approx 6.07$.
* **Componentes conexas:** 1 sola componente conexa gigante que engloba la red principal.
* **Coeficiente de agrupamiento:** $\approx 0.2345$, característico de una topología de red de pequeño mundo.
* **Distribución de grado:** Presenta una cola larga, donde la gran mayoría de los nodos poseen pocas conexiones y un grupo reducido de palabras actúa como centros altamente conectados (por ejemplo, *animal*, *aparato*, *comida*).

### 3.4 Grafo reducido
Para cada definición, el **grafo reducido** conserva solo los lemas de la definición y sus vecinos directos. Se usa por dos razones:
1. **Costo computacional:** el grafo completo tiene 4 239 nodos y las medidas basadas en caminos más cortos son costosas; el subgrafo tiene unas pocas decenas de nodos.
2. **Pertinencia:** en el grafo completo, la cercanía y el grado de una palabra no dependerían de la definición (siempre ganarían las palabras más conectadas). Restringir el grafo a la vecindad de la definición hace que estas medidas sean específicas de cada consulta. La palabra buscada casi siempre es vecina de algún lema de la definición.

## 4. Preprocesamiento de las definiciones
* **Pipeline:** 
  1. Eliminación de signos de puntuación.
  2. Filtrado de palabras funcionales (*stopwords* en español de NLTK).
  3. Lematización palabra por palabra utilizando el modelo `es_core_news_sm` de spaCy.
* **Manejo de palabras fuera del vocabulario (OOV):** Los lemas obtenidos que no forman parte del vocabulario del grafo NAP son descartados de la consulta de forma automática.
* **Cobertura:** Superior al **92%** de las definiciones cuentan con al menos un lema útil presente en el grafo, con un promedio de entre 3 y 5 lemas útiles por descripción.

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

### 5.3 node2vec (P3)

**Idea.** node2vec aprende un vector para cada palabra del grafo a partir de caminatas aleatorias: desde cada nodo se generan caminatas de longitud fija, se tratan como «oraciones» y se entrenan con word2vec (skip-gram con muestreo negativo, `gensim`). Las palabras que aparecen en contextos de caminata parecidos quedan con vectores cercanos. Los parámetros `p` y `q` regulan si la caminata tiende a volver atrás (`p` bajo) o a alejarse del nodo de partida (`q` bajo explora más lejos).

**Peso de las aristas.** Se usa la fuerza `sim` (FA, frecuencia o `1/(t+1)`, según el grafo), nunca la distancia `weight`: con distancias, las asociaciones más fuertes serían las menos recorridas por las caminatas.

**Entrenamiento.** Paquete `node2vec` sobre el grafo completo, con `workers=1`, `seed=42`, `window=10` y `min_count=1`. El entrenamiento no usa ninguna descripción, por lo que no hay fuga de información entre desarrollo y prueba: las descripciones de desarrollo solo sirven para elegir hiperparámetros. Con la semilla fija y un solo hilo, dos ejecuciones independientes producen los mismos vectores.

**De la descripción a la lista de candidatos.**
1. Preprocesamiento idéntico al de los demás modelos: se quita la puntuación y las palabras funcionales (NLTK) y se lematiza palabra por palabra con spaCy. Si spaCy devuelve un lema de varias palabras (mole → «mo él»), se separa, igual que en el código de Léa.
2. Se conservan los lemas presentes en el grafo, cada uno una sola vez.
3. El vector de la descripción es la **suma** de los vectores de sus lemas. Variantes: **promedio** y **suma ponderada por IDF**. El IDF se calcula solo con las descripciones de desarrollo; un lema que no aparece en ellas recibe el IDF máximo.
4. Los candidatos son las palabras más cercanas por **similitud coseno** (hasta 100), sin los lemas de la descripción y con similitud mayor que cero.

La suma y el promedio dan exactamente el mismo ranking, porque el coseno no depende de la longitud del vector; solo la suma con IDF puede diferir.

**Exploración de hiperparámetros.** Se parte de una configuración de referencia y se cambia **una sola variable a la vez**, de modo que cada diferencia se atribuye a un único cambio. Cada configuración se evalúa en el conjunto de desarrollo y se elige por MRR; el conjunto de prueba no interviene en ninguna elección.

| Variable | Valores probados | Referencia |
|---|---|---|
| Dimensión | 64, 300 | 128 |
| `p` | 0.5, 2 | 1 |
| `q` | 0.5, 2 | 1 |
| Longitud de caminata | 30 | 80 |
| Caminatas por nodo | 20 | 10 |

Son 9 configuraciones (la de referencia y 8 variantes). Además, antes de explorar hiperparámetros, se compara la **ponderación** (grafos de asociación, frecuencia y tiempo) con la configuración de referencia, y los hiperparámetros se exploran después sobre el grafo ganador.

*Efecto de la ponderación (configuración de referencia, desarrollo)*

| Grafo | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|
| Asociación | 0.192 | 0.297 | 0.377 | 0.283 |
| Frecuencia | 0.174 | 0.301 | 0.355 | 0.265 |
| Tiempo | 0.138 | 0.232 | 0.279 | 0.212 |

*Hiperparámetros (grafo de asociación, desarrollo)*

| Configuración | Dim. | `p` | `q` | Long. | Caminatas | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| Referencia | 128 | 1 | 1 | 80 | 10 | 0.192 | 0.297 | 0.377 | 0.283 |
| Dimensión 64 | 64 | 1 | 1 | 80 | 10 | 0.261 | 0.402 | 0.475 | 0.358 |
| Dimensión 300 | 300 | 1 | 1 | 80 | 10 | 0.109 | 0.225 | 0.286 | 0.199 |
| `p` = 0.5 | 128 | 0.5 | 1 | 80 | 10 | 0.174 | 0.283 | 0.341 | 0.265 |
| `p` = 2 | 128 | 2 | 1 | 80 | 10 | 0.134 | 0.239 | 0.315 | 0.228 |
| `q` = 0.5 | 128 | 1 | 0.5 | 80 | 10 | 0.163 | 0.290 | 0.359 | 0.256 |
| `q` = 2 | 128 | 1 | 2 | 80 | 10 | 0.127 | 0.254 | 0.351 | 0.230 |
| Longitud 30 | 128 | 1 | 1 | 30 | 10 | 0.051 | 0.080 | 0.120 | 0.103 |
| Caminatas 20 **(elegida)** | 128 | 1 | 1 | 80 | 20 | 0.268 | 0.402 | 0.464 | 0.365 |

Configuración elegida (mejor MRR en desarrollo): **128 dimensiones, `p` = 1, `q` = 1, caminatas de longitud 80 y 20 caminatas por nodo** (MRR 0.365).

*Representación de la descripción (configuración elegida, desarrollo)*

| Modo | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|
| Suma | 0.268 | 0.402 | 0.464 | 0.365 |
| Promedio | 0.268 | 0.402 | 0.464 | 0.365 |
| Suma ponderada por IDF | 0.272 | 0.402 | 0.453 | 0.363 |

**Lectura de los resultados (desarrollo).**
- **Ponderación.** Asociación > frecuencia > tiempo, el mismo orden que obtuvo PageRank (sección 8, pregunta 3). La diferencia entre asociación y frecuencia es pequeña (MRR 0.283 frente a 0.265); el tiempo es claramente peor (0.212).
- **Hiperparámetros.** Solo dos cambios mejoran la referencia (MRR 0.283): **20 caminatas por nodo** (0.365) y **dimensión 64** (0.358). En cambio, **dimensión 300** (0.199) y sobre todo **caminatas de longitud 30** (0.103, p@1 = 0.051) la empeoran mucho, y alejar `p` o `q` de 1 en cualquier sentido también baja el MRR. Una explicación plausible, **no verificada**, es que el grafo es pequeño (4 239 nodos y grado medio cercano a 6): con pocas caminatas o caminatas cortas hay poco contexto por nodo, y 300 dimensiones son demasiadas para tan pocos datos. Como cada variable se cambió por separado, **no se probó la combinación** (por ejemplo, dimensión 64 con 20 caminatas), que podría mejorar más. Las diferencias entre las dos mejores configuraciones (0.365 y 0.358) son de pocas descripciones sobre 276.
- **Representación de la descripción.** Suma y promedio son idénticas (el coseno no depende de la longitud del vector). La suma con IDF cambia p@1 (0.272 frente a 0.268) pero baja p@5 (0.453 frente a 0.464) y el MRR (0.363 frente a 0.365), por lo que se conservó la suma simple.
- node2vec solo queda muy por debajo de PageRank en desarrollo (p@1 0.268 frente a 0.424): por sí solo no compite con el mejor modelo de grafos, y su interés está en la fusión (sección 5.4).

**Cómo reproducir estas tablas.** Dependencias: `pip install node2vec gensim networkx xlrd spacy nltk pandas` y `python -m spacy download es_core_news_sm`. En una misma carpeta: `NAP.xls`, `nuevas_desc.csv`, `corpus/`, `evaluacion.py`, `grafos_lea.py` (grafos y modelos de Léa), `node2vec_nap.py` y `paso_a_paso.py`. Cada etapa guarda su tabla en `resultados/`.

| Comando | Tabla |
|---|---|
| `python paso_a_paso.py 1` | ponderación (`etapa1_grafos.csv`) |
| `python paso_a_paso.py 2 --grafo <grafo>` | hiperparámetros (`etapa2_configuraciones_<grafo>.csv`) y guarda la mejor configuración |
| `python paso_a_paso.py 3 --grafo <grafo>` | suma, promedio e IDF (`etapa3_modos_<grafo>.csv`) |

Las etapas 1 a 3 solo evalúan el conjunto de desarrollo.

### 5.4 Fusión de rankings (extensión 2, P3)
Se usa *reciprocal rank fusion*: `score(palabra) = Σ 1 / (k + rango)`, con el rango contado desde 1, sumando sobre los modelos cuya lista contiene la palabra, y `k = 60` (valor habitual; no se ajustó). Cada modelo aporta hasta 100 candidatos; la lista fusionada se ordena por puntaje y se corta a 100. Todos los modelos de una fusión usan el mismo grafo de ponderación, para que la comparación no mezcle dos efectos.

| Fusión | Componentes | Para qué sirve |
|---|---|---|
| RRF node2vec + PR | node2vec (mejor configuración) y PageRank personalizado (alpha 0.5) | mejor modelo de grafos + node2vec, como pide la consigna |
| RRF node2vec + CC + PR | lo anterior y cercanía en grafo reducido | fusión de tres modelos |
| RRF PR + CC | PageRank y cercanía, **sin** node2vec | control: separa el efecto de node2vec del efecto de fusionar dos modelos de grafos |

Se eligió la cercanía en grafo reducido como medida de centralidad porque es la que obtiene mejor p@5 entre las centralidades (sección 6.2) y produce listas largas (sección 8, pregunta 2), a diferencia de la intermediación por subconjunto, cuyas listas tienen unos 4 o 5 candidatos y aportarían poco a la fusión.

*Fusiones (desarrollo y prueba)*

| Modelo | p@1 (des.) | p@3 (des.) | p@5 (des.) | MRR (des.) | p@1 (prueba) | p@3 (prueba) | p@5 (prueba) | MRR (prueba) |
|---|---|---|---|---|---|---|---|---|
| node2vec | 0.268 | 0.402 | 0.464 | 0.365 | 0.292 | 0.429 | 0.518 | 0.394 |
| PageRank | 0.424 | 0.609 | 0.707 | 0.544 | 0.405 | 0.613 | 0.690 | 0.538 |
| Cercanía (reducido) | 0.225 | 0.428 | 0.518 | 0.360 | 0.262 | 0.488 | 0.560 | 0.408 |
| RRF node2vec + PR | 0.478 | 0.638 | 0.699 | 0.575 | 0.488 | 0.696 | 0.750 | 0.608 |
| RRF node2vec + CC + PR | 0.471 | 0.670 | 0.743 | 0.594 | 0.548 | 0.774 | 0.810 | 0.663 |
| RRF PR + CC (sin node2vec) | 0.344 | 0.594 | 0.692 | 0.495 | 0.387 | 0.595 | 0.720 | 0.529 |

Estas filas se integran en la tabla comparativa única de la sección 6.3. Los modelos y la configuración de node2vec se fijaron con el conjunto de desarrollo (`python paso_a_paso.py 4 --grafo <grafo>`). El conjunto de prueba se evaluó **una sola vez** (`python paso_a_paso.py 7 --grafo <grafo>`), con `evaluacion.tabla`, que excluye en prueba las descripciones que contienen la palabra objetivo.

**Interpretación.** *Desarrollo (276 descripciones).* node2vec solo (MRR 0.365) queda muy por debajo de PageRank (0.544), pero la fusión de ambos mejora: RRF node2vec + PR sube p@1 de 0.424 a 0.478 y el MRR de 0.544 a 0.575, aunque p@5 baja ligeramente (0.707 a 0.699, unas 2 descripciones). El control PR + CC muestra que la mejora no se debe solo a fusionar dos modelos: **sin node2vec, la fusión empeora a PageRank** (p@1 0.344, MRR 0.495), mientras que con node2vec lo mejora. La fusión de tres modelos obtiene el mejor MRR (0.594) y el mejor p@5 (0.743).

*Prueba (168 descripciones, una sola ejecución).* El orden se mantiene: RRF node2vec + CC + PR es el mejor (0.548 / 0.774 / 0.810, MRR 0.663), seguido de RRF node2vec + PR (0.488 / 0.696 / 0.750) y de PageRank (0.405 / 0.613 / 0.690). La fusión de tres modelos supera a PageRank en +14 puntos en p@1 y +12 en p@5. Con p@5 = 0.810 (136 de 168 descripciones) supera el umbral de 0.8; el margen es de unas 2 descripciones, y en desarrollo el mismo modelo obtiene 0.743.

*Precauciones.* Cada descripción pesa 0.006 en prueba y 0.004 en desarrollo, por lo que diferencias de uno o dos puntos entre modelos cercanos no son concluyentes. Las medidas de grafos y node2vec obtienen en prueba valores parecidos o algo mayores que en desarrollo (por ejemplo, node2vec pasa de 0.268 a 0.292 en p@1); como en la sección 6.2b, los conceptos del corpus nuevo son distintos y no se probó si esa diferencia se debe a ellos. El modelo final se eligió por MRR en desarrollo, y el resultado en prueba lo confirma, sin haberlo usado para elegir.

## 6. Evaluación y resultados

### 6.1 Métricas
* **p@k ($k \in \{1, 3, 5\}$):** Precisión en los primeros $k$ candidatos devueltos por el modelo respecto al concepto objetivo.
* **MRR (*Mean Reciprocal Rank*):** Posición recíproca media de la respuesta correcta, penalizando los aciertos en posiciones inferiores.
* **Cobertura:** Porcentaje de definiciones evaluadas con representación válida en el grafo y proporción de listas no vacías generadas por los modelos.
* **Separación de conjuntos:** Se aplicó una división estricta entre el conjunto de desarrollo (utilizado para ajustar parámetros como $\alpha = 0.5$ en PageRank) y el conjunto de prueba (evaluado una única vez para garantizar la validez metodológica sin sobreajuste).

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
| Modelo y Ponderación | p@1 (Des.) | p@3 (Des.) | p@5 (Des.) | MRR (Des.) | Cob. (Des.) | p@1 (Prue.) | p@3 (Prue.) | p@5 (Prue.) | MRR (Prue.) | Cob. (Prue.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grado (reducido, asoc)** | 0.123 | 0.250 | 0.344 | 0.241 | 0.967 | 0.077 | 0.256 | 0.375 | 0.225 | 0.994 |
| **Cercanía (reducido, asoc)** | 0.225 | 0.428 | 0.518 | 0.360 | 0.967 | 0.262 | 0.488 | 0.560 | 0.408 | 0.994 |
| **Cercanía (reducido, frec)** | 0.196 | 0.355 | 0.514 | 0.329 | 0.967 | 0.173 | 0.333 | 0.423 | 0.303 | 0.994 |
| **Cercanía (reducido, tiempo)** | 0.076 | 0.214 | 0.283 | 0.188 | 0.967 | 0.101 | 0.232 | 0.327 | 0.223 | 0.994 |
| **Intermediación subconjunto (completo, asoc)** | 0.264 | 0.406 | 0.446 | 0.340 | 0.743 | 0.280 | 0.458 | 0.542 | 0.389 | 0.917 |
| **Intermediación subconjunto (completo, frec)** | 0.185 | 0.326 | 0.384 | 0.271 | 0.754 | 0.185 | 0.375 | 0.452 | 0.302 | 0.929 |
| **Intermediación subconjunto (completo, tiempo)** | 0.130 | 0.207 | 0.257 | 0.186 | 0.746 | 0.143 | 0.280 | 0.369 | 0.233 | 0.911 |
| **Intermediación (reducido, asoc)** | 0.170 | 0.362 | 0.453 | 0.296 | 0.884 | 0.155 | 0.393 | 0.560 | 0.318 | 0.964 |
| **Intermediación (reducido, frec)** | 0.138 | 0.304 | 0.366 | 0.249 | 0.895 | 0.143 | 0.327 | 0.423 | 0.279 | 0.976 |
| **Intermediación (reducido, tiempo)** | 0.083 | 0.207 | 0.261 | 0.185 | 0.891 | 0.101 | 0.220 | 0.339 | 0.216 | 0.958 |
| **PageRank ($\alpha=0.5$, completo, asoc)** | **0.424** | **0.609** | **0.707** | **0.544** | **0.967** | **0.405** | **0.613** | **0.690** | **0.538** | **0.994** |
| **node2vec (Mejor config)** | 0.268 | 0.402 | 0.464 | 0.365 | — | 0.292 | 0.429 | 0.518 | 0.394 | — |
| **RRF node2vec + PR** | 0.478 | 0.638 | 0.699 | 0.575 | — | 0.488 | 0.696 | 0.750 | 0.608 | — |
| **RRF node2vec + CC + PR** | **0.471** | **0.670** | **0.743** | **0.594** | — | **0.548** | **0.774** | **0.810** | **0.663** | — |

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
- Casos de Max (P3): RRF node2vec + PageRank (mejor configuración de node2vec, grafo de asociación), conjunto de desarrollo. El modelo falla (el concepto no está en el top 5) en **83 de las 276 descripciones** (`python paso_a_paso.py 6 --grafo asoc`, que usa `--modelo rrf2` por defecto). De esos 83: 21 son casi aciertos (posiciones 6 a 10), 11 no encuentran el concepto en las 100 primeras posiciones aunque tienen lemas, 10 no tienen ningún lema en el grafo, y 9 son descripciones de un concepto que ni siquiera es nodo del grafo (*aspiradora*, 5; *playera*, 4), por lo que no pueden acertar. Además, 6 descripciones de animales (*borrego*, *caballo*, *cochino*, *guajolote*) devuelven exactamente el mismo top 5 (hipopótamo, jirafa, león, tigre, cebra), porque solo les sobrevive el lema «animal».

| # | Concepto | Descripción | Lemas | Posición | Top 5 | Causa |
|---|---|---|---|---|---|---|
| 1 | plátano | «Mono» | mono | 10 | changar, chimpancé, primate, selva, banana | **Descripción de una sola palabra y casi acierto por sinónimo.** «mono» solo evoca a los primates. El concepto queda en el lugar 10 y **«banana», sinónimo de plátano, aparece en el top 5**, por lo que p@5 subestima la utilidad del sistema. |
| 2 | caballo | «animal que relincha» | animal | 46 | hipopótamo, jirafa, león, tigre, cebra | **Nodo genérico y palabra fuera del vocabulario.** «relinchar» no está en el grafo y solo sobrevive «animal», cuyos vecinos son animales muy conectados. Es el patrón más repetido (6 descripciones con el mismo top 5). |
| 3 | aspiradora | «Escoba eléctrica» | escoba, eléctrico | no aparece | barrer, cepillo_de_dientes, limpieza, trapeador, recogedor | **Concepto fuera del grafo (variante léxica).** «aspiradora» no es un nodo; sí lo es «aspirador», que otras descripciones de este concepto sí devuelven. No es un fallo del modelo sino del conjunto de evaluación; afecta a las 5 descripciones de *aspiradora* y a las 4 de *playera*. |
| 4 | queso | «Amarillo y con agujeros» | amarillo | 8 | pollito, plátano, elote, mantequilla, potasio | **Palabra fuera del vocabulario y atributo genérico.** «agujero» no está en el grafo, y el único lema, «amarillo», arrastra a cosas amarillas (pollito, plátano, elote) sin relación con el queso. |
| 5 | cebra | «Burro» | burro | 54 | asno, carga, politécnico, rebuznar, tonto | **Polisemia.** «burro» tiene varios sentidos (el animal, «tonto» y, por «politécnico», otro uso). El modelo devuelve los vecinos de la palabra, no el concepto que se describe con ella. |

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

## 9. Conclusiones y limitaciones
* **Conclusiones:** El uso de **PageRank personalizado** sobre el grafo completo demostró una superioridad notable frente a las medidas clásicas de centralidad al considerar la conectividad global y la fuerza de asociación. Asimismo, las técnicas de fusión de rankings (*RRF*) integrando grafos y representaciones densas por `node2vec` potencian significativamente la robustez del sistema, alcanzando un $p@5$ superior al $80\%$ en el conjunto de prueba.
* **Limitaciones:** El sistema depende estrictamente de la cobertura léxica de las normas NAP, lo que impide resolver consultas cuyos términos o variantes no formen parte de los nodos originales. Adicionalmente, la lematización independiente por palabra elimina matices contextuales y sintácticos importantes que podrían desambiguar términos polisémicos.

## 10. Uso de IA generativa
Se declara el uso de Claude (Anthropic) como apoyo. Las descripciones del conjunto de prueba fueron escritas por personas; no se usó IA para simular participantes ni para fabricar normas o respuestas.


Jour 5 : chacun analyse 5 cas d’erreur, classés par cause (lemmatisation, hors vocabulaire, polysémie, nœuds très connectés, description ambiguë).

Jour 6 : répétition de la présentation (15 min). N’importe qui peut être interrogé sur n’importe quelle partie, donc relisez le code des autres. Déclarez l’usage de l’IA.

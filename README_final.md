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
* **Cobertura:** En desarrollo, el **95.7 %** de las descripciones (264 de 276) tienen al menos un lema útil presente en el grafo, con un promedio de 3.04 lemas útiles por descripción; las 12 restantes (4.3 %) no tienen ninguno y ningún modelo puede acertarlas. Los modelos devuelven una lista no vacía para el 96.4 % de las descripciones de desarrollo y el 99.4 % de las de prueba.

## 5. Modelos

### 5.1 Modelos basados en centralidad (P2)
En todos los casos, los lemas de la definición se excluyen de la lista de candidatos y solo se conservan los candidatos con puntaje mayor que cero (función `conceptos()` de la clase).

| Modelo | Grafo | Qué mide | Peso usado |
|---|---|---|---|
| **Cercanía** (`diccionario_cc`) | reducido | distancia promedio de cada palabra al resto de la vecindad de la definición | `weight` |
| **Intermediación por subconjunto** (`diccionario_btc`) | completo | cuántos caminos más cortos entre los lemas de la definición pasan por cada palabra | `weight` |
| **Intermediación** (`diccionario_btc_reducido`) | reducido | lo mismo, pero calculado solo sobre la vecindad de la definición | `weight` |
| **Grado** (`diccionario_dc`) | reducido | número de vecinos de cada palabra en la vecindad de la definición | sin peso |

Se descartaron Katz, vector propio y comunicabilidad: en los resultados de referencia de la clase (Tabla 1) son las medidas con menor desempeño y las de mayor costo.

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

**Preprocesamiento** En esta parte se ha corregido la lematización: los lemas se han puesto en minúsculas antes de utilizar SpaCy; el objetivo era también comprobar el efecto que tiene un pequeño cambio como este en los resultados. En el archivo «resultados_2» encontrarás, en la etapa 0, los resultados de las pruebas realizadas con los modelos anteriores utilizando esta nueva lematización, a modo de comparación.

**Peso de las aristas.** Se usa la fuerza `sim` (FA, frecuencia o `1/(t+1)`, según el grafo), nunca la distancia `weight`: con distancias, las asociaciones más fuertes serían las menos recorridas por las caminatas.

**Entrenamiento.** Paquete `node2vec` sobre el grafo completo, con `workers=1`, `seed=42`, `window=10` y `min_count=1`. El entrenamiento no usa ninguna descripción, por lo que no hay fuga de información entre desarrollo y prueba: las descripciones de desarrollo solo sirven para elegir hiperparámetros. Con la semilla fija y un solo hilo, dos ejecuciones independientes con el mismo código y las mismas versiones de las librerías producen los mismos vectores.

**De la descripción a la lista de candidatos.**
1. Preprocesamiento de las descripciones en esta parte: se pasa el texto a minúsculas antes de usar spaCy, se quita la puntuación y las palabras funcionales (NLTK) y se lematiza palabra por palabra con spaCy. Si spaCy devuelve un lema de varias palabras (mole → «mo él»), se separa, igual que en el código de Léa.
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

Son 10 configuraciones: la de referencia, 8 variantes de una sola variable y la combinación de las dos que mejoran (dimensión 64 con 20 caminatas por nodo), probada después de ver las variantes individuales y también sobre el desarrollo. Además, antes de explorar hiperparámetros, se compara la **ponderación** (grafos de asociación, frecuencia y tiempo) con la configuración de referencia, y los hiperparámetros se exploran después sobre el grafo ganador.

*Efecto de la ponderación (configuración de referencia, desarrollo)*

| Grafo | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|
| Asociación | 0.207 | 0.315 | 0.377 | 0.295 |
| Frecuencia | 0.178 | 0.308 | 0.370 | 0.271 |
| Tiempo | 0.145 | 0.228 | 0.275 | 0.216 |

*Hiperparámetros (grafo de asociación, desarrollo)*

| Configuración | Dim. | `p` | `q` | Long. | Caminatas | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| Referencia | 128 | 1 | 1 | 80 | 10 | 0.207 | 0.315 | 0.377 | 0.295 |
| Dimensión 64 | 64 | 1 | 1 | 80 | 10 | 0.275 | 0.420 | 0.478 | 0.371 |
| Dimensión 300 | 300 | 1 | 1 | 80 | 10 | 0.116 | 0.232 | 0.290 | 0.206 |
| `p` = 0.5 | 128 | 0.5 | 1 | 80 | 10 | 0.185 | 0.286 | 0.351 | 0.275 |
| `p` = 2 | 128 | 2 | 1 | 80 | 10 | 0.145 | 0.254 | 0.322 | 0.239 |
| `q` = 0.5 | 128 | 1 | 0.5 | 80 | 10 | 0.167 | 0.293 | 0.366 | 0.259 |
| `q` = 2 | 128 | 1 | 2 | 80 | 10 | 0.138 | 0.254 | 0.362 | 0.238 |
| Longitud 30 | 128 | 1 | 1 | 30 | 10 | 0.054 | 0.091 | 0.127 | 0.110 |
| Caminatas 20 | 128 | 1 | 1 | 80 | 20 | 0.283 | 0.417 | 0.471 | 0.378 |
| Caminatas 20 y dimensión 64 **(elegida)** | 64 | 1 | 1 | 80 | 20 | 0.308 | 0.475 | 0.525 | 0.409 |

Configuración elegida (mejor MRR en desarrollo): **64 dimensiones, `p` = 1, `q` = 1, caminatas de longitud 80 y 20 caminatas por nodo** (MRR 0.409).

*Representación de la descripción (configuración elegida, desarrollo)*

| Modo | p@1 | p@3 | p@5 | MRR |
|---|---|---|---|---|
| Suma | 0.308 | 0.475 | 0.525 | 0.409 |
| Promedio | 0.308 | 0.475 | 0.525 | 0.409 |
| Suma ponderada por IDF | 0.304 | 0.457 | 0.511 | 0.400 |

**Lectura de los resultados (desarrollo).**
- **Ponderación.** Asociación > frecuencia > tiempo, el mismo orden que obtuvo PageRank (sección 8, pregunta 3). La diferencia entre asociación y frecuencia es pequeña (MRR 0.295 frente a 0.271); el tiempo es claramente peor (0.216).
- **Hiperparámetros.** Solo dos cambios individuales mejoran la referencia (MRR 0.295): **20 caminatas por nodo** (0.378) y **dimensión 64** (0.371). En cambio, **dimensión 300** (0.206) y sobre todo **caminatas de longitud 30** (0.110, p@1 = 0.054) la empeoran mucho, y alejar `p` o `q` de 1 en cualquier sentido también baja el MRR (0.238 a 0.275). Una explicación plausible, **no verificada**, es que el grafo es pequeño (4 239 nodos y grado medio cercano a 6): con pocas caminatas o caminatas cortas hay poco contexto por nodo, y 300 dimensiones son demasiadas para tan pocos datos. La combinación de dimensión 64 y 20 caminatas es la mejor (MRR 0.409) y es la configuración que usan todas las etapas posteriores.
- **Representación de la descripción.** Suma y promedio son idénticas (el coseno no depende de la longitud del vector). La suma con IDF queda por debajo en los cuatro indicadores (p@1 0.304 frente a 0.308, p@5 0.511 frente a 0.525, MRR 0.400 frente a 0.409), por lo que se conservó la suma simple.
- node2vec solo queda muy por debajo de PageRank en desarrollo (p@1 0.308 frente a 0.428; MRR 0.409 frente a 0.549): por sí solo no compite con el mejor modelo de grafos, y su interés está en la fusión (sección 5.4).

**Cómo reproducir estas tablas.** Dependencias: `pip install -r requirements.txt` y `python -m spacy download es_core_news_sm` (el modelo de spaCy se descarga aparte). En una misma carpeta: `NAP.xls`, `nuevas_desc.csv`, `corpus/`, `evaluacion.py`, `grafos_lea.py` (grafos y modelos de Léa), `node2vec_nap_2.py`, `paso_a_paso_2.py` y `requirements.txt`. Cada etapa guarda su tabla en `resultados_2/`; las etapas 2 y 3 guardan además la mejor configuración en `resultados_2/mejor_<grafo>.json`, que las etapas 4 a 7 necesitan, por lo que hay que ejecutarlas en orden.

| Comando | Qué hace | Salida en `resultados_2/` |
|---|---|---|
| `python paso_a_paso_2.py 0` | modelos de grafos de Léa (grado, cercanía, intermediación y PageRank) en los tres grafos, sobre desarrollo | `etapa0_rrf_dev_asoc.csv` (el nombre usa el valor de `--grafo`, pero contiene los tres grafos) |
| `python paso_a_paso_2.py 1` | ponderación | `etapa1_grafos.csv` |
| `python paso_a_paso_2.py 2 --grafo asoc` | hiperparámetros; guarda la mejor configuración | `etapa2_configuraciones_asoc.csv` |
| `python paso_a_paso_2.py 3 --grafo asoc` | suma, promedio e IDF | `etapa3_modos_asoc.csv` |
| `python paso_a_paso_2.py 4 --grafo asoc` | modelos y fusiones RRF, sobre desarrollo | `etapa4_rrf_dev_asoc.csv` |
| `python paso_a_paso_2.py 5 --grafo asoc` | demostración en vivo; opciones `--modelo` (`rrf2` por defecto, `rrf3`, `pr`, `node2vec`, `cc`, `rrfg`) y `--gradio` (interfaz web) | ninguna |
| `python paso_a_paso_2.py 6 --grafo asoc` | descripciones de desarrollo donde falla RRF (`--modelo`, `rrf2` por defecto), node2vec y PageRank | `fallos_rrf2_asoc.csv`, `fallos_Node2vec_asoc.csv`, `fallos_PR_asoc.csv` |
| `python paso_a_paso_2.py 7 --grafo asoc` | tabla final en desarrollo **y prueba**; pide escribir `si` | `etapa7_tabla_final_asoc.csv` |

Las etapas 0 a 6 solo evalúan el conjunto de desarrollo; la 7 es la única que evalúa la prueba. Las etapas 3 a 7 vuelven a entrenar node2vec con la configuración elegida (64 dimensiones, longitud 80, 20 caminatas, `p` = 1, `q` = 1), que está escrita directamente en `paso_a_paso_2.py`: si se cambia la configuración, hay que cambiarla también ahí.

### 5.4 Fusión de rankings (extensión 2, P3)
Se usa *reciprocal rank fusion*: `score(palabra) = Σ 1 / (k + rango)`, con el rango contado desde 1, sumando sobre los modelos cuya lista contiene la palabra, y `k = 60` (valor habitual; no se ajustó). Cada modelo aporta hasta 100 candidatos; la lista fusionada se ordena por puntaje y se corta a 100. Todos los modelos de una fusión usan el mismo grafo de ponderación, para que la comparación no mezcle dos efectos.

| Fusión | Componentes | Para qué sirve |
|---|---|---|
| RRF node2vec + PR | node2vec (mejor configuración) y PageRank personalizado (alpha 0.5) | mejor modelo de grafos + node2vec, como pide la consigna |
| RRF node2vec + CC + PR | lo anterior y cercanía en grafo reducido | fusión de tres modelos |
| RRF PR + CC | PageRank y cercanía, **sin** node2vec | control: separa el efecto de node2vec del efecto de fusionar dos modelos de grafos |
| RRF node2vec + BTC + PR | node2vec, intermediación por subconjunto (grafo completo) y PageRank | control adicional con otra medida de centralidad |

Se eligió la cercanía en grafo reducido como medida de centralidad principal porque es la que obtiene mejor p@5 entre las centralidades (sección 6.2) y produce listas largas (sección 8, pregunta 2), a diferencia de la intermediación por subconjunto, cuyas listas tienen unos 4 o 5 candidatos y se esperaba que aportaran poco a la fusión; la cuarta fila comprueba esa expectativa. En `etapa4_rrf_dev_asoc.csv` la fila `rff4` es RRF node2vec + BTC + PR y la fila `node cc pr` es la misma fusión que RRF node2vec + CC + PR (resultados idénticos).

*Fusiones (desarrollo y prueba)*

| Modelo | p@1 (des.) | p@3 (des.) | p@5 (des.) | MRR (des.) | p@1 (prueba) | p@3 (prueba) | p@5 (prueba) | MRR (prueba) |
|---|---|---|---|---|---|---|---|---|
| node2vec | 0.308 | 0.475 | 0.525 | 0.409 | 0.268 | 0.518 | 0.589 | 0.406 |
| PageRank | 0.428 | 0.620 | 0.703 | 0.549 | 0.417 | 0.637 | 0.708 | 0.548 |
| Cercanía (reducido) | 0.221 | 0.428 | 0.525 | 0.360 | 0.262 | 0.476 | 0.560 | 0.405 |
| RRF node2vec + PR | 0.475 | 0.652 | 0.696 | 0.576 | 0.494 | 0.720 | 0.738 | 0.615 |
| RRF node2vec + CC + PR | 0.438 | 0.667 | 0.739 | 0.573 | 0.548 | 0.768 | 0.804 | 0.664 |
| RRF PR + CC (sin node2vec) | 0.341 | 0.598 | 0.688 | 0.493 | 0.375 | 0.583 | 0.714 | 0.520 |
| RRF node2vec + BTC + PR | 0.464 | 0.627 | 0.681 | 0.567 | 0.518 | 0.690 | 0.762 | 0.625 |

Estas filas se integran en la tabla comparativa única de la sección 6.3. Los modelos y la configuración de node2vec se fijaron con el conjunto de desarrollo (`python paso_a_paso_2.py 4 --grafo asoc`). El conjunto de prueba se evalúa únicamente con la etapa 7 (`python paso_a_paso_2.py 7 --grafo asoc`), con `evaluacion.tabla`, que excluye en prueba las descripciones que contienen la palabra objetivo.


**Interpretación.** *Desarrollo (276 descripciones).* node2vec solo (MRR 0.409) queda muy por debajo de PageRank (0.549), pero la fusión de ambos mejora: RRF node2vec + PR sube p@1 de 0.428 a 0.475 (13 descripciones más) y el MRR de 0.549 a 0.576, aunque p@5 baja ligeramente (0.703 a 0.696, unas 2 descripciones). El control PR + CC muestra que la mejora no se debe solo a fusionar dos modelos: **sin node2vec, la fusión empeora a PageRank** (p@1 0.341, MRR 0.493), mientras que con node2vec lo mejora. Las dos fusiones con node2vec quedan prácticamente empatadas en MRR (0.576 la de dos modelos, 0.573 la de tres): la de tres modelos tiene el mejor p@3 (0.667) y el mejor p@5 (0.739), y la de dos el mejor p@1 (0.475). Cambiar la cercanía por la intermediación (RRF node2vec + BTC + PR) no mejora a node2vec + PR en desarrollo (MRR 0.567).

*Prueba (168 descripciones, última ejecución de la etapa 7).* RRF node2vec + CC + PR es el mejor en los cuatro indicadores (0.548 / 0.768 / 0.804, MRR 0.664), seguido por MRR de RRF node2vec + BTC + PR (0.625), RRF node2vec + PR (0.615) y PageRank (0.548). La fusión de tres modelos supera a PageRank en +13 puntos en p@1 y p@3 y en +9.5 puntos en p@5. Con p@5 = 0.804 (135 de 168 descripciones) supera el umbral de 0.8 por una sola descripción (0.8 equivale a 134.4 descripciones), y en desarrollo el mismo modelo obtiene 0.739.

*Precauciones.* Cada descripción pesa 0.006 en prueba y 0.004 en desarrollo, por lo que diferencias de uno o dos puntos entre modelos cercanos no son concluyentes (por ejemplo, el empate de MRR en desarrollo entre las dos fusiones). Desarrollo y prueba no se mueven en el mismo sentido para todos los modelos: node2vec baja en p@1 (0.308 a 0.268) y sube en p@5 (0.525 a 0.589), la cercanía sube en p@1 (0.221 a 0.262) y PageRank queda casi igual (0.428 a 0.417); como en la sección 6.2b, los conceptos del corpus nuevo son distintos y no se probó si esa diferencia se debe a ellos. En desarrollo la elección entre las dos fusiones con node2vec no es nítida (MRR 0.576 frente a 0.573), así que la preferencia por la de tres modelos se apoya en su p@3 y p@5 de desarrollo y en su resultado en prueba, y no debe presentarse como una selección hecha solo por MRR en desarrollo. 
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
| **Cercanía (reducido, asoc)** | 0.221 | 0.428 | 0.525 | 0.360 | 0.964 | 0.262 | 0.476 | 0.560 | 0.405 | 0.994 |
| **Cercanía (reducido, frec)** | 0.196 | 0.355 | 0.514 | 0.329 | 0.967 | 0.173 | 0.333 | 0.423 | 0.303 | 0.994 |
| **Cercanía (reducido, tiempo)** | 0.076 | 0.214 | 0.283 | 0.188 | 0.967 | 0.101 | 0.232 | 0.327 | 0.223 | 0.994 |
| **Intermediación subconjunto (completo, asoc)** | 0.264 | 0.406 | 0.446 | 0.340 | 0.743 | 0.280 | 0.458 | 0.542 | 0.389 | 0.917 |
| **Intermediación subconjunto (completo, frec)** | 0.185 | 0.326 | 0.384 | 0.271 | 0.754 | 0.185 | 0.375 | 0.452 | 0.302 | 0.929 |
| **Intermediación subconjunto (completo, tiempo)** | 0.130 | 0.207 | 0.257 | 0.186 | 0.746 | 0.143 | 0.280 | 0.369 | 0.233 | 0.911 |
| **Intermediación (reducido, asoc)** | 0.170 | 0.362 | 0.453 | 0.296 | 0.884 | 0.155 | 0.393 | 0.560 | 0.318 | 0.964 |
| **Intermediación (reducido, frec)** | 0.138 | 0.304 | 0.366 | 0.249 | 0.895 | 0.143 | 0.327 | 0.423 | 0.279 | 0.976 |
| **Intermediación (reducido, tiempo)** | 0.083 | 0.207 | 0.261 | 0.185 | 0.891 | 0.101 | 0.220 | 0.339 | 0.216 | 0.958 |
| **PageRank ($\alpha=0.5$, completo, asoc)** | 0.428 | 0.620 | 0.703 | 0.549 | 0.964 | 0.417 | 0.637 | 0.708 | 0.548 | 0.994 |
| **node2vec (Mejor config)** | 0.308 | 0.475 | 0.525 | 0.409 | 0.964 | 0.268 | 0.518 | 0.589 | 0.406 | 0.994 |
| **RRF node2vec + PR** | **0.475** | 0.652 | 0.696 | **0.576** | 0.964 | 0.494 | 0.720 | 0.738 | 0.615 | 0.994 |
| **RRF PR + CC (sin node2vec)** | 0.341 | 0.598 | 0.688 | 0.493 | 0.964 | 0.375 | 0.583 | 0.714 | 0.520 | 0.994 |
| **RRF node2vec + BTC + PR** | 0.464 | 0.627 | 0.681 | 0.567 | 0.964 | 0.518 | 0.690 | 0.762 | 0.625 | 0.994 |
| **RRF node2vec + CC + PR** | 0.438 | **0.667** | **0.739** | 0.573 | 0.964 | **0.548** | **0.768** | **0.804** | **0.664** | 0.994 |

En negrita, el mejor valor de cada indicador (p@1, p@3, p@5 y MRR). Las filas de node2vec, PageRank, cercanía (asociación) y las fusiones provienen de `etapa7_tabla_final_asoc.csv` (preprocesamiento con minúsculas antes de spaCy, sección 5.3); las demás (grado, intermediación, frecuencia y tiempo) no aparecen en ese archivo y se mantienen con los valores anteriores.

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
### 7.3 Casos de Max (P3): node2vec y RRF node2vec + PageRank

Se analizan los fallos del conjunto de desarrollo (grafo de asociación, 276 descripciones) de dos modelos, **RRF node2vec + PageRank** (`--modelo rrf2`, el valor por defecto) y **node2vec solo**, con 5 casos de cada uno; después se comparan con PageRank (7.3.3). Un fallo es una descripción cuyo concepto no está en el top 5. Las listas completas se generan con `python paso_a_paso_2.py 6 --grafo asoc` (`resultados_2/fallos_rrf2_asoc.csv`, `fallos_Node2vec_asoc.csv` y `fallos_PR_asoc.csv`); la columna `causa_automatica` marca las descripciones sin lemas y las de concepto fuera del grafo, y la causa de los casos de abajo se asignó a mano. «Posición» es el lugar del concepto correcto en la lista de candidatos (hasta 100).

| Modelo | Fallos | Sin lemas en el grafo | Concepto fuera del grafo | Casi aciertos (6 a 10) | Posiciones 11 a 50 | Posiciones 51 a 100 | No aparece en las 100 primeras |
|---|---|---|---|---|---|---|---|
| node2vec | 131 | 12 | 9 | 17 | 43 | 15 | 35 |
| PageRank | 82 | 12 | 9 | 23 | 28 | 3 | 7 |
| RRF node2vec + PR | 84 | 12 | 9 | 20 | 23 | 10 | 10 |

- Los 21 fallos de «sin lemas» (12) y «concepto fuera del grafo» (9: *aspiradora*, 5; *playera*, 4) son los mismos en los tres modelos y no dependen de ellos: son el 25 % de los fallos de RRF. Los demás son modelables: 110 en node2vec, 61 en PageRank y 63 en RRF.
- Los 84 fallos de RRF son también fallos de node2vec: la fusión nunca falla donde node2vec acierta. Frente a PageRank, RRF corrige 10 fallos y pierde 12 aciertos (7.3.3).
- En los tres modelos, las descripciones cuyo único lema es «animal» devuelven el mismo top 5 sin importar el concepto (7.3.1 y 7.3.2).

#### 7.3.1 RRF node2vec + PageRank

| # | Concepto | Descripción | Lemas | Posición RRF | Top 5 (RRF) | PageRank / node2vec | Causa |
|---|---|---|---|---|---|---|---|
| 1 | plátano | «Mono» | mono | 10 | changar, chimpancé, primate, selva, banana | PR acierta; node2vec 28 | **Descripción de una sola palabra, casi acierto por sinónimo y acierto de PageRank que la fusión pierde.** «mono» solo evoca a los primates; «banana», sinónimo de plátano, aparece en el top 5, por lo que p@5 subestima la utilidad del sistema. PageRank sí lo pone en el top 5 y node2vec lo deja en el lugar 28. Explicación plausible, no verificada: changar, chimpancé y primate, que node2vec también coloca arriba, desplazan al concepto que solo PageRank favorece. |
| 2 | caballo | «animal que relincha» | animal | 44 | hipopótamo, jirafa, león, tigre, cebra | PR 23; node2vec no aparece | **Nodo genérico y palabra fuera del vocabulario.** «relinchar» no está en el grafo y solo sobrevive «animal», cuyos vecinos son animales muy conectados. Es el patrón más repetido: 6 descripciones (*borrego* ×2, *caballo* ×2, *cochino*, *guajolote*) con el mismo top 5. |
| 3 | aspiradora | «Escoba eléctrica» | escoba, eléctrico | no aparece | barrer, limpieza, cepillo_de_dientes, recogedor, trapeador | PR y node2vec: no aparece | **Concepto fuera del grafo (variante léxica).** «aspiradora» no es un nodo; sí lo es «aspirador», que otras descripciones de este concepto sí devuelven. No es un fallo del modelo sino del conjunto de evaluación; afecta a las 5 descripciones de *aspiradora* y a las 4 de *playera*. |
| 4 | ojo | «Órgano que usamos para ver» | órgano, ver | 38 | doctor, enfermedad, curar, bata, medicina | PR acierta; node2vec no aparece | **Polisemia que node2vec introduce en la fusión.** «órgano» arrastra el campo médico: el top 5 de node2vec es ayudar, medicar, gokú, doctor, alergólogo y el concepto queda fuera de sus 100 primeras posiciones. PageRank acierta, pero RRF hereda la deriva médica y lo deja en el lugar 38. |
| 5 | cebra | «Burro» | burro | 56 | asno, carga, politécnico, rebuznar, oreja | PR 24; node2vec no aparece | **Polisemia y descripción por sinónimo.** El modelo devuelve los vecinos de la palabra (asno, rebuznar, carga), no el concepto que se describe con ella; «politécnico» indica que «burro» se asocia también a otro sentido. Además, la descripción no contiene ningún rasgo que distinga a la cebra. |

#### 7.3.2 node2vec

node2vec falla en 131 descripciones. Los cinco casos siguientes los aciertan PageRank y RRF.

| # | Concepto | Descripción | Lemas | Posición | Top 5 (node2vec) | Causa |
|---|---|---|---|---|---|---|
| 1 | zanahoria | «lo que comen los conejos» | conejo | 19 | pascua, bugs_bunny, apapachador, peluchón, despellejar | **Relación no capturada con un solo lema.** node2vec devuelve el campo de «conejo» (pascua, bugs_bunny, peluchón), no lo que el conejo come. Hipótesis, no verificada con las aristas de NAP: PageRank acierta porque sigue la asociación directa conejo → zanahoria, mientras que el vector de node2vec resume el contexto de las caminatas y no la arista directa. |
| 2 | pájaro | «Vuela en el cielo y no es avión» | volar, cielo, avión | 32 | aerolínea, helicóptero, vuelo, mexicana, aeroplano | **Negación ignorada.** «avión» es un lema de la descripción y sus vecinos (aerolínea, aeroplano, helicóptero) dominan el top 5, aunque la descripción dice que no es un avión: la consulta es una suma de vectores y no distingue «no es». |
| 3 | ganso | «Pato pato» | pato | 15 | cuac, donald, lucas, parvada, paperino | **Descripción de una palabra y nodos de personajes.** Con un solo lema, el top 5 mezcla el campo de «pato» con personajes de dibujos animados (donald, lucas, paperino); el concepto está cerca de «pato», pero queda en el lugar 15. |
| 4 | león | «Ruge y vive en la selva» | vivir, selva | 21 | zoológico, casa, vivienda, jirafa, hogar | **Verbo polisémico y nodos genéricos.** «vivir» arrastra el vocabulario de la vivienda (casa, vivienda, hogar), y «zoológico» y «jirafa» son nodos muy conectados del campo de los animales. |
| 5 | avión | «medio de transporte que vuela» | medio, transporte, volar | 6 | helicóptero, hélice, metralleta, afi, militar | **Casi acierto con vecinos de la misma categoría y nodos ruidosos.** helicóptero y hélice son vecinos legítimos, pero metralleta y afi no tienen relación con la descripción; el concepto queda en el lugar 6 y un p@10 sí lo contaría como acierto. |

Además, 7 descripciones de animales (*borrego* ×2, *caballo* ×2, *cochino*, *guajolote* y *león*) devuelven en node2vec exactamente el mismo top 5 (zoológico, selva, salvaje, áfrica, implacable) porque solo sobrevive el lema «animal». «rico» aparece en el top 5 de 8 fallos de node2vec, sin relación semántica evidente con las descripciones.

#### 7.3.3 PageRank frente a node2vec (desarrollo)

PageRank acierta 53 descripciones en las que node2vec falla, y node2vec acierta 4 en las que PageRank falla; los 78 fallos restantes de ambos son comunes (21 no modelables y 57 modelables).

- **Dónde PageRank es mejor.** (i) Descripciones con un lema que tiene una asociación directa fuerte con el concepto (conejo → zanahoria, pato → ganso, extremidad → mano): PageRank devuelve los vecinos directos del lema, mientras que node2vec los mezcla con todo el campo semántico. Es una hipótesis que se puede comprobar consultando las aristas de NAP. (ii) Lemas polisémicos o negados: en los casos vistos (*ojo* con «órgano», *pájaro* con «no es avión») PageRank acierta y node2vec no. El número de lemas no explica la diferencia: las 53 descripciones tienen 2.98 lemas de media, casi igual que los 2.85 de todos los fallos modelables de node2vec. En 24 de las 53, node2vec coloca el concepto más allá de la posición 20 o fuera de la lista.
- **Dónde node2vec es mejor.** Las 4 descripciones que acierta node2vec y falla PageRank (*camisa*, *fresa*, *melón* y *pan*) tienen entre 3 y 8 lemas (4.75 de media, frente a 2.85), y PageRank las deja cerca del top 5 (*camisa* 12, *fresa* 6, *melón* 10, *pan* 6) con nodos genéricos: dinero, plastilina y hot_cakes en «pan»; calzón, agua y huevo en «melón»; collar, playero y vela en «camisa»; niño, sandía y cacahuate en «fresa». Hipótesis, no verificada: con varios lemas, la probabilidad de reinicio se reparte y parte de ella termina en nodos muy conectados, mientras que node2vec suma los vectores de los lemas y busca la zona donde coinciden. Cuatro casos son pocos para generalizar.
- **Complementariedad.** RRF corrige 10 fallos de PageRank: las 4 descripciones anteriores y 6 en las que ambos modelos fallan (*fresa*, *helado*, *león* «El animal del escudo de Gryffindor», *pantalón*, *payaso* y *queso* «Amarillo y con agujeros»), en las que la fusión sube el concepto al top 5 aunque ninguno lo tuviera ahí.
- **Coste de la fusión.** RRF falla en 12 descripciones que PageRank acierta (*bicicleta*, *cebra* «equino», *cochino* «el animal de la alcancía», *dulce*, *elote* «Tortilla», *espagueti*, *falda*, *gelatina*, *mano* «Pie», *melón*, *ojo* y *plátano* «Mono»); en las 12, node2vec también falla. Por eso RRF pasa de 82 a 84 fallos en p@5 aunque mejora p@1 (0.428 a 0.475) y el MRR (0.549 a 0.576). Ejemplo: *falda*, «tela en forma circular para mujeres» (lemas forma, circular, mujer): node2vec devuelve amasar, popular, escultura, viscosa, plastilina; RRF, plastilina, moldear, moldeable, figura, lámpara, y el concepto queda en el lugar 31.

#### 7.3.4 Mejoras posibles, no aplicadas

Todas son hipótesis sin probar.
- **Nodo vacío.** Un nodo con nombre vacío (cadena vacía) aparece en el top 5 de 11 fallos de node2vec, 13 de PageRank y 12 de RRF (por ejemplo, *abrigo*, *boca*, *mano*, *manzana*, *pan*, *plátano*). Probablemente proviene de celdas vacías de `NAP.xls` convertidas a texto (no verificado); habría que eliminarlo del grafo.
- **RRF ponderado** (menos peso para node2vec) o ajuste de `k` (k = 60 no se ajustó), para reducir los 12 aciertos de PageRank que la fusión pierde.
- **Nodos poco frecuentes o genéricos.** Quitar de las listas de node2vec los nodos de grado muy bajo (donald, paperino, afi) y penalizar los nodos muy conectados en las listas de ambos modelos (dinero, plastilina, lavadora, zoológico).
- **Lemas genéricos** (animal, forma, órgano, hacer): ponderarlos por su grado en el grafo; la ponderación IDF calculada sobre las descripciones no mejoró (MRR 0.400 frente a 0.409).
- Lematizar la frase completa con contexto, en lugar de palabra por palabra (7.2; «saca» sigue lematizándose como «saco» en *queso*), y tratar *aspirador* y *aspiradora* como equivalentes en la evaluación.

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
* **Conclusiones:** El uso de **PageRank personalizado** sobre el grafo completo demostró una superioridad notable frente a las medidas clásicas de centralidad al considerar la conectividad global y la fuerza de asociación. Asimismo, las técnicas de fusión de rankings (*RRF*) integrando grafos y representaciones densas por `node2vec` potencian significativamente la robustez del sistema, alcanzando un $p@5$ superior al $80\%$ en el conjunto de prueba (0.804, 135 de 168 descripciones, con la fusión de tres modelos).
* **Limitaciones:** El sistema depende estrictamente de la cobertura léxica de las normas NAP, lo que impide resolver consultas cuyos términos o variantes no formen parte de los nodos originales. Adicionalmente, la lematización independiente por palabra elimina matices contextuales y sintácticos importantes que podrían desambiguar términos polisémicos.

## 10. Uso de IA generativa
Se declara el uso de Claude (Anthropic) como apoyo. Las descripciones del conjunto de prueba fueron escritas por personas; no se usó IA para simular participantes ni para fabricar normas o respuestas.

Jour 6 : répétition de la présentation (15 min). N’importe qui peut être interrogé sur n’importe quelle partie, donc relisez le code des autres. Déclarez l’usage de l’IA.

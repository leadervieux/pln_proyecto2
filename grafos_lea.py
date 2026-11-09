"""
grafos_lea.py - Modèles de graphes de Léa (notebook grafo_lea-3.ipynb passé en .py, sans Colab ni Drive).

Le comportement est identique à celui du notebook : mêmes graphes, mêmes prétraitements, mêmes modèles.
Ce qui n'est PAS repris : rendimiento*, errores, estadisticas_listas (l'évaluation se fait avec evaluacion.py).

Dépendances : pip install xlrd networkx spacy nltk
              python -m spacy download es_core_news_sm

Utilisation depuis un autre fichier :
    import grafos_lea as gl
    grafos = gl.construye_grafos()                  # {"frec": ..., "tiempo": ..., "asoc": ...}
    modelo = gl.crea_modelo(gl.diccionario_pr, grafos["asoc"], alpha=0.5)
    modelo("animal que ladra")                      # -> liste de mots candidats, du meilleur au pire
"""
import os
import string
from functools import lru_cache

import networkx as nx
import nltk
import spacy
import xlrd

# ------------------------------------------------------------------ Config
DIRECTORIO = "./"
NAP_XLS = os.path.join(DIRECTORIO, "NAP.xls")
ALPHA_PR = 0.5          # alpha de PageRank fixé par Léa sur le jeu de développement

_nlp = None
_stop = None


def _cargar_nlp():
    global _nlp, _stop
    if _nlp is None:
        _nlp = spacy.load("es_core_news_sm")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        _stop = set(nltk.corpus.stopwords.words("spanish"))
    return _nlp, _stop


@lru_cache(maxsize=None)
def _lema(palabra):
    nlp, _ = _cargar_nlp()
    return nlp(palabra)[0].lemma_


# ------------------------------------------------------------------ Graphes
def Construye_Grafo(ruta=NAP_XLS):
    """Charge NAP.xls et renvoie TROIS graphes avec la même structure, qui ne diffèrent que par les poids :
        weight = distance (pour les chemins les plus courts : intermédiarité, proximité)
        sim    = force    (pour PageRank et node2vec)
      GrafoFrec  : weight = 1/frecuencia      sim = frecuencia
      GrafoTiempo: weight = tiempo            sim = 1/(tiempo+1)
      GrafoAsoc  : weight = 100 - asociado    sim = asociado
    """
    libro = xlrd.open_workbook(ruta)
    hoja = libro.sheet_by_index(0)
    GrafoFrec = nx.Graph()
    GrafoTiempo = nx.Graph()
    GrafoAsoc = nx.Graph()
    num_filas = hoja.nrows
    estimulo = ""
    for fila in range(0, num_filas - 1):
        cadena = hoja.cell(fila, 0).value
        if cadena == "--PALABRAS--":
            continue
        elif cadena == "======":
            estimulo = hoja.cell(fila + 1, 4).value
        elif cadena == "" or cadena == "=":
            continue
        elif hoja.cell(fila, 2).value == "":
            continue
        else:
            frecuencia = hoja.cell(fila, 1).value
            tiempo = hoja.cell(fila, 2).value
            asociado = hoja.cell(fila, 3).value
            palabra = hoja.cell(fila, 4).value
            estimulo = estimulo.strip()
            lema = str(palabra).strip()
            GrafoFrec.add_edge(estimulo, lema, weight=1 / float(frecuencia), sim=float(frecuencia))
            GrafoTiempo.add_edge(estimulo, lema, weight=float(tiempo), sim=1 / (float(tiempo) + 1))
            GrafoAsoc.add_edge(estimulo, lema, weight=100 - float(asociado), sim=float(asociado))
    return GrafoFrec, GrafoTiempo, GrafoAsoc


def construye_grafos(ruta=NAP_XLS):
    """Pareil, mais renvoie un dictionnaire {'frec', 'tiempo', 'asoc'} (plus pratique)."""
    f, t, a = Construye_Grafo(ruta)
    return {"frec": f, "tiempo": t, "asoc": a}


# ------------------------------------------------------------------ Prétraitement
def limpia_lematiza(cadena):
    """Ponctuation et mots vides retirés, lemmatisation mot par mot. Renvoie UNE CHAÎNE de lemmes."""
    _, stop = _cargar_nlp()
    limpiado = ""
    for c in string.punctuation:
        cadena = cadena.replace(c, "")
    cadena = cadena.strip()
    for palabra in cadena.split():
        if palabra not in stop:
            limpiado += _lema(palabra) + " "
    return limpiado


def _subconjunto_lemas(grafo, definicion):
    """Lemmes de la description présents dans le graphe (comme dans le notebook : sans dédoublonner)."""
    texto = limpia_lematiza(definicion).lower()
    return [t for t in texto.split(" ") if t != "" and t in grafo.nodes()]


def conceptos(lista, subconjunto):
    """Garde au plus 100 candidats, sans les mots de la description et avec un score > 0."""
    datos = []
    tope = min(len(lista), 100)
    for x in range(0, tope):
        if str(lista[x][0]) not in subconjunto and float(lista[x][1]) > 0:
            datos.append(str(lista[x][0]))
    return datos


def reducir_grafo(subconjunto, grafo_original):
    """Sous-graphe formé des lemmes de la description et de leurs voisins directs."""
    nuevos_nodos = []
    nuevos_nodos.extend(subconjunto)
    for nodo in subconjunto:
        for vecino in grafo_original[nodo]:
            nuevos_nodos.append(vecino)
    return grafo_original.subgraph(nuevos_nodos)


def _top(resultado):
    return sorted(resultado.items(), key=lambda x: x[1], reverse=True)[0:99]


# ------------------------------------------------------------------ Modèles
def diccionario_cc(grafo, definicion):
    """Proximité (closeness), graphe réduit, poids 'weight'."""
    sub = _subconjunto_lemas(grafo, definicion)
    if len(sub) > 0:
        resultado = nx.closeness_centrality(reducir_grafo(sub, grafo), distance="weight")
        return conceptos(_top(resultado), sub)
    return []


def diccionario_btc(grafo, definicion):
    """Intermédiarité par sous-ensemble, graphe complet (LENT)."""
    sub = _subconjunto_lemas(grafo, definicion)
    if len(sub) > 0:
        resultado = nx.betweenness_centrality_subset(grafo, sub, sub, weight="weight", normalized=True)
        return conceptos(_top(resultado), sub)
    return []


def diccionario_dc(grafo, definicion):
    """Degré, graphe réduit."""
    sub = _subconjunto_lemas(grafo, definicion)
    if len(sub) > 0:
        resultado = nx.degree_centrality(reducir_grafo(sub, grafo))
        return conceptos(_top(resultado), sub)
    return []


def diccionario_btc_reducido(grafo, definicion):
    """Intermédiarité, graphe réduit."""
    sub = _subconjunto_lemas(grafo, definicion)
    if len(sub) > 0:
        resultado = nx.betweenness_centrality(reducir_grafo(sub, grafo), weight="weight", normalized=True)
        return conceptos(_top(resultado), sub)
    return []


def diccionario_pr(grafo, definicion, alpha=ALPHA_PR):
    """PageRank personnalisé, graphe complet, poids 'sim' (force). Redémarrage sur les lemmes de la description."""
    sub = _subconjunto_lemas(grafo, definicion)
    if len(sub) > 0:
        pers = {n: (1 if n in sub else 0) for n in grafo.nodes()}
        resultado = nx.pagerank(grafo, alpha=alpha, personalization=pers, weight="sim")
        return conceptos(_top(resultado), sub)
    return []


# ------------------------------------------------------------------ Interface « modèle »
def crea_modelo(funcion, grafo, **kw):
    """Transforme diccionario_xx(grafo, definicion) en modele(definicion) -> liste de mots.
    C'est le format attendu par evaluacion.evaluar et par node2vec_nap.modelo_rrf.
    Les résultats sont gardés en mémoire (même description = même réponse, sans recalcul)."""
    memoria = {}

    def modelo(definicion):
        if definicion not in memoria:
            memoria[definicion] = funcion(grafo, definicion, **kw)
        return list(memoria[definicion])

    return modelo


if __name__ == "__main__":
    grafos = construye_grafos()
    for nombre, g in grafos.items():
        print(f"{nombre}: {g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes")

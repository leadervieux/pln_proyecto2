"""
node2vec_nap.py - Partie de Max : node2vec, fusion (RRF), démo.

Ce fichier ne construit PAS le graphe et ne décide PAS de dév/test :
  - le graphe vient de grafos_lea.py (les trois graphes de Léa, avec l'attribut 'sim' = force) ;
  - les données et la partition viennent de evaluacion.py (corpus de classe = développement,
    nuevas_desc.csv = test, avec PARTICION = "fuente").

Dépendances :
    pip install node2vec gensim networkx xlrd spacy nltk pandas
    python -m spacy download es_core_news_sm

Lancement : voir paso_a_paso.py
"""
import math
import os
import string
from collections import Counter
from functools import lru_cache

import nltk
import pandas as pd
import spacy
from gensim.models import KeyedVectors
from node2vec import Node2Vec

import evaluacion as ev

import paso_a_paso_2 as ps

# ------------------------------------------------------------------ Config
DIRECTORIO = "./"
NAP_XLS = os.path.join(DIRECTORIO, "NAP.xls")
EMB_DIR = os.path.join(DIRECTORIO, "embeddings")
SEED = 42

# Configuration de référence. On change UNE seule variable à la fois (VARIANTES).
CONFIG_REF = dict(dimensions=128, walk_length=80, num_walks=10, p=1.0, q=1.0)
VARIANTES = [
    {},
    {"dimensions": 64}, {"dimensions": 300},
    {"p": 0.5}, {"p": 2.0},
    {"q": 0.5}, {"q": 2.0},
    {"walk_length": 30},
    {"num_walks": 20},
]


# ------------------------------------------------- Prétraitement du texte
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


def limpia_lematiza(cadena):
    """Même prétraitement que le notebook de classe et que grafos_lea.py : ponctuation et mots vides
    retirés, lemmatisation mot par mot. Renvoie une LISTE de lemmes en minuscules.
    Si spaCy renvoie un lemme à plusieurs mots (« mole » -> « mo él »), on le coupe en deux,
    comme le fait Léa avec .split(" ")."""
    _, stop = _cargar_nlp()
    for c in string.punctuation:
        cadena = cadena.replace(c, "")
    lemas = []
    for p in cadena.split():
        if p not in stop:
            lemas += _lema(p).lower().split()
    return lemas


# ---------------------------------------------------------------- node2vec
def nombre_config(grafo, weight_key, dimensions, walk_length, num_walks, p, q, seed=SEED):
    """Nom du fichier de cache. Il contient le NOM DU GRAPHE (asoc, frec, tiempo) :
    sans lui, deux graphes avec les mêmes hyperparamètres se partageraient le même cache."""
    return f"{grafo}_{weight_key}_d{dimensions}_l{walk_length}_w{num_walks}_p{p}_q{q}_s{seed}"


def entrena_node2vec(G, grafo, weight_key="sim", dimensions=128, walk_length=80,
                     num_walks=10, p=1.0, q=1.0, window=10, seed=SEED, forzar=False):
    """Entraîne (ou recharge depuis le disque) des embeddings node2vec.
    grafo      = nom du graphe ('asoc', 'frec', 'tiempo'), utilisé seulement pour nommer le cache.
    weight_key = attribut d'arête utilisé comme poids : une FORCE ('sim'), jamais une distance ('weight').
    workers=1 + seed pour la reproductibilité. Aucune description n'est utilisée ici : pas de fuite dév/test."""
    if not all(weight_key in d for _, _, d in G.edges(data=True)):
        raise ValueError(f"Certaines arêtes n'ont pas l'attribut '{weight_key}'")
    os.makedirs(EMB_DIR, exist_ok=True)
    nombre = nombre_config(grafo, weight_key, dimensions, walk_length, num_walks, p, q, seed)
    print(nombre)
    ruta = os.path.join(EMB_DIR, nombre + ".emb")
    if os.path.exists(ruta) and not forzar:
        return KeyedVectors.load_word2vec_format(ruta, binary=False)
    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                   num_walks=num_walks, p=p, q=q, weight_key=weight_key,
                   workers=1, seed=seed, quiet=True)
    modelo = n2v.fit(window=window, min_count=1, batch_words=4, seed=seed)
    modelo.wv.save_word2vec_format(ruta)
    return modelo.wv


# ------------------------------------- Description -> vecteur -> classement
def calcula_idf(listas_de_lemas):
    """IDF calculé sur un ensemble de descriptions (celles du dév)."""
    N = len(listas_de_lemas)
    df = Counter(l for lemas in listas_de_lemas for l in set(lemas))
    return {l: math.log(N / c) for l, c in df.items()}


def idf_dev(datos):
    """IDF calculé UNIQUEMENT sur les descriptions de développement."""
    defs = datos.loc[datos["particion"] == "desarrollo", "definicion"]
    return calcula_idf([limpia_lematiza(d) for d in defs])


def vector_definicion(wv, lemas, modo="suma", idf=None):
    """modo : 'suma', 'media' ou 'idf' (somme pondérée par l'IDF).
    Les lemmes répétés ne comptent qu'une fois (comme semillas() de evaluacion.py).
    Renvoie (vecteur ou None, lemmes utiles)."""
    utiles = []
    for l in lemas:
        if l and l in wv and l not in utiles:
            utiles.append(l)
    if not utiles:
        return None, utiles
    if modo == "suma":
        vec = sum(wv[l] for l in utiles)
    elif modo == "media":
        vec = sum(wv[l] for l in utiles) / len(utiles)
    elif modo == "idf":
        if idf is None:
            raise ValueError("modo='idf' demande idf=idf_dev(datos)")
        por_defecto = max(idf.values())
        vec = sum(idf.get(l, por_defecto) * wv[l] for l in utiles)
    else:
        raise ValueError(f"modo inconnu : {modo}")
    return vec, utiles


def ranking(wv, definicion, modo="suma", idf=None, topn=100):
    """Liste classée de mots candidats (comme conceptos() du notebook :
    on retire les mots de la description et les similarités <= 0)."""
    vec, utiles = vector_definicion(wv, limpia_lematiza(definicion), modo, idf)
    if vec is None:
        return []
    excluidos = set(utiles)
    vecinos = wv.most_similar(positive=[vec], topn=topn + len(excluidos))
    return [w for w, s in vecinos if w not in excluidos and s > 0][:topn]


def crea_modelo(wv, modo="suma", idf=None, topn=100):
    """Un 'modèle' = fonction definicion -> liste classée de mots (interface de evaluacion.evaluar).
    Les résultats sont gardés en mémoire pour ne pas recalculer la même description."""
    memoria = {}

    def modelo(definicion):
        if definicion not in memoria:
            memoria[definicion] = ranking(wv, definicion, modo, idf, topn)
        return list(memoria[definicion])

    return modelo


# ------------------------------------------------------------------ Fusion
def rrf(listas, k=60, topn=100):
    """Reciprocal Rank Fusion : score[mot] += 1 / (k + rang), rang à partir de 1."""
    puntaje = Counter()
    for lista in listas:
        for rango, palabra in enumerate(lista, start=1):
            puntaje[palabra] += 1 / (k + rango)
    return [w for w, _ in puntaje.most_common(topn)]


def modelo_rrf(modelos, k=60):
    """Fusionne plusieurs modèles (node2vec + graphe, ou node2vec + proximité + PageRank)."""
    return lambda definicion: rrf([m(definicion) for m in modelos], k)


# ------------------------------------------------------------- Évaluation
def evaluar_dev(modelo, datos):
    """Évalue sur le DÉVELOPPEMENT, toujours. (evaluacion.evaluar prend 'prueba' par défaut :
    ne jamais l'appeler sans particion pendant le réglage.)"""
    return ev.evaluar(modelo, datos, particion="desarrollo")


# ------------------------------------------------------------ Expériences (dév seulement)
def _etiqueta(var):
    return "base" if not var else ", ".join(f"{k}={v}" for k, v in var.items())


def experimentos(G, datos, grafo, weight_key="sim", modo="suma"):
    """Un tableau : une ligne par configuration (une variable change à la fois), résultats sur le dév."""
    filas = []
    for var in VARIANTES:
        cfg = {**CONFIG_REF, **var}
        wv = entrena_node2vec(G, grafo, weight_key=weight_key, **cfg)
        res = evaluar_dev(crea_modelo(wv, modo), datos)
        filas.append({"cambio": _etiqueta(var), **cfg, "grafo": grafo, "modo": modo, **res})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in filas[-1].items()})
    return pd.DataFrame(filas)


def comparar_modos(wv, datos):
    """Somme / moyenne / somme pondérée IDF pour un même jeu d'embeddings (dév seulement).
    Remarque : 'suma' et 'media' donnent EXACTEMENT les mêmes classements, car la similarité cosinus
    ignore la longueur du vecteur et la moyenne n'est que la somme divisée par n."""
    idf = idf_dev(datos)
    filas = []
    for modo in ("suma", "media", "idf"):
        res = evaluar_dev(crea_modelo(wv, modo, idf), datos)
        filas.append({"modo": modo, **res})
    return pd.DataFrame(filas)


def comparar_grafos(grafos, datos, cfg=None, weight_key="sim", modo="suma"):
    """Même configuration sur les trois graphes (asoc, frec, tiempo) : quelle force marche le mieux ?"""
    cfg = cfg or CONFIG_REF
    filas = []
    for nombre, G in grafos.items():
        wv = entrena_node2vec(G, nombre, weight_key=weight_key, **cfg)
        res = evaluar_dev(crea_modelo(wv, modo), datos)
        filas.append({"grafo": nombre, **cfg, "modo": modo, **res})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in filas[-1].items()})
    return pd.DataFrame(filas)


# -------------------------------------------------------------------- Démo
def demo(modelo, k=5):
    """Démo en direct dans le terminal : tape une description, obtiens le top k."""
    while True:
        definicion = input("Definición (vacío para salir): ").strip()
        if not definicion:
            break
        candidatos = modelo(definicion)[:k]
        print("  ->", ", ".join(candidatos) if candidatos else "(sin resultados)")


def demo_gradio(modelo, k=5):
    """Démo web (+3 points). Nécessite : pip install gradio"""
    import gradio as gr
    gr.Interface(
        fn=lambda d: ", ".join(modelo(d)[:k]) or "(sin resultados)",
        inputs=gr.Textbox(label="Descripción", placeholder="animal que ladra y es el mejor amigo del hombre"),
        outputs=gr.Textbox(label=f"Top {k} palabras candidatas"),
        title="Diccionario inverso (NAP)",
    ).launch()


if __name__ == "__main__":
    print(__doc__)
    print("Ce fichier est une bibliothèque. Lance plutôt : python paso_a_paso.py 1")
    grafos, datos = ps.cargar()
    wv = entrena_node2vec(grafos["asoc"], "asoc", dimensions=200, walk_length=20, num_walks=10, p=1.0, q=1.0 )
    res = evaluar_dev(crea_modelo(wv, "suma"), datos)
    print(res)

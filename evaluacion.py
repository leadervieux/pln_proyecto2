"""
evaluacion.py

    semillas(definicion)     -> lemmes utiles d'une description (ceux qui sont dans le graphe)
    cargar_datos(ruta)       -> DataFrame avec toutes les descriptions + partition développement/test
    evaluar(modelo, datos)   -> p@1, p@3, p@5, MRR, couverture
    tabla(modelos, datos)    -> tableau comparatif (développement et test)
    fallos(modelo, datos)    -> descriptions où le modèle échoue (analyse d'erreurs)

Un "modèle" est une fonction modele(definicion: str) -> list[str] qui renvoie des mots candidats
ordonnés du meilleur au pire (peut renvoyer plus que TOPN ; il est coupé). À l'intérieur du modèle, 
utilisez semillas(definicion) pour obtenir les lemmes de la description.
"""
import glob
import os
import random
import re
import string
import unicodedata

import pandas as pd

# ----------------------------------------------------------------------------- configuration
SEMILLA = 42                 # Graine pour la partition (ne pas changer une fois l'évaluation commencée)
FRAC_DESARROLLO = 0.4      # Fraction de concepts en développement, le reste est pour le test
TOPN = 100                   # Nombre maximal de candidats considérés
PREPROCESO = "curso"         # "curso" = comme limpia_lematiza du cours ; "spacy" = lemmatise la phrase complète
PARTICION = "fuente"         # "fuente" : corpus du cours = développement, descriptions de l'équipe = test (comme le README)
                             # "concepto" : partition aléatoire par concept (fixée dans particion_conceptos.csv)
USAR_ALIAS = False           # False = comme la classe (Tableau 1) : 'aspiradora' et 'playera' ne sont pas des nœuds du graphe,
                             # donc leurs descriptions ne peuvent jamais être trouvées. True = on les relie à leur variante.

# Concepts du corpus qui ne sont pas des nœuds NAP mais dont une variante l'est (vérifié dans NAP.xls).
# Utilisé seulement si USAR_ALIAS = True. ('papas' n'en a pas besoin : le titre du fichier Papas.txt est « Papa ».
# 'tijera' est déjà un nœud ; 'tijeras' n'en est pas un.)
ALIAS = {"playera": "playero", "aspiradora": "aspirador"}


def _alias(c):
    return ALIAS.get(c, c) if USAR_ALIAS else c

_nlp = None
_STOP = None
_LISTA_STOP = None
_NODOS = None
_cache = {}

# ----------------------------------------------------------------------------- prétraitement
def _cargar_nlp():
    """Charge le modèle spaCy et les stopwords"""
    global _nlp, _STOP, _LISTA_STOP
    if _nlp is not None:
        return
    import spacy
    _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        _STOP = set(nltk.corpus.stopwords.words("spanish"))
        _LISTA_STOP = "nltk"
    except Exception:
        _STOP = set(_nlp.Defaults.stop_words)
        _LISTA_STOP = "spaCy (NLTK non disponible)"
    print(f"[evaluacion] PREPROCESO={PREPROCESO} | mots vides (stopwords) : {_LISTA_STOP}")


def fijar_grafo(G):
    """Indique quels mots existent dans le graphe. À appeler une fois, avant semillas()/cargar_datos()"""
    global _NODOS
    _NODOS = set(G.nodes()) if hasattr(G, "nodes") else set(G)
    _cache.clear()
    con_may = sum(1 for n in _NODOS if str(n) != str(n).strip().lower())
    if con_may > 0.01 * len(_NODOS):
        print(f"[evaluacion] AVERTISSEMENT : {con_may} nœuds avec des majuscules ou des espaces ; "
              "le graphe doit avoir des nœuds en minuscules et sans espaces")


def semillas(definicion):
    """Lemmes utiles de la description ie ceux présents dans le graphe (sans répétition, dans l'ordre d'apparition)"""
    if _NODOS is None:
        raise RuntimeError("Appelez d'abord fijar_grafo(G).")
    clave = (PREPROCESO, definicion)
    if clave in _cache:
        return list(_cache[clave])
    _cargar_nlp()
    vistos = []
    if PREPROCESO == "curso":
        # Identique à limpia_lematiza + filtrage du diccionario_btc du cours
        texto = definicion
        for c in string.punctuation:
            texto = texto.replace(c, "")
        for palabra in texto.strip().split():
            if palabra not in _STOP:
                lema = _nlp(palabra)[0].lemma_.lower()
                if lema and lema in _NODOS and lema not in vistos:
                    vistos.append(lema)
    else:
        for t in _nlp(definicion.lower()):
            if not t.is_alpha or t.text in _STOP or t.lemma_.lower() in _STOP:
                continue
            lema = t.lemma_.lower()
            pal = lema if lema in _NODOS else (t.text if t.text in _NODOS else None)
            if pal and pal not in vistos:
                vistos.append(pal)
    _cache[clave] = tuple(vistos)
    return vistos


def _norm(s):
    """Normalise une chaîne (minuscules et suppression des accents)"""
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def menciona_objetivo(definicion, concepto):
    """True si la description contient le mot cible (ou un dérivé simple : pluriel, diminutif)"""
    objetivos = {_norm(concepto), _norm(_alias(concepto))}
    texto = _norm(definicion)
    if any(len(o) >= 5 and o in texto for o in objetivos):     # sous-chaîne (ex. « telarañas » contient « araña »)
        return True
    for t in re.findall(r"\w+", texto):
        for o in objetivos:
            if len(o) <= 4:
                if t in (o, o + "s", o + "es"):
                    return True
            else:
                raiz = o[:-1] if o[-1] in "aeo" else o
                if t.startswith(raiz):
                    return True
    return False


# ----------------------------------------------------------------------------- données
def _quita_u(s):   # Noms de fichiers du .zip 
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


def _asigna_particion(conceptos, ruta):
    """Partition PAR CONCEPT, figée dans ruta/particion_conceptos.csv pour qu'elle ne change pas avec de nouvelles données"""
    archivo = os.path.join(ruta, "particion_conceptos.csv")
    mapa = {}
    if os.path.exists(archivo):
        previo = pd.read_csv(archivo)
        mapa = dict(zip(previo["concepto"], previo["particion"]))
    nuevos = sorted(c for c in conceptos if c not in mapa)
    if nuevos:
        rng = random.Random(SEMILLA)
        rng.shuffle(nuevos)
        n_dev = int(round(len(nuevos) * FRAC_DESARROLLO))
        for i, c in enumerate(nuevos):
            mapa[c] = "desarrollo" if i < n_dev else "prueba"
        try:
            pd.DataFrame(sorted(mapa.items()), columns=["concepto", "particion"]).to_csv(archivo, index=False)
        except OSError as e:
            print("[evaluacion] Impossible d'enregistrer particion_conceptos.csv :", e)
    return mapa


def cargar_datos(ruta, G=None):
    """Charge corpus/*.txt (sans la ligne de titre) et nuevas_desc.csv (concepto, definicion, fuente[, particion])."""
    if G is not None:
        fijar_grafo(G)
    filas = []
    for f in sorted(glob.glob(os.path.join(ruta, "corpus", "*.txt"))):
        concepto = _quita_u(os.path.basename(f))[:-4].strip().lower()
        lineas = [l.strip() for l in open(f, encoding="utf8").read().splitlines() if l.strip()]
        if lineas and len(lineas[0].split()) == 1:      # La première ligne est le titre
            concepto = lineas[0].strip().lower()        # comme la classe : le concept est le titre (Papas.txt -> « papa »)
            lineas = lineas[1:]
        filas += [dict(concepto=concepto, definicion=l, fuente="curso") for l in lineas]
    df = pd.DataFrame(filas)
    for nombre in ("nuevas_desc.csv", "descripciones_nuevas.csv"):
        p = os.path.join(ruta, nombre)
        if os.path.exists(p):
            dn = pd.read_csv(p)
            if "fuente" not in dn:
                dn["fuente"] = "equipo"
            dn = dn[["concepto", "definicion", "fuente"]].dropna()
            dn["concepto"] = dn["concepto"].astype(str).str.strip().str.lower()
            dn["definicion"] = dn["definicion"].astype(str).str.strip()
            dn = dn[dn["definicion"] != ""]
            df = pd.concat([df, dn], ignore_index=True)
            print(f"[evaluacion] lu {nombre} : {len(dn)} descriptions")
            break
    df = df.drop_duplicates(subset=["concepto", "definicion"]).reset_index(drop=True)
    df["objetivo"] = df["concepto"].map(_alias)
    if PARTICION == "fuente":
        df["particion"] = df["fuente"].map(lambda f: "desarrollo" if f == "curso" else "prueba")
    else:
        df["particion"] = df["concepto"].map(_asigna_particion(df["concepto"].unique(), ruta))
    if _NODOS is not None:
        df["objetivo_en_grafo"] = df["objetivo"].isin(_NODOS)
        df["semillas"] = df["definicion"].apply(semillas)
    else:
        df["objetivo_en_grafo"] = True
        df["semillas"] = [[] for _ in range(len(df))]
        print("[evaluacion] AVERTISSEMENT : pas de graphe ; appelez fijar_grafo(G) puis rechargez les données")
    df["n_semillas"] = df["semillas"].apply(len)
    df["menciona_objetivo"] = [menciona_objetivo(d, c) for d, c in zip(df["definicion"], df["concepto"])]
    return df


def resumen(datos):
    """Affiche et renvoie des statistiques du jeu d'évaluation (pour le rapport)"""
    r = {
        "descripciones": len(datos),
        "conceptos": datos["concepto"].nunique(),
        "cobertura de lemas (% con >=1 lema en el grafo)": round(100 * (datos["n_semillas"] > 0).mean(), 1),
        "lemas útiles por descripción (media)": round(datos["n_semillas"].mean(), 2),
        "conceptos fuera del grafo": sorted(datos.loc[~datos["objetivo_en_grafo"], "concepto"].unique()),
        "descripciones que mencionan la palabra objetivo": int(datos["menciona_objetivo"].sum()),
    }
    print(datos.groupby(["fuente", "particion"]).agg(descripciones=("definicion", "size"),
                                                     conceptos=("concepto", "nunique")))
    for k, v in r.items():
        print(f"{k}: {v}")
    conteo = datos.groupby("concepto").size()
    if (conteo < 5).any():
        print("AVERTISSEMENT : concepts avec moins de 5 descriptions :", {c: int(n) for c, n in conteo[conteo < 5].items()})
    return r


# ----------------------------------------------------------------------------- évaluation
def _subconjunto(datos, particion, excluir_mencion):
    """Filtre les données par partition et/ou en excluant les descriptions qui mentionnent la cible"""
    d = datos if particion is None else datos[datos["particion"] == particion]
    if excluir_mencion:
        d = d[~d["menciona_objetivo"]]
    return d


def _candidatos(modelo, fila, topn):
    sem = set(fila["semillas"])
    try:
        cand = modelo(fila["definicion"]) or []
    except Exception as e:                      # une erreur du modèle ne doit pas arrêter toute l'évaluation
        print("[evaluacion] erreur dans le modèle :", repr(e)[:100], "|", fila["definicion"][:50])
        cand = []
    return [w for w in cand if w not in sem][:topn]


def evaluar(modelo, datos, particion="prueba", topn=TOPN, excluir_mencion=False):
    """
    Métriques d'un modèle. particion : "desarrollo", "prueba" ou None (toutes).
    - p@k : proportion de descriptions dont le concept cible est parmi les k premiers candidats (k = 1, 3, 5).
    - MRR : moyenne de 1/rang du concept cible (0 s'il n'apparaît pas dans les topn premiers).
    - cobertura_modelo : proportion de descriptions pour lesquelles le modèle renvoie au moins un candidat.
    - cobertura_lemas  : proportion de descriptions avec >= 1 lemme dans le graphe (ne dépend pas du modèle).
    - lemas_medios     : nombre moyen de lemmes utiles par description.
    Les mots de la description sont exclus des candidats ici même, comme conceptos() dans le cours.
    """
    d = _subconjunto(datos, particion, excluir_mencion)
    n = len(d)
    if n == 0:
        raise ValueError("Aucune description dans cette partition.")
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    con_cand = 0
    for _, fila in d.iterrows():
        cand = _candidatos(modelo, fila, topn)
        if cand:
            con_cand += 1
        if fila["objetivo"] in cand:
            r = cand.index(fila["objetivo"]) + 1
            rr += 1.0 / r
            for k in hits:
                hits[k] += r <= k
    return {"n": n, "p@1": hits[1] / n, "p@3": hits[3] / n, "p@5": hits[5] / n, "MRR": rr / n,
            "cobertura_modelo": con_cand / n,
            "cobertura_lemas": float((d["n_semillas"] > 0).mean()),
            "lemas_medios": float(d["n_semillas"].mean())}


def tabla(modelos, datos, excluir_mencion_prueba=True, **kw):
    """
    modelos = {"nom": fonction}. Une ligne par modèle, avec développement et test.
    Par défaut, en test on écarte les descriptions qui contiennent le mot cible (comme dans le README) ;
    en développement on garde les 276 descriptions, comparables à la Tabla 1.
    """
    filas = []
    for nombre, fn in modelos.items():
        fila = {"modelo": nombre}
        for part, etq in (("desarrollo", "desa"), ("prueba", "prue")):
            m = evaluar(fn, datos, particion=part, excluir_mencion=(excluir_mencion_prueba and part == "prueba"), **kw)
            fila.update({f"{k} ({etq})": v for k, v in m.items() if k in ("p@1", "p@3", "p@5", "MRR", "cobertura_modelo")})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def fallos(modelo, datos, particion="prueba", k=5, topn=TOPN):
    """Descriptions où le concept cible n'est pas parmi les k premiers, pour l'analyse d'erreurs."""
    d = _subconjunto(datos, particion, False)
    filas = []
    for _, fila in d.iterrows():
        cand = _candidatos(modelo, fila, topn)
        r = cand.index(fila["objetivo"]) + 1 if fila["objetivo"] in cand else None
        if r is None or r > k:
            causa = ("sin lemas en el grafo" if not fila["semillas"]
                     else "objetivo fuera del grafo" if not fila["objetivo_en_grafo"]
                     else "menciona la palabra objetivo" if fila["menciona_objetivo"] else "")
            filas.append(dict(concepto=fila["concepto"], definicion=fila["definicion"],
                              lemas=", ".join(fila["semillas"]), rango=r, top5=", ".join(cand[:5]),
                              causa_automatica=causa, causa_manual=""))
    return pd.DataFrame(filas)

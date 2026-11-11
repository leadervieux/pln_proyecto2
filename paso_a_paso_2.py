"""
paso_a_paso.py - Expériences de Max (node2vec + fusion RRF), dans l'ordre.

Dossier de travail : NAP.xls, nuevas_desc.csv, corpus/, evaluacion.py, grafos_lea.py, node2vec_nap.py

    python paso_a_paso_2.py                 #Genere les résultat sur les graphes de léa
    python paso_a_paso.py 1                 # même config sur les 3 graphes (asoc, frec, tiempo) : quelle force ?
    python paso_a_paso.py 2 --grafo asoc    # une variable à la fois ; enregistre la meilleure config (long)
    python paso_a_paso.py 3 --grafo asoc    # somme / moyenne / IDF avec la meilleure config
    python paso_a_paso.py 4 --grafo asoc    # RRF : node2vec + PageRank (+ proximité), sur le DÉV
    python paso_a_paso.py 5 --grafo asoc    # démo en direct   (--modelo rrf2 ; --gradio pour l'interface web)
    python paso_a_paso.py 6 --grafo asoc    # descriptions où le modèle échoue (analyse d'erreurs, sur le DÉV)
    python paso_a_paso.py 7 --grafo asoc    # TABLEAU FINAL dév + TEST : à lancer UNE SEULE FOIS, à la fin

Convention : dév = corpus de classe (evaluacion.PARTICION = "fuente"), test = nuevas_desc.csv.
Les étapes 1 à 6 n'évaluent que le dév. Seule l'étape 7 touche le test.
Résultats : resultados/*.csv et resultados/mejor_<grafo>.json
"""
import argparse
import json
import os
import sys

import pandas as pd

import evaluacion as ev
import grafos_lea as gl
import node2vec_nap_2 as n2

RES = os.path.join(n2.DIRECTORIO, "resultados_2")
os.makedirs(RES, exist_ok=True)

CLAVES_MODELO = {          # clé en ligne de commande -> nom dans les tableaux
    "node2vec": "node2vec",
    "pr": "PageRank",
    "cc": "Cercanía (reducido)",
    "rrf2": "RRF node2vec+PR",
    "rrf3": "RRF node2vec+CC+PR",
    "rrfg": "RRF PR+CC (sans node2vec)",
}
CFG_CLAVES = ["dimensions", "walk_length", "num_walks", "p", "q"]


# ------------------------------------------------------------------ utilitaires
def cargar():
    """Les trois graphes de Léa + le DataFrame complet de evaluacion (dév ET test, filtrés plus tard)."""
    grafos = gl.construye_grafos(n2.NAP_XLS)
    for nombre, g in grafos.items():
        print(f"Graphe {nombre} : {g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes")
    base = set(grafos["asoc"].nodes())
    if any(set(g.nodes()) != base for g in grafos.values()):
        print("ATTENTION : les trois graphes n'ont pas les mêmes nœuds")
    datos = ev.cargar_datos(n2.DIRECTORIO, grafos["asoc"])
    ev.resumen(datos)
    n_dev = int((datos["particion"] == "desarrollo").sum())
    n_test = int((datos["particion"] == "prueba").sum())
    print(f"\n>>> Développement : {n_dev} descriptions | Test : {n_test} descriptions\n")
    if n_dev == 0 or n_test == 0:
        sys.exit("Une des deux partitions est vide : vérifie corpus/, nuevas_desc.csv et evaluacion.PARTICION")
    return grafos, datos


def guardar(df, nombre):
    ruta = os.path.join(RES, nombre)
    df.to_csv(ruta, index=False)
    print(f"\n-> {ruta}")


def _ruta_mejor(grafo):
    return os.path.join(RES, f"mejor_{grafo}.json")


def guardar_mejor(nombre_grafo, **campos):
    ruta = _ruta_mejor(nombre_grafo)
    actual = json.load(open(ruta)) if os.path.exists(ruta) else {}
    actual.update(campos)
    json.dump(actual, open(ruta, "w"), indent=2)


def cargar_mejor(grafo):
    ruta = _ruta_mejor(grafo)
    if not os.path.exists(ruta):
        sys.exit(f"Il manque {ruta} : lance d'abord l'étape 2 (puis la 3) avec --grafo {grafo}")
    m = json.load(open(ruta))
    m.setdefault("modo", "suma")
    return m


def mejor_indice(tabla, criterio="MRR"):
    """Ligne avec le meilleur MRR (à égalité, la première)."""
    return tabla[criterio].idxmax()


def construir_modelos(grafos, datos, grafo):
    """Tous les modèles sur le MÊME graphe : node2vec (meilleure config), PageRank, proximité, fusions RRF."""
    mejor = cargar_mejor(grafo)
    G = grafos[grafo]
    wv = n2.entrena_node2vec(G, grafo, weight_key="sim", dimensions=64, walk_length=80, num_walks=20, p=1.0, q=1.0)
    idf = n2.idf_dev(datos) if mejor["modo"] == "idf" else None
    m_n2v = n2.crea_modelo(wv, mejor["modo"], idf)
    m_pr = gl.crea_modelo(gl.diccionario_pr, G, alpha=gl.ALPHA_PR)
    m_cc = gl.crea_modelo(gl.diccionario_cc, G)
    m_btc = gl.crea_modelo(gl.diccionario_btc, G)
    return {
        CLAVES_MODELO["node2vec"]: m_n2v,
        CLAVES_MODELO["pr"]: m_pr,
        CLAVES_MODELO["cc"]: m_cc,
        CLAVES_MODELO["rrf2"]: n2.modelo_rrf([m_n2v, m_pr]),
        CLAVES_MODELO["rrf3"]: n2.modelo_rrf([m_n2v, m_cc, m_pr]),
        CLAVES_MODELO["rrfg"]: n2.modelo_rrf([m_cc, m_pr]),
        "rff4" : n2.modelo_rrf([m_n2v, m_btc, m_pr]), 
        "node cc pr" : n2.modelo_rrf([m_n2v, m_cc, m_pr]),
    }

def construir_modelos2(grafos, datos, grafo):
    G = grafos[grafo]
    m_btc = gl.crea_modelo(gl.diccionario_btc, G)
    m_btc_r = gl.crea_modelo(gl.diccionario_btc_reducido, G)
    m_dc = gl.crea_modelo(gl.diccionario_dc, G)

    return {
        "btc": m_btc, 
        "btc_r": m_btc_r, 
        "m_dc" : m_dc,
    }

def construir_modelos_Lea(grafos) : 
    m_btc = gl.crea_modelo(gl.diccionario_btc, grafos)
    m_btc_r = gl.crea_modelo(gl.diccionario_btc_reducido, grafos)
    m_dc = gl.crea_modelo(gl.diccionario_dc, grafos)
    m_pr = gl.crea_modelo(gl.diccionario_pr, grafos, alpha=gl.ALPHA_PR)
    m_cc = gl.crea_modelo(gl.diccionario_cc, grafos)
    return {
        "btc": m_btc, 
        "btc_r": m_btc_r, 
        "dc" : m_dc, 
        "pr" : m_pr, 
        "cc" : m_cc      
    }



# ------------------------------------------------------------------ étapes
def etapa_0(grafos, datos, args) :
    filas = []
    for nombre_G, G in grafos.items() :

        modelos = construir_modelos_Lea(G)

        for nombre, fn in modelos.items():
            filas.append({"modelo": nombre, "grafo":nombre_G, **n2.evaluar_dev(fn, datos)})
            print(nombre, {k: round(v, 3) for k, v in filas[-1].items() if k in ("p@1", "p@3", "p@5", "MRR")})
            tabla = pd.DataFrame(filas)
            print("\n" + tabla.to_string())
    guardar(tabla, f"etapa4_rrf_dev_{args.grafo}.csv")


def etapa_1(grafos, datos, args):
    """Même configuration (CONFIG_REF) sur les trois graphes : quelle force (asoc, frec, tiempo) pour node2vec ?
    Compare avec ton ancien résultat (poids 1/frecuencia : p@1 4,5 % ... p@10 12,7 %)."""
    tabla = n2.comparar_grafos(grafos, datos)
    print("\n" + tabla.to_string())
    print(f"\nMeilleur graphe selon le MRR (dév) : {tabla.loc[mejor_indice(tabla), 'grafo']}")
    guardar(tabla, "etapa1_grafos.csv")


def etapa_2(grafos, datos, args):
    """Une variable à la fois (dimension, p, q, longueur, nombre de marches) sur le graphe choisi."""
    tabla = n2.experimentos(grafos[args.grafo], datos, args.grafo)
    print("\n" + tabla.to_string())
    i = mejor_indice(tabla)
    cfg = {k: (float(tabla.loc[i, k]) if k in ("p", "q") else int(tabla.loc[i, k])) for k in CFG_CLAVES}
    print(f"\nMeilleure config selon le MRR (dév) : {tabla.loc[i, 'cambio']} -> {cfg}")
    guardar(tabla, f"etapa2_configuraciones_{args.grafo}.csv")
    guardar_mejor(args.grafo, cfg=cfg)


def etapa_3(grafos, datos, args):
    """Somme / moyenne / IDF avec la meilleure config de l'étape 2."""
    wv = n2.entrena_node2vec(grafos[args.grafo], args.grafo, weight_key="sim", dimensions=64, walk_length=80, num_walks=20, p=1.0, q=1.0)
    tabla = n2.comparar_modos(wv, datos)
    print("\n" + tabla.to_string())
    modo = tabla.loc[mejor_indice(tabla), "modo"]
    print(f"\nMeilleur mode selon le MRR (dév) : {modo}")
    guardar(tabla, f"etapa3_modos_{args.grafo}.csv")
    guardar_mejor(args.grafo, modo=modo)


def etapa_4(grafos, datos, args):
    """RRF sur le DÉV : node2vec seul, PageRank seul, proximité seule, et les fusions."""
    modelos = construir_modelos(grafos, datos, args.grafo)
    filas = []
    for nombre, fn in modelos.items():
        filas.append({"modelo": nombre, **n2.evaluar_dev(fn, datos)})
        print(nombre, {k: round(v, 3) for k, v in filas[-1].items() if k in ("p@1", "p@3", "p@5", "MRR")})
    tabla = pd.DataFrame(filas)
    print("\n" + tabla.to_string())
    guardar(tabla, f"etapa4_rrf_dev_{args.grafo}.csv")


def _elegir(modelos, clave):
    nombre = CLAVES_MODELO[clave]
    return nombre, modelos[nombre]


def etapa_5(grafos, datos, args):
    """Démo en direct : tape une description, obtiens le top 5."""
    modelos = construir_modelos(grafos, datos, args.grafo)
    nombre, modelo = _elegir(modelos, args.modelo)
    print(f"Modèle de la démo : {nombre}")
    (n2.demo_gradio if args.gradio else n2.demo)(modelo)


def etapa_6(grafos, datos, args):
    """Descriptions du DÉV où le modèle n'a pas le bon mot dans le top 5 (à classer par cause à la main)."""
    modelos = construir_modelos(grafos, datos, args.grafo)
    nombre, modelo = _elegir(modelos, args.modelo)
    tabla = ev.fallos(modelo, datos, particion="desarrollo", k=5)
    print(f"{nombre} : {len(tabla)} échecs sur le dév")
    print(tabla.head(15).to_string())
    guardar(tabla, f"fallos_{args.modelo}_{args.grafo}.csv")
    print("Remplis la colonne causa_manual (lematización, fuera del vocabulario, polisemia, nodo muy conectado, ambigua).")

    #Pour node2vec seulement
    mejor = cargar_mejor(args.grafo)
    G = grafos[args.grafo]

    wv = n2.entrena_node2vec(G, args.grafo, weight_key="sim", dimensions=64, walk_length=80, num_walks=20, p=1.0, q=1.0)
    idf = n2.idf_dev(datos) if mejor["modo"] == "idf" else None
    m_n2v = n2.crea_modelo(wv, mejor["modo"], idf) 
    tabla = ev.fallos(m_n2v, datos, particion="desarrollo", k=5)
    print(f"Node2vec : {len(tabla)} échecs sur le dév")
    print(tabla.head(15).to_string())
    guardar(tabla, f"fallos_Node2vec_{args.grafo}.csv")
    print("Remplis la colonne causa_manual (lematización, fuera del vocabulario, polisemia, nodo muy conectado, ambigua).")

    #Pour PageRank seulement
    m_pr = gl.crea_modelo(gl.diccionario_pr, G, alpha=gl.ALPHA_PR)
    tabla = ev.fallos(m_pr, datos, particion="desarrollo", k=5)
    print(f"PR : {len(tabla)} échecs sur le dév")
    print(tabla.head(15).to_string())
    guardar(tabla, f"fallos_PR_{args.grafo}.csv")
    print("Remplis la colonne causa_manual (lematización, fuera del vocabulario, polisemia, nodo muy conectado, ambigua).")

def etapa_7(grafos, datos, args):
    """TABLEAU FINAL dév + test (evaluacion.tabla). À lancer une seule fois : il évalue le TEST."""
    print("ATTENTION : cette étape évalue le jeu de TEST. Elle ne doit être lancée qu'une fois,")
    print("avec la config et le graphe définitifs, choisis sur le dév.")
    if input("Écris 'si' pour continuer : ").strip().lower() != "si":
        sys.exit("Annulé.")
    modelos = construir_modelos(grafos, datos, args.grafo)
    tabla = ev.tabla(modelos, datos)
    print("\n" + tabla.round(3).to_string())
    tabla.to_csv(os.path.join(RES, f"etapa7_tabla_final_{args.grafo}.csv"))
    print(f"\n-> {os.path.join(RES, f'etapa7_tabla_final_{args.grafo}.csv')}")

def etapa_8(grafos, datos, args) : 
    modelos = construir_modelos2(grafos, datos, args.grafo)
    tabla = ev.tabla(modelos, datos)
    print("\n" + tabla.round(3).to_string())
    
    # Chemin vers le fichier CSV créé à l'étape 7
    chemin_fichier = os.path.join(RES, f"etapa7_tabla_final_{args.grafo}.csv")
    
    # Ajout des nouvelles lignes à la fin du fichier existant
    # header=False évite de réécrire les noms de colonnes
    # index=True (par défaut) conserve l'index si l'étape 7 l'a gardé
    tabla.to_csv(chemin_fichier, mode='a', header=False)
    
    print(f"\n-> Nouvelles lignes ajoutées avec succès dans : {chemin_fichier}")


ETAPAS = {"0": etapa_0, "1": etapa_1, "2": etapa_2, "3": etapa_3, "4": etapa_4, "5": etapa_5, "6": etapa_6, "7": etapa_7, "8" : etapa_8}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=ETAPAS)
    ap.add_argument("--grafo", choices=["asoc", "frec", "tiempo"], default="asoc",
                    help="graphe (force) utilisé par node2vec, PageRank et proximité (défaut : asoc)")
    ap.add_argument("--modelo", choices=CLAVES_MODELO, default="rrf2",
                    help="modèle pour les étapes 5 et 6 (défaut : rrf2 = RRF node2vec+PR)")
    ap.add_argument("--gradio", action="store_true", help="étape 5 : interface web (pip install gradio)")
    args = ap.parse_args()
    grafos, datos = cargar()
    ETAPAS[args.etapa](grafos, datos, args)

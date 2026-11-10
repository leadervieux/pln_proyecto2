"""
tabla_final.py - UNE seule évaluation pour tous les modèles, avec evaluacion.py.

    python tabla_final.py            # tout calculer (long : prévoir 30 à 60 min la première fois)
    MUESTRA=30 python tabla_final.py # essai rapide sur 30 descriptions par partition (écrit resultados_muestra.csv)

Dossier : NAP.xls, corpus/, nuevas_desc.csv, evaluacion.py, modelos_lea.py (et les modèles de Max).
Reprise automatique : chaque modèle terminé est enregistré dans resultados_finales.csv ; si le programme
s'arrête (Colab se déconnecte), on le relance et il continue là où il s'était arrêté.
"""
import os
import sys

# Résultats REPRODUCTIBLES : en cas d'égalité de score, l'ordre des mots dépend du « hash » des chaînes de Python, qui
# change à chaque exécution (cela fait varier Grado et Cercanía d'environ ±0,01). On fixe donc PYTHONHASHSEED=0 et
# on relance le programme une fois.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    import subprocess
    sys.exit(subprocess.call([sys.executable] + sys.argv, env={**os.environ, "PYTHONHASHSEED": "0"}))

import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
import evaluacion as ev          # noqa: E402
import modelos_lea as ml         # noqa: E402

MUESTRA = int(os.environ.get("MUESTRA", "0"))
SOLO = os.environ.get("SOLO", "")      # ex. SOLO="Cercanía,Grado" : ne lance que les modèles dont le nom contient un de ces mots
ARCHIVO = os.path.join(CARPETA, "resultados_muestra.csv" if MUESTRA else "resultados_finales.csv")


def rrf(*modelos, k=60, topn=100):
    """Fusion de rangs : score(mot) = somme de 1 / (k + rang) sur les listes des modèles donnés."""
    def fusion(definicion):
        puntaje = {}
        for m in modelos:
            for r, w in enumerate(m(definicion)[:topn], 1):
                puntaje[w] = puntaje.get(w, 0.0) + 1.0 / (k + r)
        return [w for w, _ in sorted(puntaje.items(), key=lambda x: (-x[1], x[0]))][:topn]
    return fusion


if __name__ == "__main__":
    GrafoFrec, GrafoTiempo, GrafoAsoc = ml.Construye_Grafo(os.path.join(CARPETA, "NAP.xls"))
    print("Graphe :", GrafoAsoc.number_of_nodes(), "nœuds,", GrafoAsoc.number_of_edges(), "arêtes")
    ev.fijar_grafo(GrafoAsoc)
    datos = ev.cargar_datos(CARPETA)
    if MUESTRA:
        datos = pd.concat([datos[datos.particion == "desarrollo"].head(MUESTRA), datos[datos.particion == "prueba"].head(MUESTRA)])
    ev.resumen(datos)

    modelos = ml.modelos(GrafoFrec, GrafoTiempo, GrafoAsoc)

    # ------------------------------------------------------------------ AJOUTEZ ICI LES MODÈLES DE MAX
    # Chaque modèle est une fonction  f(definicion: str) -> liste de mots candidats (du meilleur au moins bon).
    # modelos["node2vec A (128d, p=1, q=1)"] = fonction_node2vec_A
    # modelos["RRF (cercanía + node2vec)"] = rrf(modelos["Cercanía | reducido | asoc"], fonction_node2vec_A)
    # modelos["RRF à trois (+ PageRank)"] = rrf(modelos["Cercanía | reducido | asoc"], fonction_node2vec_A,
    #                                           modelos["PageRank (alpha 0.5) | completo | asoc"])
    # ------------------------------------------------------------------------------------------------

    previo = pd.read_csv(ARCHIVO, index_col=0) if os.path.exists(ARCHIVO) else pd.DataFrame()
    filas = [previo] if len(previo) else []
    for nombre, fn in modelos.items():
        if SOLO and not any(s.strip() in nombre for s in SOLO.split(",")):
            continue
        if len(previo) and nombre in previo.index:
            print("déjà calculé :", nombre)
            continue
        print("calcul :", nombre, flush=True)
        fila = ev.tabla({nombre: fn}, datos)
        filas.append(fila)
        pd.concat(filas).to_csv(ARCHIVO)          # sauvegarde après chaque modèle
    tabla = pd.concat(filas).round(3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n", tabla.to_string())
    print("\nFichier :", ARCHIVO)

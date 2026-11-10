"""
probar_evaluacion.py - Verifica que NAP, corpus, nuevas_desc.csv y evaluacion.py funcionan juntos.

Poner en la MISMA carpeta que NAP.xls, corpus/, nuevas_desc.csv y evaluacion.py, y ejecutar:
    python probar_evaluacion.py

Dependencias:  pip install pandas networkx xlrd spacy nltk
               python -m spacy download es_core_news_sm
Cada prueba imprime OK o FALLO; al final hay un resumen. Tarda 1-2 minutos.
"""
import os
import sys

import networkx as nx
import xlrd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
resultados = []


def prueba(descripcion, condicion, detalle=""):
    resultados.append(bool(condicion))
    print(("OK     " if condicion else "FALLO  ") + descripcion + (f"  ->  {detalle}" if detalle else ""))
    return bool(condicion)


print("== 1. Archivos ==")
for nombre in ["NAP.xls", "corpus", "nuevas_desc.csv", "evaluacion.py"]:
    prueba(f"existe {nombre}", os.path.exists(os.path.join(CARPETA, nombre)))
n_txt = len([f for f in os.listdir(os.path.join(CARPETA, "corpus")) if f.endswith(".txt")]) if os.path.isdir(os.path.join(CARPETA, "corpus")) else 0
prueba("corpus/ tiene 56 archivos .txt", n_txt == 56, f"hay {n_txt}")
if not all(resultados):
    sys.exit("Faltan archivos: corrige esto antes de seguir.")

import evaluacion as ev  # noqa: E402

print("\n== 1b. El módulo está completo (un copy-paste cortado deja funciones sin definir) ==")
for nombre in ["fijar_grafo", "semillas", "cargar_datos", "resumen", "evaluar", "tabla", "fallos", "menciona_objetivo"]:
    prueba(f"evaluacion.{nombre} existe", hasattr(ev, nombre))
if not all(resultados):
    sys.exit("evaluacion.py está incompleto o cortado: vuelve a copiarlo entero.")


print("\n== 2. Grafo NAP ==")
hoja = xlrd.open_workbook(os.path.join(CARPETA, "NAP.xls")).sheet_by_index(0)
G = nx.Graph()
est = ""
for f in range(hoja.nrows - 1):
    c = hoja.cell(f, 0).value
    if c == "--PALABRAS--":
        continue
    elif c == "======":
        est = str(hoja.cell(f + 1, 4).value).strip().lower()
    elif c in ("", "=") or hoja.cell(f, 2).value == "":
        continue
    else:
        lema = str(hoja.cell(f, 4).value).strip().lower()
        if lema in ("", "=") or est == "" or lema == est:
            continue
        asoc = float(hoja.cell(f, 3).value)
        G.add_edge(est, lema, sim=asoc, dist=max(100 - asoc, 0.01))
prueba("el grafo tiene unos 4,2xx nodos (esperado 4,200 a 4,260)", 4200 <= G.number_of_nodes() <= 4260, f"{G.number_of_nodes()} nodos")
prueba("el grafo tiene unas 12,8xx aristas (esperado 12,700 a 12,900)", 12700 <= G.number_of_edges() <= 12900, f"{G.number_of_edges()} aristas")
prueba("el grafo es una sola componente conexa", nx.number_connected_components(G) == 1)
prueba("'perro' y 'abeja' son nodos", "perro" in G and "abeja" in G)

print("\n== 3. Datos ==")
ev.fijar_grafo(G)
datos = ev.cargar_datos(CARPETA)
r = ev.resumen(datos)
prueba("hay descripciones del curso (esperado 276)", (datos["fuente"] == "curso").sum() == 276, f"{(datos['fuente'] == 'curso').sum()}")
prueba("hay descripciones nuevas (esperado >= 150)", (datos["fuente"] == "equipo").sum() >= 150, f"{(datos['fuente'] == 'equipo').sum()}")
esperados_fuera = [] if ev.USAR_ALIAS else ["aspiradora", "playera"]
prueba(f"conceptos fuera del grafo = {esperados_fuera} (como en la clase; 'aspiradora' y 'playera' no son nodos de NAP)",
       r["conceptos fuera del grafo"] == esperados_fuera, str(r["conceptos fuera del grafo"]))
prueba("los conceptos NUEVOS existen todos como nodo del grafo", datos.loc[datos["fuente"] == "equipo", "objetivo_en_grafo"].all())
prueba("el concepto de Papas.txt es 'papa' (el título del archivo, como en la clase)", "papa" in set(datos["concepto"]) and "papas" not in set(datos["concepto"]))
prueba("partición: corpus del curso = desarrollo, descripciones del equipo = prueba",
       ev.PARTICION != "fuente" or (set(datos.loc[datos["fuente"] == "curso", "particion"]) == {"desarrollo"}
                                    and set(datos.loc[datos["fuente"] == "equipo", "particion"]) == {"prueba"}))
prueba("ningún concepto está en desarrollo y en prueba a la vez",
       datos.groupby("concepto")["particion"].nunique().max() == 1)
n_menc = int(datos.loc[(datos["particion"] == "prueba") & datos["menciona_objetivo"]].shape[0])
print(f"INFO   descripciones de prueba que mencionan la palabra objetivo: {n_menc} (con el nuevas_desc.csv original son 6, como dice el README)")
prueba("las dos particiones tienen descripciones", set(datos["particion"]) == {"desarrollo", "prueba"})
prueba("cobertura de lemas >= 90 %", r["cobertura de lemas (% con >=1 lema en el grafo)"] >= 90,
       f"{r['cobertura de lemas (% con >=1 lema en el grafo)']} %")
prueba("la línea del título no entró como descripción (ningún concepto del curso con 1 sola palabra igual al concepto)",
       not ((datos["fuente"] == "curso") & (datos["definicion"].str.lower().str.strip() == datos["concepto"])).any())
prueba("semillas() funciona", "ropa" in ev.semillas("Artefacto donde metes a lavar tu ropa"), str(ev.semillas("Artefacto donde metes a lavar tu ropa")))
prueba("menciona_objetivo detecta 'fruta de color naranja'", ev.menciona_objetivo("fruta de color naranja", "naranja"))
prueba("menciona_objetivo NO se activa en 'animal que ladra'", not ev.menciona_objetivo("animal que ladra", "perro"))
datos2 = ev.cargar_datos(CARPETA)
prueba("al recargar, la partición es idéntica (está congelada)", (datos2["particion"] == datos["particion"]).all())

print("\n== 4. Que evaluar() mida bien ==")
# Solo descripciones con texto único (si dos conceptos comparten exactamente el mismo texto, el "oráculo" no puede distinguirlos).
unicas = datos[~datos["definicion"].duplicated(keep=False)]
tabla_obj = {d: o for d, o in zip(unicas["definicion"], unicas["objetivo"])}
oraculo = lambda d: [tabla_obj[d], "xx1", "xx2"]       # siempre acierta en el 1.º lugar
vacio = lambda d: []                                    # nunca devuelve nada
malo = lambda d: ["xx1", "xx2", "xx3", "xx4", "xx5", "xx6"]   # nunca acierta
# Nota: si una descripción contiene la palabra objetivo, evaluar() la quita de los candidatos (regla de conceptos()
# de la clase), así que ni un modelo perfecto puede acertar ahí. Por eso estas pruebas excluyen esas descripciones.
m = ev.evaluar(oraculo, unicas, particion=None, excluir_mencion=True)
prueba("modelo perfecto -> p@1 = 1 y MRR = 1", m["p@1"] == 1 and abs(m["MRR"] - 1) < 1e-9, f"p@1={m['p@1']:.3f} MRR={m['MRR']:.3f}")
m = ev.evaluar(vacio, unicas, particion=None)
prueba("modelo vacío -> todo 0 y cobertura_modelo = 0", m["p@5"] == 0 and m["MRR"] == 0 and m["cobertura_modelo"] == 0)
m = ev.evaluar(malo, unicas, particion=None)
prueba("modelo que nunca acierta -> p@5 = 0 y cobertura_modelo = 1", m["p@5"] == 0 and m["cobertura_modelo"] == 1)
segundo = lambda d: ["xx1", tabla_obj[d], "xx2"]
m = ev.evaluar(segundo, unicas, particion=None, excluir_mencion=True)
prueba("acierto siempre en 2.º lugar -> p@1 = 0, p@3 = 1, MRR = 0.5",
       m["p@1"] == 0 and m["p@3"] == 1 and abs(m["MRR"] - 0.5) < 1e-9, f"p@1={m['p@1']:.2f} p@3={m['p@3']:.2f} MRR={m['MRR']:.2f}")

print("\n== 5. Modelo de ejemplo (cercanía en el grafo reducido, ponderación asociación) ==")


def cercania(definicion):
    sem = ev.semillas(definicion)
    if not sem:
        return []
    nodos = set(sem)
    for s in sem:
        nodos.update(G[s])
    c = nx.closeness_centrality(G.subgraph(nodos), distance="dist")
    return [w for w, v in sorted(c.items(), key=lambda x: (-x[1], x[0])) if v > 0]   # desempate fijo: por palabra


m = ev.evaluar(cercania, datos, particion="prueba")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
prueba("la cercanía acierta algo (p@5 de prueba > 0.30; con el corpus del curso suele dar 0.5 a 0.6)", m["p@5"] > 0.30, f"p@5={m['p@5']:.3f}")
prueba("p@1 <= p@3 <= p@5", m["p@1"] <= m["p@3"] <= m["p@5"])
f = ev.fallos(cercania, datos, particion="prueba")
prueba("fallos() devuelve las descripciones donde falla", 0 < len(f) < m["n"], f"{len(f)} fallos de {m['n']}")

print(f"\n== Resumen: {sum(resultados)} de {len(resultados)} pruebas OK ==")
if not all(resultados):
    print("Mira las líneas que dicen FALLO. Si no entiendes una, copia el mensaje y pregúntame.")

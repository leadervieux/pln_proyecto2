import evaluacion as ev
import networkx as nx
import pandas as pd

print("Lecture de NAP.csv...")
df_nap = pd.read_csv("NAP.csv")
print("Dimensions de NAP.csv :", df_nap.shape)

print("Construction du graphe à partir de NAP.csv...")
G = nx.Graph()
current_cue = None

for _, row in df_nap.iterrows():
    col0 = str(row.iloc[0]).strip()
    col1 = str(row.iloc[1]).strip()
    
    # Détection deq mots cibles
    if col1.lower() in ["nan", "", "frecuencia", "--palabras--"]:
        if col0.lower() not in ["nan", "", "======", "--palabras--"]:
            current_cue = col0.lower()
    else:
        if current_cue and col0.lower() not in ["nan", "", "*"]:
            resp = col0.lower()
            try:
                frec = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 1.0
                tiempo = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 1.0
                fa = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0.1
                
                G.add_edge(current_cue, resp, frec=frec, tiempo=tiempo, sim=fa)
            except ValueError:
                pass

print(f"Graphe construit avec succès : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")

#Fixer le graphe réel dans evaluacion.py
ev.fijar_grafo(G)

#Charger les données (corpus + nuevas_desc.csv)
datos = ev.cargar_datos(".")
print(f"\nDonnées chargées avec succès : {len(datos)} lignes au total.")

# Afficher le résumé stat
stats = ev.resumen(datos)
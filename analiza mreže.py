# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:40:47 2021

@author: Nola Čumlievski
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms import community
import matplotlib.colors as mcolors

# funkcija za učitavanje naslova i podataka
def get_data(file):
    #upotreba context managera za učitavanje podataka"
    with open(file, "r", encoding = "utf8") as line:
        lines = line.read().split("\n") #transformacija file-a u string
        data = [line.split(",") for line in lines] #separacija file-a prema zarezu
        header = data[0] #header je prvi red
        data = data[1:] #podaci su svi ostali redovi
    return header, data

node_header, node_data = get_data("C:/Users/38591/Desktop/Faks/UZ/nodes.csv")
edge_header, edge_data = get_data("C:/Users/38591/Desktop/Faks/UZ/edges.csv")

edge_header = [y.strip('"') for y in edge_header]
node_header = [y.strip('"') for y in node_header]
for i in range(len(node_data)):
    node_data[i] = [y.strip('"') for y in node_data[i]]
    
for i in range(len(edge_data)):
    edge_data[i] = [y.strip('"') for y in edge_data[i]]
    
# kreiranje graph objekta
graf = nx.Graph()

# učitavanje čvorova u graf
for node in node_data:
    graf.add_node(int(node[2]), name = node[1])

print(graf.nodes.data())

for edge in edge_data:
    graf.add_edge(int(edge[0]), int(edge[1]))

print(graf.edges.data())

# broj čvorova
graf.number_of_nodes()

# broj veza
graf.number_of_edges()

#prosječni broj veza
susjedi = dict(graf.degree)
prosjek = sum(susjedi.values()) / graf.number_of_edges()

# broj komponenti

nx.number_connected_components(graf)

# prosječni najkraći put
nx.average_shortest_path_length(graf)

#dijametar
nx.diameter(graf)

# eccentricity
eccentric = nx.eccentricity(graf)
polja = np.fromiter(eccentric.values(), dtype=float)
avg_e = np.mean(polja)

# global efficiency 
nx.global_efficiency(graf)

# globalni koeficijent grupiranja
g_k_g = nx.clustering(graf)
a_g_k_g = np.mean(np.fromiter(g_k_g.values(), dtype=float))

# prosječni koeficijent grupiranja
nx.average_clustering(graf)

# asortativnost
nx.degree_assortativity_coefficient(graf)

#  dijagram distribucije stupnjeva
def graf_distribucije_stupnjeva(graf):
    stupnjevi = [graf.degree(n) for n in graf.nodes()]
    plt.style.use("seaborn")
    plt.hist(stupnjevi, color = "#ff4d94")
    plt.xlabel("Stupanj")
    plt.ylabel("Broj čvorova")
    plt.show()

graf_distribucije_stupnjeva(graf)

# degree centrality
centralnost_stupnjeva = nx.degree_centrality(graf)
centralnost_top10 = sorted(centralnost_stupnjeva, key=centralnost_stupnjeva.get, reverse=True)[:10]

graf.nodes[1073]['name']

# closeness
closeness = nx.closeness_centrality(graf)
closeness_top10 = sorted(closeness, key=closeness.get, reverse=True)[:10]

# betweenness
between = nx.betweenness_centrality(graf)
between_top10 = sorted(between, key=between.get, reverse=True)[:10]

# prosječna centralnost blizine
a_c_c = np.mean(np.fromiter(closeness.values(), dtype=float))

# prosječna međupoloženost
a_b_c = np.mean(np.fromiter(between.values(), dtype=float))

# podjela u zajednice
communities = community.asyn_fluidc(graf, 10)
communities = list(communities)
# modularnost
modularnost = community.modularity(graf, communities)

#vizualizacija
plt.figure(3,figsize=(200,200)) 
pos = nx.spring_layout(graf, seed=1)
sc = nx.draw_networkx_nodes(G=graf, pos = pos, nodelist = graf.nodes(), alpha=0.9, node_size = 10, node_color="#1a1aff")
nx.draw_networkx_edges(G = graf, pos = pos, edge_color='#818a8c', alpha=0.6, width=1)
plt.show()
# vizualizacija na centralnost čvorova
def draw(G, pos, node_size, measures, measure_name):
    # grafovi prema centralnosti
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    edges = nx.draw_networkx_edges(G, pos, edge_color='#818a8c', alpha=0.6, width=1)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

plt.figure(3,figsize=(200,200)) 
node_sizes = []
for n in centralnost_stupnjeva.values():
        node_sizes.append( 15000 * n )
draw(graf, pos, node_sizes, dict(graf.degree), 'Centralnost stupnja')


# closeness
plt.figure(3,figsize=(200,200))
node_sizes = []
for n in closeness.values():
        node_sizes.append( 200 * n )
draw(graf, pos, node_sizes, nx.closeness_centrality(graf), 'Centralnost blizine')

# betweenness
plt.figure(3,figsize=(200,200))
node_sizes = []
for n in between.values():
        node_sizes.append( 10000 * n )
draw(graf, pos, node_sizes, nx.betweenness_centrality(graf), 'Centralnost međupoloženosti')

# graf zajednica
color_map = []
for node in graf:
    if node in communities[0]:
        color_map.append('#0000ff')
    elif node in communities[1]:
        color_map.append("#00b300")
    elif node in communities[2]:
        color_map.append("#660066")
    elif node in communities[3]:
        color_map.append("#ff1ac6")
    elif node in communities[4]:
        color_map.append("#e6e600")
    elif node in communities[5]:
        color_map.append("#ff9900")
    elif node in communities[6]:
        color_map.append("#ff0000")
    elif node in communities[7]:
        color_map.append("#664400")
    elif node in communities[8]:
        color_map.append("#5c5c3d")
    else: 
        color_map.append('#b3ffb3') 

plt.figure(3,figsize=(200,200)) 
sc = nx.draw_networkx_nodes(G=graf, pos = pos, nodelist = graf.nodes(), alpha=0.9, node_size = 10, node_color=color_map)
nx.draw_networkx_edges(G = graf, pos = pos, edge_color='#818a8c', alpha=0.6, width=1)
plt.show()

# broj čvorova u zajednici

zajednica = graf.subgraph(communities[0])
zajednica.number_of_nodes()
zajednica.number_of_edges()

# druga zajednica
zajednica = graf.subgraph(communities[1])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[2])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[3])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[4])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[5])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[6])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[7])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[8])
zajednica.number_of_nodes()
zajednica.number_of_edges()

zajednica = graf.subgraph(communities[9])
zajednica.number_of_nodes()
zajednica.number_of_edges()

# dodatak
#Katz centrality
katz = nx.katz_centrality_numpy(graf)
katz_top10 = sorted(katz, key=katz.get, reverse=True)[:10]

# information centrality
inf = nx.current_flow_closeness_centrality(graf)
inf_top10 = sorted(inf, key=inf.get, reverse=True)[:10]

# load centrality
load = nx.load_centrality(graf)
load_top10 = sorted(load, key=load.get, reverse=True)[:10]

# harmonic centrality
harm = nx.harmonic_centrality(graf)
harmonic_top10 = sorted(harm, key=harm.get, reverse=True)[:10]

# eigenventor centrality
eigen = nx.eigenvector_centrality(graf)
eigen_top10 = sorted(eigen, key=eigen.get, reverse=True)[:10]

central = list(centralnost_stupnjeva.values())
bet = list(between.values())
close = list(closeness.values())
katzc = list(katz.values())
info = list(inf.values())
loadc = list(load.values())
harmo = list(preloc.values())
eige = list(eigen.values())

data = {"degree":central, "betweenness":bet, "closeness":close, "katz": katzc, "information":info, "load": loadc, "harmonic": prelo, "eigenvector": eige}

df = pd.DataFrame.from_dict(data,orient='index').transpose()
res = df.corr(method = "spearman")



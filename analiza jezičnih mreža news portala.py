# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:24:24 2020

@author: Nola Čumlievski
"""
import pandas as pd
import numpy as np
from datetime import datetime
import string
import seaborn as sns
import matplotlib.pyplot as plt
import re
import collections
import networkx as nx
from networkx.algorithms import community
import matplotlib.colors as mcolors
from random import sample

#učitavanje csv datoteke
podaci = pd.read_csv("C:/Users/38591/AppData/Local/Programs/Python/Python36/baranjainfopravi.csv", sep=";")

#provjera oblika data frame-a
podaci.shape
podaci.columns

#pregled tipova podataka
podaci.info()

#izbacivanje nepotrebnih varijabli
podaci = podaci.drop(columns = ["Broj", "Link", "Autor", "Kategorija"])
podaci.shape

podaci = podaci.rename(columns={'Datum objave': 'Datum'})

#konvertanje svih tekstova u lower case zbog pronalaska određenih riječi
podaci['Naslov'] = podaci['Naslov'].str.lower()
podaci['Tekst'] = podaci['Tekst'].str.lower()
podaci.head(3)

#ukupan broj korona objava
rijeci_korona = ["koron\w+", "koronavirus\w*", "virus\w*", "capak", "epidemij\w+", "covid\w*", "bolnic\w+", "zaražen\w*", "novooboljel\w+","samoizolacij\w+", "karanten\w+", "lockdown\w*", "novozaražen\w*", "testiranj\w+", "pozitivn\w+", "mjer\w+", "stožer\w*", "epidemiološk\w+", "zaraza", "novopozitivn\w+", "cjepiv\w+", "oboljel\w+", "respirator\w*", "propusnic\w+", "dezinficir\w+", "socijaln\w+\sdistanc\w+", "alemka", "beroš", "oboljeli", "preboljel\w+", "Markotić", "Božinović"]
ukupno_korona = podaci[(podaci["Naslov"].str.contains('|'.join(rijeci_korona))) & (podaci["Tekst"].str.contains('|'.join(rijeci_korona)))]
ukupno_korona.shape
ukupno_korona.head()
podaci = ukupno_korona

#pretvorba datuma u datetime objekt za lakšu kasniju manipulaciju, izbacivanje NaN vrijednosti
podaci["Datum"] = pd.to_datetime(podaci["Datum"], format="%Y-%m-%d", errors='coerce')

podaci.info()

# eliminacija beskorisnih redova, članci s datumom većim od 25.8.2020
podaci = podaci[podaci.Datum < "2020-08-26"]

# grupiranje stupaca
podaci["Text"] = podaci["Naslov"] + podaci ["Tekst"]

# uklananje ekstra stupaca
podaci = podaci.drop(columns = ["Naslov", "Tekst"])

# čišćenje teksta
#zaustavne
zaustavne = ["a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bismo", "biste", "biti", "bumo", "da", "do", "duž", "ga", "hoće", "hoćemo", "hoćete", "hoćeš", "hoću", "i", "iako", "ih", "ili", "iz", "ja", "je", "jedna", "jedne", "jedno", "jer", "jesam", "jesi", "jesmo", "jest", "jeste", "jesu", "jim", "joj", "još", "ju", "kada", "kako", "kao", "koja", "koje", "koji", "kojima", "koju", "kroz", "li", "me", "mene", "meni", "mi", "mimo", "moj", "moja", "moje", "mu", "na", "nad", "nakon", "nam", "nama", "nas", "naš", "naša", "naše", "našeg", "ne", "nego", "neka", "neki", "nekog", "neku", "nema", "netko", "neće", "nećemo", "nećete", "nećeš", "neću", "nešto", "ni", "nije", "nikoga", "nikoje", "nikoju", "nisam", "nisi", "nismo", "niste", "nisu", "njega", "njegov", "njegova", "njegovo", "njemu", "njezin", "njezina", "njezino", "njih", "njihov", "njihova", "njihovo", "njim", "njima", "njoj", "nju", "no", "o", "od", "odmah", "on", "ona", "oni", "ono", "ovo", "ova", "pa", "pak", "po", "pod", "pored", "prije", "s", "sa", "sam", "samo", "se", "sebe", "sebi", "si", "smo", "ste", "su", "sve", "svi", "svog", "svoj", "svoja", "svoje", "svom", "ta", "tada", "taj", "tako", "te", "tebe", "tebi", "ti", "to", "toj", "tome", "tu", "tvoj", "tvoja", "tvoje", "u", "uz", "vam", "vama", "vas", "vaš", "vaša", "vaše", "već", "vi", "vrlo", "za", "zar", "će", "ćemo", "ćete", "ćeš", "ću", "što"]
pattern = r'\b(?:{})\b'.format('|'.join(zaustavne))
podaci['Text'] = podaci["Text"].str.replace(pattern, '')

# zamjenice, veznici i slično
rijeci = ["ovoliko", "toliko", "onoliko", "nekoliko", "zadnja", "prva", "druga", "iduća", "malo", "premalo", "više", "previše", "prekoviše", "najviše", "ponajviše", "manje", "najmanje", "ponajmanje", "dosta", "odveć", "opet", "još", "sasvim", "potpuno", "previše", "odsad", "otad", "oduvijek", "odavna", "danas", "večeras", "noćas", "jučer", "sinoć", "preksinoć", "sutra", "preksutra", "ljetos", "proljetos", "jesenas", "zimus", "proljeti", "ljeti", "jeseni", "zimi", "nadlani", "preklani", "obdan", "obnoć", "odmah", "smjesta", "sada", "tada", "onda", "ikada", "bilo", "kada", "nikada", "nekada", "ponekad", "katkad", "uvijek", "svagda", "često", "rijetko", "rano", "kasno", "prije", "poslije", "potom", "nedavno", "skoro", "uskoro", "napokon", "dosad", "dotad", "dogodine", "ovdje", "tu", "ondje", "negdje", "igdje", "nigdje", "onegdje", "gore", "dolje", "unutra", "vani", "ovamo", "onamo", "tamo", "nekamo", "nikamo", "ikamo", "naprijed", "natrag", "ovuda", "onuda", "tuda", "nikuda", "odavde", "otud", "odatle", "odonud", "niotkuda", "odozgo", "odozdo", "odostraga", "izdaleka", "izvana", "izbliza", "donekle", "onako", "ovako", "tako", "slučajno", "zato", "stoga", "uzalud", "uzaman", "utaman", "ovoliko", "toliko", "onoliko", "nekoliko", "malo", "premalo", "više", "previše", "prekoviše", "najviše", "ponajviše", "manje", "najmanje", "ponajmanje", "dosta", "odveć", "opet", "još", "sasvim", "potpuno", "previše", "ah", "oh", "hehe", "he", "hura", "jao", "joj", "oho", "uh", "uf", "ijuju", "haj", "eh", "ehe", "i", "pa", "te", "ni", "niti", "ili", "samo", "samo što", "jedino", "jedino što", "tek", "tek što", "dakle", "zato", "stoga", "ali", "nego", "no", "već", "da", "te", "kako", "čega", "neka", "jer", "ka", "prema", "napram"]
druga = ["nadomak", "nadohvat", "posljednjeg", "godine", "odnosno", "trenutni","trenutnog","trenutnih","imamo","imati","donjeg", "jedan", "prvog", "drugog", "dva", "tri", "četiri", "pet", "šest", "sedam", "osam", "devet", "posljednja", "deset", "broj", "godine", "trenutno", "gornjeg", "dana", "nasuprot", "usuprot", "usprkos", "unatoč", "protiv", "kroz", "niz", "uz", "na", "po", "mimo", "među", "nad", "pod", "pred", "za", "od", "do", "iz", "ispred", "iza", "izvan", "van", "unutar", "iznad", "ispod", "više", "poviše", "niže", "prije", "uoči", "poslije", "nakon", "za", "tijekom", "tokom", "dno", "vrh", "čelo", "nakraj", "onkraj", "krajem", "potkraj", "nasred", "oko", "okolo", "blizu", "kod", "kraj", "pokraj", "pored", "nadomak", "nadohvat", "mimo", "moj", "moje", "mojeg", "i", "u", "duž", "uzduž", "širom", "diljem", "preko", "bez", "osim", "mjesto", "umjesto", "namjesto", "uime", "putem", "pomoću", "posredstvom", "između", "naspram", "put", "protiv", "nasuprot", "usuprot", "usprkos", "unatoč", "zbog", "uslijed", "rad", "zaradi", "poradi", "glede", "prigodom", "prilikom", "povodom", "moj", "tvoj", "tvog", "tvojeg", "njegov", "njegovog", "njegovoga", "njezin", "njezinog", "njenog", "naš", "vaš", "njihov", "njihovoga", "ovaj", "ovome", "taj", "onaj", "ovakav", "onakav", "takav", "ovolik", "tolik", "onolik", "ovom", "ovog", "ovim", "ovih", "koji", "kojeg", "kojem", "kojih", "kojim", "koje", "tko", "što", "koji", "čiji", "kakav", "koji", "kakav", "nitko", "ništa", "ničiji", "nikakav", "netko", "nešto", "neki", "nekakav", "nečiji", "gdjetko", "gdješto", "gdjekoji", "gdjekakav", "tkogod", "štogod", "kojigod", "kakavgod", "čijigod", "itko", "išta", "ikoji", "ikakav", "svatko", "svašta", "svaki", "svačiji", "svakakav", "sav", "ma tko", "ma šta", "ma koji", "ma kakav", "ma čiji", "ma kolik", "kojetko", "koješta", "kojekakav", "bilo tko", "bilo što", "bilo koji", "bilo čiji", "bilo kakav", "god", "seb\w+", "ja", "ti", "on", "ona", "ono", "meni", "mene", "me", "mnom", "mi", "nas", "nama", "naš", "nama", "ti", "tebe", "te", "tebi", "tobom", "vi", "vas", "vaš", "vama", "vam", "vama", "njega", "ga", "mu", "njemu", "njim", "njime", "oni", "njih", "ih", "njima", "im", "nje", "je", "njoj", "joj", "nju", "ju", "njoj", "njome", "one", "njega", "ga", "njemu", "mu", "njim", "njime", "ona", "donjeg", "gornjeg", "kojih", "novih", "starih", "rekao", "reći", "rekla", "toga", "ukupno", "više", "manje", "god", "sata", "sati", "sada", "više", "novih", "starih", "dana", "rekao", "eur", "hrk", "iznosi"]
rijeci = rijeci + druga
pattern2 = r'\b(?:{})\b'.format('|'.join(rijeci))
podaci['Text'] = podaci["Text"].str.replace(pattern2, '')

# specijalni znakovi
spec_chars = ["!",'\"', "#", "%", "\$", "&","\'","\(","\)","\*","\+",",","-","\.","/",":",";","<", "=",">","\?","@","\[","\\","\]","\^","_", "`","\{","\|","\}","~","–"]
pattern = r'^\w+{}$'.format('|'.join(spec_chars))
podaci['Text'] = podaci["Text"].str.replace(pattern, '')

# brojevi i dodatni znakovi
podaci[["Text"]] = podaci[["Text"]].replace({'–':''}, regex=True)
podaci[["Text"]] = podaci[["Text"]].replace('\d+', '', regex = True)
podaci[["Text"]] = podaci[["Text"]].replace(',', '', regex = True)
podaci[["Text"]] = podaci[["Text"]].replace('’', '', regex = True)

# grupiranje po periodima
pocetak = "2020-01-01"
kraj = "2020-02-24"
posli = podaci["Datum"] >= pocetak 
prije = podaci["Datum"] <= kraj
izmedu = posli & prije
prvi_period = podaci.loc[izmedu]

pocetak = "2020-02-25"
kraj = "2020-03-13"
posli = podaci["Datum"] >= pocetak 
prije = podaci["Datum"] <= kraj
izmedu = posli & prije
drugi_period = podaci.loc[izmedu]

pocetak = "2020-03-14"
kraj = "2020-05-11"
posli = podaci["Datum"] >= pocetak 
prije = podaci["Datum"] <= kraj
izmedu = posli & prije
treci_period = podaci.loc[izmedu]

pocetak = "2020-05-12"
kraj = "2020-08-25"
posli = podaci["Datum"] >= pocetak 
prije = podaci["Datum"] <= kraj
izmedu = posli & prije
cetvrti_period = podaci.loc[izmedu]

# izvlačenje svakog stupca u pojedini string
prvi = prvi_period.drop(columns = ["Datum"])
prvi_list = prvi['Text'].tolist()
def flatten_list(lista):
    flat_list = []
    # Iterate through the outer list
    for element in lista:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

lista_prva = flatten_list(prvi_list)
print(lista_prva)
string_prvi = " ".join(str(x) for x in lista_prva)
print(string_prvi)
string_prvi = " ".join(string_prvi.split())

# drugi period
drugi = drugi_period.drop(columns = ["Datum"])
drugi_list = drugi['Text'].tolist()
lista_druga = flatten_list(drugi_list)
print(lista_druga)
string_drugi = " ".join(str(x) for x in lista_druga)
print(string_drugi)
string_drugi = string_drugi.replace('”', " ")
string_drugi = " ".join(string_drugi.split())

# treći period
treci = treci_period.drop(columns = ["Datum"])
treci_list = treci['Text'].tolist()
lista_treca = flatten_list(treci_list)
treca = []
for x in lista_treca:
    treca.append(x.strip())
string_treci = " ".join(str(x) for x in treca)
print(string_treci)
string_treci = " ".join(string_treci.split())
string_treci = string_treci.replace('\\', " ")

cetvrti = cetvrti_period.drop(columns = ["Datum"])
cetvrti_list = cetvrti['Text'].tolist()
lista_cetvrta = flatten_list(cetvrti_list)
string_cetvrti = " ".join(str(x) for x in lista_cetvrta)
print(string_cetvrti)
string_cetvrti = " ".join(string_cetvrti.split())
string_cetvrti = string_cetvrti.replace('•', "")

# stvaranje dataframe-a sa parovima i frekvencijama

def broj_parova(string):
    rijeci = re.findall("\w+", string)
    parovi = zip(rijeci, rijeci[1:])
    return collections.Counter(parovi)
    
lista_1 = broj_parova(string_prvi)
df = pd.DataFrame.from_dict(lista_1, orient='index').reset_index()
df.columns = ["rijeci", "tezina"]
tezina = df["tezina"]
df1 = pd.DataFrame(df['rijeci'].tolist(), index=df.index) 
df1 = df1.join(tezina)
df1.columns = ["source", "target", "tezina"]

lista_2 = broj_parova(string_drugi)
df = pd.DataFrame.from_dict(lista_2, orient='index').reset_index()
df.columns = ["rijeci", "tezina"]
tezina = df["tezina"]
df2 = pd.DataFrame(df['rijeci'].tolist(), index=df.index) 
df2 = df2.join(tezina)
df2.columns = ["source", "target", "tezina"]

lista_3 = broj_parova(string_treci)
df = pd.DataFrame.from_dict(lista_3, orient='index').reset_index()
df.columns = ["rijeci", "tezina"]
tezina = df["tezina"]
df3 = pd.DataFrame(df['rijeci'].tolist(), index=df.index) 
df3 = df3.join(tezina)
df3.columns = ["source", "target", "tezina"]

lista_4 = broj_parova(string_cetvrti)
df = pd.DataFrame.from_dict(lista_4, orient='index').reset_index()
df.columns = ["rijeci", "tezina"]
tezina = df["tezina"]
df4 = pd.DataFrame(df['rijeci'].tolist(), index=df.index) 
df4 = df4.join(tezina)
df4.columns = ["source", "target", "tezina"]

# ukupni za globalnu mrežu
frames = [df1, df2, df3, df4]
ukupno = pd.concat(frames)

# izrada mreže
graf = nx.from_pandas_edgelist(ukupno, source = "source", target = "target", edge_attr=True, create_using = nx.DiGraph)

# broj čvorova
graf.number_of_nodes()

# broj veza
graf.number_of_edges()

# prosječan stupanj
stupanj = dict(graf.degree(weight = "tezina"))
sum(stupanj.values())/graf.number_of_nodes()

# gustoća
nx.density(graf)
nx.number_strongly_connected_components(graf)

# prosječna duljina najkraćeg puta
nx.average_shortest_path_length(graf, weight = "tezina") 
#9.979163030202649
# dijametar
nx.diameter(graf) # 24

# prosječni koeficijent grupiranja
gl = nx.clustering(graf)
av = np.mean(np.fromiter(gl.values(), dtype=float))

# koeficijent asortativnosti
nx.degree_assortativity_coefficient(graf, weight = "tezina")

# histogram distribucije
def graf_distribucije_stupnjeva(graf):
    stupnjevi = [graf.degree(n) for n in graf.nodes()]
    plt.style.use("seaborn")
    plt.hist(stupnjevi, color = "#53d475")
    plt.xlabel("Stupanj")
    plt.ylabel("Broj čvorova")
    plt.show()

graf_distribucije_stupnjeva(graf)

# mreže prema periodima
#prvi period
graf1 = nx.from_pandas_edgelist(df1, source = "source", target = "target", edge_attr=True, create_using = nx.DiGraph)

centralnost_stupnjeva = nx.degree_centrality(graf1)
centralnost_top10 = sorted(centralnost_stupnjeva, key=centralnost_stupnjeva.get, reverse=True)[:10]

page_rank = nx.pagerank(graf1, weight = "tezina")
page_rank_top10 = sorted(page_rank, key=page_rank.get, reverse=True)[:10]

# vizualizacija
def draw(G, pos, node_size, measures, measure_name):
    # grafovi prema centralnosti
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, cmap=plt.cm.jet, 
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
        node_sizes.append( 5000 * n )
pos = nx.spring_layout(graf1, seed=1)
draw(graf1, pos, node_sizes, dict(graf1.degree), 'Centralnost stupnja - prvi period')

plt.figure(3,figsize=(200,200))
node_sizes = []
for n in page_rank.values():
        node_sizes.append( 10000 * n )
draw(graf1, pos, node_sizes, nx.pagerank(graf1), 'Page rank - prvi period')

# drugi period
graf2 = nx.from_pandas_edgelist(df2, source = "source", target = "target", edge_attr=True, create_using = nx.DiGraph)

centralnost_stupnjeva = nx.degree_centrality(graf2)
centralnost2_top10 = sorted(centralnost_stupnjeva, key=centralnost_stupnjeva.get, reverse=True)[:10]

page_rank = nx.pagerank(graf2, weight = "tezina")
page_rank2_top10 = sorted(page_rank, key=page_rank.get, reverse=True)[:10]

# vizualizacija
plt.figure(3,figsize=(200,200)) 
node_sizes = []
for n in centralnost_stupnjeva.values():
        node_sizes.append( 5000 * n )
pos = nx.spring_layout(graf2, seed=1)
draw(graf2, pos, node_sizes, dict(graf2.degree), 'Centralnost stupnja - drugi period')

plt.figure(3,figsize=(200,200))
node_sizes = []
for n in page_rank.values():
        node_sizes.append( 10000 * n )
draw(graf2, pos, node_sizes, nx.pagerank(graf2), 'Page rank - drugi period')

# treći period
graf3 = nx.from_pandas_edgelist(df3, source = "source", target = "target", edge_attr=True, create_using = nx.DiGraph)

centralnost_stupnjeva = nx.degree_centrality(graf3)
centralnost3_top10 = sorted(centralnost_stupnjeva, key=centralnost_stupnjeva.get, reverse=True)[:10]

page_rank = nx.pagerank(graf3, weight = "tezina")
page_rank3_top10 = sorted(page_rank, key=page_rank.get, reverse=True)[:10]

# vizualizacija
random_nodes = sample(list(graf3.nodes()), 3000)
graf3_1 = graf3.subgraph(random_nodes)
plt.figure(3,figsize=(200,200)) 
node_sizes = []
for n in centralnost_stupnjeva.values():
        node_sizes.append( 3000 * n )
pos = nx.spring_layout(graf3_1, seed=1)
draw(graf3_1, pos, node_sizes, dict(graf3_1.degree), 'Centralnost stupnja - treći period')

plt.figure(3,figsize=(200,200))
node_sizes = []
for n in page_rank.values():
        node_sizes.append( 10000 * n )
draw(graf3_1, pos, node_sizes, nx.pagerank(graf3_1), 'Page rank - treći period')

# cetvrti period
graf4 = nx.from_pandas_edgelist(df4, source = "source", target = "target", edge_attr=True, create_using = nx.DiGraph)

centralnost_stupnjeva = nx.degree_centrality(graf4)
centralnost4_top10 = sorted(centralnost_stupnjeva, key=centralnost_stupnjeva.get, reverse=True)[:10]

page_rank = nx.pagerank(graf4, weight = "tezina")
page_rank4_top10 = sorted(page_rank, key=page_rank.get, reverse=True)[:10]

# vizualizacija
plt.figure(3,figsize=(200,200)) 
node_sizes = []
for n in centralnost_stupnjeva.values():
        node_sizes.append( 3000 * n )
pos = nx.spring_layout(graf4, seed=1)
draw(graf4, pos, node_sizes, dict(graf4.degree), 'Centralnost stupnja - četvrti period')

plt.figure(3,figsize=(200,200))
node_sizes = []
for n in page_rank.values():
        node_sizes.append( 10000 * n )
draw(graf4, pos, node_sizes, nx.pagerank(graf4), 'Page rank - četvrti period')

# zajednice
communities = community.asyn_lpa_communities(graf1, weight = "tezina")
communities = list(communities)

color_map = []
for node in graf4:
    if node in communities[0]:
        color_map.append('#5d8aa8')
    elif node in communities[1]:
        color_map.append("#f0f8ff")
    elif node in communities[2]:
        color_map.append("#e32636")
    elif node in communities[3]:
        color_map.append("#ffbf00")
    elif node in communities[4]:
        color_map.append("#9966cc")
    elif node in communities[5]:
        color_map.append("#a4c639")
    elif node in communities[6]:
        color_map.append("#cd9575")
    elif node in communities[7]:
        color_map.append("#915c83")
    elif node in communities[8]:
        color_map.append("#008000")
    elif node in communities[9]:
        color_map.append("#00ffff")
    elif node in communities[10]:
        color_map.append("#4b5320")
    elif node in communities[11]:
        color_map.append("#e9d66b")
    elif node in communities[12]:
        color_map.append("#007fff")
    elif node in communities[13]:
        color_map.append("#3d2b1f")
    elif node in communities[14]:
        color_map.append("#cc0000")
    else:
        color_map.append("#536872")
    
    

plt.figure(3,figsize=(200,200)) 
sc = nx.draw_networkx_nodes(G=graf1, pos = pos, nodelist = graf1.nodes(), alpha=0.9, node_size = 10, node_color=color_map)
nx.draw_networkx_edges(G = graf1, pos = pos, edge_color='#818a8c', alpha=0.6, width=1)
plt.show() # 6 zajednica

communities = community.asyn_lpa_communities(graf2, weight = "tezina")
communities = list(communities)
plt.figure(3,figsize=(200,200)) 
sc = nx.draw_networkx_nodes(G=graf2, pos = pos, nodelist = graf2.nodes(), alpha=0.9, node_size = 10, node_color=color_map)
nx.draw_networkx_edges(G = graf2, pos = pos, edge_color='#818a8c', alpha=0.6, width=1)
plt.show() # 9 zajednica

communities = community.asyn_lpa_communities(graf3, weight = "tezina")
communities = list(communities)


plt.figure(3,figsize=(200, 200)) 
sc = nx.draw_networkx_nodes(G=graf3_1, pos = pos, nodelist = graf3_1.nodes(), alpha=0.8, node_size = 5, node_color=color_map)
nx.draw_networkx_edges(G = graf3_1, pos = pos, edge_color='#818a8c', alpha=0.4, width=1)
plt.show() # 16 zajednica
random_nodes = sample(list(graf3.nodes()), 5000)
graf3_1 = graf3.subgraph(random_nodes)

communities = community.asyn_lpa_communities(graf4, weight = "tezina")
communities = list(communities)

plt.figure(3,figsize=(200, 200)) 
sc = nx.draw_networkx_nodes(G=graf4, pos = pos, nodelist = graf4.nodes(), alpha=0.8, node_size = 5, node_color=color_map)
nx.draw_networkx_edges(G = graf4, pos = pos, edge_color='#818a8c', alpha=0.4, width=0.8)
plt.show()

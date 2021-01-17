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
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import jaccard_score
import difflib
from difflib import SequenceMatcher
import textdistance
import scipy
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import distance
from io import StringIO

#učitavanje csv datoteke
podaci = pd.read_csv("C:/Users/38591/AppData/Local/Programs/Python/Python36/baranjainfopravi.csv", sep=";")

#provjera zaglavlja
podaci.head(4)

#provjera oblika data frame-a
podaci.shape
podaci.columns

#pregled tipova podataka
podaci.info()

#izbacivanje nepotrebnih varijabli
podaci = podaci.drop(columns = ["Broj", "Link"])
podaci.shape

podaci = podaci.rename(columns={'Datum objave': 'Datum'})

#pretvorba datuma u datetime objekt za lakšu kasniju manipulaciju, izbacivanje NaN vrijednosti
podaci["Datum"] = pd.to_datetime(podaci["Datum"], format="%Y-%m-%d", errors='coerce')
podaci["Datum"].isna().sum()
podaci = podaci.dropna(subset = ['Datum'])
podaci["Datum"].isna().sum()

podaci.info()

#konvertanje svih tekstova u lower case zbog pronalaska određenih riječi
podaci['Naslov'] = podaci['Naslov'].str.lower()
podaci['Tekst'] = podaci['Tekst'].str.lower()
podaci['Kategorija'] = podaci['Kategorija'].str.lower()
podaci.head(3)

#ukupan broj korona objava
rijeci_korona = ["koron\w+", "koronavirus\w*", "virus\w*", "capak", "epidemij\w+", "covid\w*", "bolnic\w+", "zaražen\w*", "novooboljel\w+","samoizolacij\w+", "karanten\w+", "lockdown\w*", "novozaražen\w*", "testiranj\w+", "pozitivn\w+", "mjer\w+", "stožer\w*", "epidemiološk\w+", "zaraza", "novopozitivn\w+", "cjepiv\w+", "oboljel\w+", "respirator\w*", "propusnic\w+", "dezinficir\w+", "socijaln\w+\sdistanc\w+", "alemka", "beroš", "oboljeli", "preboljel\w+", "Markotić", "Božinović"]
ukupno_korona = podaci[(podaci["Naslov"].str.contains('|'.join(rijeci_korona))) & (podaci["Tekst"].str.contains('|'.join(rijeci_korona)))]
ukupno_korona.shape
ukupno_korona.head()

# grupirano prema danima i brojevi clanaka
grupirano_po_datumima = podaci.groupby("Datum")
brojevi_clanaka_datum = grupirano_po_datumima.size().reset_index(name="Broj_članaka")
grupirano_po_datumima.head()

#izradag grupiranog skupa podataka prema datumima
grupirana_datum = grupirano_po_datumima[["Naslov", "Tekst", "Autor", "Kategorija"]].agg(lambda column: " ".join(column))
grupirana_datum

#izradag grupiranog skupa podataka prema datumima vezano uz koronu
grupirano_po_datumima_korona = ukupno_korona.groupby("Datum")
brojevi_clanaka_datum_korona = grupirano_po_datumima_korona.size().reset_index(name="Broj_članaka_korona")
grupirano_po_datumima.head()

#join dataframe-a za tablični prikaz
final_datum_korona = brojevi_clanaka_datum.merge(brojevi_clanaka_datum_korona, on='Datum', how = "outer")
final_datum_korona['Datum'] = pd.to_datetime(final_datum_korona['Datum']).dt.date
fig, ax = plt.subplots()
ax.plot(final_datum_korona["Datum"], final_datum_korona["Broj_članaka"])
ax.plot(final_datum_korona["Datum"], final_datum_korona["Broj_članaka_korona"])
plt.show()

# brojanje članaka po misecima
grupirano_po_mjesecima = podaci.groupby(podaci["Datum"].dt.month)
brojevi_clanaka_mjesec = grupirano_po_mjesecima.size().reset_index(name="Broj_članaka")
grupirano_po_mjesecima.head()
grupirano_po_mjesecima_korona = ukupno_korona.groupby(podaci.Datum.dt.month)
brojevi_clanaka_mjesec_korona = grupirano_po_mjesecima_korona.size().reset_index(name="Broj_članaka_korona")

#join dataframe-ova za mjesece
final_mjesec_korona = brojevi_clanaka_mjesec.merge(brojevi_clanaka_mjesec_korona, on='Datum', how = "outer")
final_mjesec_korona = final_mjesec_korona.replace({'Datum' : { 1 : "Siječanj", 2 : "Veljača", 3 : "Ožujak", 4: "Travanj", 5: "Svibanj", 6: "Lipanj", 7: "Srpanj", 8: "Kolovoz", 9: "Rujan", 10: "Listopad", 11: "Studeni" }})
final_mjesec_korona = final_mjesec_korona.set_index("Datum")
fig, ax = plt.subplots()
ax.plot(final_mjesec_korona.index, final_mjesec_korona["Broj_članaka"])
ax.plot(final_mjesec_korona.index, final_mjesec_korona["Broj_članaka_korona"])
plt.show()

#brojevi članaka prema kategorijama
grupirano_po_kategorijama = podaci.groupby(podaci.Kategorija)
brojevi_clanaka_kategorija = grupirano_po_kategorijama.size().reset_index(name="Broj_članaka")
grupirano_po_kategorijama_korona = ukupno_korona.groupby(podaci.Kategorija)
brojevi_clanaka_kategorija_korona = grupirano_po_kategorijama_korona.size().reset_index(name="Broj_članaka_korona")
final_kategorija_korona = brojevi_clanaka_kategorija.merge(brojevi_clanaka_kategorija_korona, on='Kategorija')

# Vizualizacije
broj_clanaka = len(podaci.index)
broj_korona_clanaka = len(ukupno_korona.index)
broj = [broj_clanaka, broj_korona_clanaka]
nazivi = ["Obični članaci", "Članci vezani uz koronu"]
fig, ax = plt.subplots()
plt.style.use("seaborn")
ax.bar(nazivi, broj, color = "#DA70D6")
ax.set_ylabel("Broj članaka")
ax.set_title("Odnos broja članaka za period 01.01.2020 - 30.11.2020")
plt.show()

# postotak korona članaka
broj_korona_clanaka/broj_clanaka*100

# vizualizacija po mjesecima
fig, ax = plt.subplots()
plt.style.use('seaborn')
plot = pd.concat([final_mjesec_korona.Broj_članaka.rename('Ukupan broj članaka'), final_mjesec_korona.Broj_članaka_korona.rename('Ukupan broj korona članaka')], axis=1).plot.bar(color = ["#3CB371", "#FF7F50"])
plt.show()

# vizualizacija po kategorijama
final_kategorija_korona = final_kategorija_korona.set_index("Kategorija")
plt.style.use('seaborn')
plot = pd.concat([final_kategorija_korona.Broj_članaka.rename('Ukupan broj članaka'), final_kategorija_korona.Broj_članaka_korona.rename('Ukupan broj korona članaka')], axis=1).plot.bar(color = ["#87CEEB", "#D2B48C"])

# izbacivanje posebnih znakova i zaustavnih riječi
zaustavne = ["a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bismo", "biste", "biti", "bumo", "da", "do", "duž", "ga", "hoće", "hoćemo", "hoćete", "hoćeš", "hoću", "i", "iako", "ih", "ili", "iz", "ja", "je", "jedna", "jedne", "jedno", "jer", "jesam", "jesi", "jesmo", "jest", "jeste", "jesu", "jim", "joj", "još", "ju", "kada", "kako", "kao", "koja", "koje", "koji", "kojima", "koju", "kroz", "li", "me", "mene", "meni", "mi", "mimo", "moj", "moja", "moje", "mu", "na", "nad", "nakon", "nam", "nama", "nas", "naš", "naša", "naše", "našeg", "ne", "nego", "neka", "neki", "nekog", "neku", "nema", "netko", "neće", "nećemo", "nećete", "nećeš", "neću", "nešto", "ni", "nije", "nikoga", "nikoje", "nikoju", "nisam", "nisi", "nismo", "niste", "nisu", "njega", "njegov", "njegova", "njegovo", "njemu", "njezin", "njezina", "njezino", "njih", "njihov", "njihova", "njihovo", "njim", "njima", "njoj", "nju", "no", "o", "od", "odmah", "on", "ona", "oni", "ono", "ovo", "ova", "pa", "pak", "po", "pod", "pored", "prije", "s", "sa", "sam", "samo", "se", "sebe", "sebi", "si", "smo", "ste", "su", "sve", "svi", "svog", "svoj", "svoja", "svoje", "svom", "ta", "tada", "taj", "tako", "te", "tebe", "tebi", "ti", "to", "toj", "tome", "tu", "tvoj", "tvoja", "tvoje", "u", "uz", "vam", "vama", "vas", "vaš", "vaša", "vaše", "već", "vi", "vrlo", "za", "zar", "će", "ćemo", "ćete", "ćeš", "ću", "što"]
pattern = r'\b(?:{})\b'.format('|'.join(zaustavne))
novi_korona = ukupno_korona
novi_korona['Naslov1'] = novi_korona["Naslov"].str.replace(pattern, '')
novi_korona['Tekst1'] = novi_korona["Tekst"].str.replace(pattern, '')

# drop starih varijabli
novi_korona = novi_korona.drop(columns = ["Naslov", "Tekst"])
novi_korona['Datum'] = pd.to_datetime(novi_korona['Datum']).dt.date

# izbacivanje priloga, prijedloga, usklika, veznika i zamjenica
rijeci = ["ovoliko", "toliko", "onoliko", "nekoliko", "zadnja", "prva", "druga", "iduća", "malo", "premalo", "više", "previše", "prekoviše", "najviše", "ponajviše", "manje", "najmanje", "ponajmanje", "dosta", "odveć", "opet", "još", "sasvim", "potpuno", "previše", "odsad", "otad", "oduvijek", "odavna", "danas", "večeras", "noćas", "jučer", "sinoć", "preksinoć", "sutra", "preksutra", "ljetos", "proljetos", "jesenas", "zimus", "proljeti", "ljeti", "jeseni", "zimi", "nadlani", "preklani", "obdan", "obnoć", "odmah", "smjesta", "sada", "tada", "onda", "ikada", "bilo", "kada", "nikada", "nekada", "ponekad", "katkad", "uvijek", "svagda", "često", "rijetko", "rano", "kasno", "prije", "poslije", "potom", "nedavno", "skoro", "uskoro", "napokon", "dosad", "dotad", "dogodine", "ovdje", "tu", "ondje", "negdje", "igdje", "nigdje", "onegdje", "gore", "dolje", "unutra", "vani", "ovamo", "onamo", "tamo", "nekamo", "nikamo", "ikamo", "naprijed", "natrag", "ovuda", "onuda", "tuda", "nikuda", "odavde", "otud", "odatle", "odonud", "niotkuda", "odozgo", "odozdo", "odostraga", "izdaleka", "izvana", "izbliza", "donekle", "onako", "ovako", "tako", "slučajno", "zato", "stoga", "uzalud", "uzaman", "utaman", "ovoliko", "toliko", "onoliko", "nekoliko", "malo", "premalo", "više", "previše", "prekoviše", "najviše", "ponajviše", "manje", "najmanje", "ponajmanje", "dosta", "odveć", "opet", "još", "sasvim", "potpuno", "previše", "ah", "oh", "hehe", "he", "hura", "jao", "joj", "oho", "uh", "uf", "ijuju", "haj", "eh", "ehe", "i", "pa", "te", "ni", "niti", "ili", "samo", "samo što", "jedino", "jedino što", "tek", "tek što", "dakle", "zato", "stoga", "ali", "nego", "no", "već", "da", "te", "kako", "čega", "neka", "jer", "ka", "prema", "napram"]
druga = ["nadomak", "nadohvat", "posljednjeg", "godine", "odnosno", "trenutni","trenutnog","trenutnih","imamo","imati","donjeg", "jedan", "prvog", "drugog", "dva", "tri", "četiri", "pet", "šest", "sedam", "osam", "devet", "posljednja", "deset", "broj", "godine", "trenutno", "gornjeg", "dana", "nasuprot", "usuprot", "usprkos", "unatoč", "protiv", "kroz", "niz", "uz", "na", "po", "mimo", "među", "nad", "pod", "pred", "za", "od", "do", "iz", "ispred", "iza", "izvan", "van", "unutar", "iznad", "ispod", "više", "poviše", "niže", "prije", "uoči", "poslije", "nakon", "za", "tijekom", "tokom", "dno", "vrh", "čelo", "nakraj", "onkraj", "krajem", "potkraj", "nasred", "oko", "okolo", "blizu", "kod", "kraj", "pokraj", "pored", "nadomak", "nadohvat", "mimo", "moj", "moje", "mojeg", "i", "u", "duž", "uzduž", "širom", "diljem", "preko", "bez", "osim", "mjesto", "umjesto", "namjesto", "uime", "putem", "pomoću", "posredstvom", "između", "naspram", "put", "protiv", "nasuprot", "usuprot", "usprkos", "unatoč", "zbog", "uslijed", "rad", "zaradi", "poradi", "glede", "prigodom", "prilikom", "povodom", "moj", "tvoj", "tvog", "tvojeg", "njegov", "njegovog", "njegovoga", "njezin", "njezinog", "njenog", "naš", "vaš", "njihov", "njihovoga", "ovaj", "ovome", "taj", "onaj", "ovakav", "onakav", "takav", "ovolik", "tolik", "onolik", "ovom", "ovog", "ovim", "ovih", "koji", "kojeg", "kojem", "kojih", "kojim", "koje", "tko", "što", "koji", "čiji", "kakav", "koji", "kakav", "nitko", "ništa", "ničiji", "nikakav", "netko", "nešto", "neki", "nekakav", "nečiji", "gdjetko", "gdješto", "gdjekoji", "gdjekakav", "tkogod", "štogod", "kojigod", "kakavgod", "čijigod", "itko", "išta", "ikoji", "ikakav", "svatko", "svašta", "svaki", "svačiji", "svakakav", "sav", "ma tko", "ma šta", "ma koji", "ma kakav", "ma čiji", "ma kolik", "kojetko", "koješta", "kojekakav", "bilo tko", "bilo što", "bilo koji", "bilo čiji", "bilo kakav", "god", "seb\w+", "ja", "ti", "on", "ona", "ono", "meni", "mene", "me", "mnom", "mi", "nas", "nama", "naš", "nama", "ti", "tebe", "te", "tebi", "tobom", "vi", "vas", "vaš", "vama", "vam", "vama", "njega", "ga", "mu", "njemu", "njim", "njime", "oni", "njih", "ih", "njima", "im", "nje", "je", "njoj", "joj", "nju", "ju", "njoj", "njome", "one", "njega", "ga", "njemu", "mu", "njim", "njime", "ona", "donjeg", "gornjeg", "kojih", "novih", "starih", "rekao", "reći", "rekla", "toga", "ukupno", "više", "manje", "god", "sata", "sati", "sada", "više", "novih", "starih", "dana", "rekao", "eur", "hrk", "iznosi"]
rijeci = rijeci + druga
pattern2 = r'\b(?:{})\b'.format('|'.join(rijeci))
novi_korona['Naslov'] = novi_korona["Naslov1"].str.replace(pattern2, '')
novi_korona['Tekst'] = novi_korona["Tekst1"].str.replace(pattern2, '')
novi_korona = novi_korona.drop(columns = ["Naslov1", "Tekst1"])

# izbacivanje specijalnih znakova
spec_chars = ["!",'\"', "#", "%", "\$", "&","\'","\(","\)","\*","\+",",","-","\.","/",":",";","<", "=",">","\?","@","\[","\\","\]","\^","_", "`","\{","\|","\}","~","–"]
pattern1 = r'^\w+{}$'.format('|'.join(spec_chars))
print(pattern1)
novi_korona['Naslov1'] = novi_korona["Naslov"].str.replace(pattern1, '')
novi_korona['Tekst1'] = novi_korona["Tekst"].str.replace(pattern1, '')
novi_korona = novi_korona.drop(columns = ["Naslov", "Tekst"])
novi_korona.info()

novi_korona = novi_korona.rename(columns={"Naslov1": "Naslov", "Tekst1":"Tekst"})
novi_korona["Datum"] = pd.to_datetime(novi_korona["Datum"], format="%Y-%m-%d", errors='coerce')

# grupiranje po mjesecima
grupirano_po_mjesecima = novi_korona.groupby(novi_korona.Datum.dt.month)
grupirana_mjesec = grupirano_po_mjesecima[["Naslov", "Tekst", "Autor", "Kategorija"]].agg(lambda column: " ".join(column))
spojene_kategorije = novi_korona.groupby([novi_korona.Datum.dt.month])['Kategorija'].apply(lambda x: ','.join(x)).reset_index()
spojeni_naslovi = novi_korona.groupby([novi_korona.Datum.dt.month])['Naslov'].apply(lambda x: ','.join(x)).reset_index()
spojene_tekstovi = novi_korona.groupby([novi_korona.Datum.dt.month])['Tekst'].apply(lambda x: ','.join(x)).reset_index()
spojene_autori = novi_korona.groupby([novi_korona.Datum.dt.month])['Autor'].apply(lambda x: ','.join(x)).reset_index()

# spajanje stvorenih data frame-ova i izbacivanje dodatnih znakova
grupirana_korona = spojeni_naslovi.merge(spojene_tekstovi, on='Datum', how='outer')
grupirana_korona = grupirana_korona.merge(spojene_kategorije, on = "Datum", how = "outer")
grupirana_korona = grupirana_korona.merge(spojene_autori, on = "Datum", how = "outer")
grupirana_korona[["Naslov", "Tekst"]] = grupirana_korona[["Naslov", "Tekst"]].replace({'–':''}, regex=True)
grupirana_korona[['Naslov', "Tekst"]] = grupirana_korona[["Naslov", "Tekst"]].replace('\d+', '', regex = True)
grupirana_korona[['Naslov', "Tekst"]] = grupirana_korona[["Naslov", "Tekst"]].replace(',', '', regex = True)
grupirana_korona[['Naslov', "Tekst"]] = grupirana_korona[["Naslov", "Tekst"]].replace('’', '', regex = True)

# liste najčešćih riječi po mjesecima
sijecanj = grupirana_korona.loc[0, ['Naslov', 'Tekst']]
sijecanj_count = sijecanj.str.split(expand=True).stack().value_counts(ascending = False)[:25]

veljaca = grupirana_korona.loc[1, ['Naslov', 'Tekst']]
veljaca_count = veljaca.str.split(expand=True).stack().value_counts(ascending = False)[:25]

ozujak = grupirana_korona.loc[2, ['Naslov', 'Tekst']]
ozujak_count = ozujak.str.split(expand=True).stack().value_counts(ascending = False)[:25]

travanj = grupirana_korona.loc[3, ['Naslov', 'Tekst']]
travanj_count = travanj.str.split(expand=True).stack().value_counts()[:25]

svibanj = grupirana_korona.loc[4, ['Naslov', 'Tekst']]
svibanj_count = svibanj.str.split(expand=True).stack().value_counts()[:25]

lipanj = grupirana_korona.loc[5, ['Naslov', 'Tekst']]
lipanj_count = lipanj.str.split(expand=True).stack().value_counts()[:25]

srpanj = grupirana_korona.loc[6, ['Naslov', 'Tekst']]
srpanj_count = srpanj.str.split(expand=True).stack().value_counts()[:25]

kolovoz = grupirana_korona.loc[7, ['Naslov', 'Tekst']]
kolovoz_count = kolovoz.str.split(expand=True).stack().value_counts()[:25]

rujan = grupirana_korona.loc[8, ['Naslov', 'Tekst']]
rujan_count = rujan.str.split(expand=True).stack().value_counts()[:25]

listopad = grupirana_korona.loc[9, ['Naslov', 'Tekst']]
listopad_count = listopad.str.split(expand=True).stack().value_counts()[:25]

studeni = grupirana_korona.loc[10, ['Naslov', 'Tekst']]
studeni_count = studeni.str.split(expand=True).stack().value_counts()[:25]

# pretvaranje stvorene serije u rjecnik za potrebe word cloud funkcije
string_sijecanj = sijecanj_count.to_dict()
string_veljaca = veljaca_count.to_dict()
string_ozujak = ozujak_count.to_dict()
string_travanj = travanj_count.to_dict()
string_svibanj = svibanj_count.to_dict()
string_lipanj = lipanj_count.to_dict()
string_srpanj = srpanj_count.to_dict()
string_kolovoz = kolovoz_count.to_dict()
string_rujan = rujan_count.to_dict()
string_listopad = listopad_count.to_dict()
string_studeni = studeni_count.to_dict()

# vizualizacija word cloud-a po mjesecima
wc_sijecanj = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_sijecanj.items()]))
fig = plt.figure()
plt.imshow(wc_sijecanj, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_veljaca = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_veljaca.items()]))
fig = plt.figure()
plt.imshow(wc_veljaca, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_ozujak = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_ozujak.items()]))
fig = plt.figure()
plt.imshow(wc_ozujak, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_travanj = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_travanj.items()]))
fig = plt.figure()
plt.imshow(wc_travanj, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_svibanj = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_svibanj.items()]))
fig = plt.figure()
plt.imshow(wc_svibanj, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_lipanj = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_lipanj.items()]))
fig = plt.figure()
plt.imshow(wc_lipanj, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_srpanj = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_srpanj.items()]))
fig = plt.figure()
plt.imshow(wc_srpanj, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_kolovoz = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_kolovoz.items()]))
fig = plt.figure()
plt.imshow(wc_kolovoz, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_rujan = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_rujan.items()]))
fig = plt.figure()
plt.imshow(wc_rujan, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_listopad = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_listopad.items()]))
fig = plt.figure()
plt.imshow(wc_listopad, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_studeni = WordCloud(max_font_size=40, background_color = "white", collocations=False).generate(" ".join([(k + ' ') * v for k,v in string_studeni.items()]))
fig = plt.figure()
plt.imshow(wc_studeni, interpolation="bilinear")
plt.axis("off")
plt.show()

# Jaccard index - 4. dio zadatka
jan = " ".join(list(sijecanj_count.reset_index()["index"]))
feb = " ".join(list(veljaca_count.reset_index()["index"]))
mar = " ".join(list(ozujak_count.reset_index()["index"]))
apr = " ".join(list(travanj_count.reset_index()["index"]))
may = " ".join(list(svibanj_count.reset_index()["index"]))
jun = " ".join(list(lipanj_count.reset_index()["index"]))
jul = " ".join(list(srpanj_count.reset_index()["index"]))
aug = " ".join(list(kolovoz_count.reset_index()["index"]))
sep = " ".join(list(rujan_count.reset_index()["index"]))
okt = " ".join(list(listopad_count.reset_index()["index"]))
nov = " ".join(list(studeni_count.reset_index()["index"]))

# skupljena lista za sve mjesece - stvaranje liste stringova
stringovi = []
stringovi.extend((aug, feb, mar, apr, may, jun, jul, nov, sep, okt, jan))
print(stringovi)

# izracun jaccard indexa, mozemo upotrijebit i sorensen, overlap, tversky
data = [jan, feb, mar, apr, may, jun, jul, aug, sep, okt, nov]
dm = [[ textdistance.jaccard(a, b) for b in data] for a in data]
textdistance_table ='\n'.join([''.join([f'{item:6.2f}' for item in row]) for row in dm])

df = pd.read_csv(StringIO(re.sub(r'[-+|]', '', textdistance_table)), sep='\s{2,}', engine='python', names = ["Siječanj", "Veljača", "Ožujak", "Travanj", "Svibanj", "Lipanj", "Srpanj", "Kolovoz", "Rujan", "Listopad", "Studeni"], header=None)
df = df.rename(index = { 0 : "Siječanj", 1 : "Veljača", 2 : "Ožujak", 3: "Travanj", 4: "Svibanj", 5: "Lipanj", 6: "Srpanj", 7: "Kolovoz", 8: "Rujan", 9: "Listopad", 10: "Studeni" })

# sorensen
dm1 = [[ textdistance.sorensen(a, b) for b in data] for a in data]
textdistance_table1 ='\n'.join([''.join([f'{item:6.2f}' for item in row]) for row in dm1])

df1 = pd.read_csv(StringIO(re.sub(r'[-+|]', '', textdistance_table1)), sep='\s{2,}', engine='python', names = ["Siječanj", "Veljača", "Ožujak", "Travanj", "Svibanj", "Lipanj", "Srpanj", "Kolovoz", "Rujan", "Listopad", "Studeni"], header=None)
df1 = df1.rename(index = { 0 : "Siječanj", 1 : "Veljača", 2 : "Ožujak", 3: "Travanj", 4: "Svibanj", 5: "Lipanj", 6: "Srpanj", 7: "Kolovoz", 8: "Rujan", 9: "Listopad", 10: "Studeni" })

# overlap
dm2 = [[ textdistance.overlap(a, b) for b in data] for a in data]
textdistance_table2 ='\n'.join([''.join([f'{item:6.2f}' for item in row]) for row in dm2])

df2 = pd.read_csv(StringIO(re.sub(r'[-+|]', '', textdistance_table2)), sep='\s{2,}', engine='python', names = ["Siječanj", "Veljača", "Ožujak", "Travanj", "Svibanj", "Lipanj", "Srpanj", "Kolovoz", "Rujan", "Listopad", "Studeni"], header=None)
df2 = df2.rename(index = { 0 : "Siječanj", 1 : "Veljača", 2 : "Ožujak", 3: "Travanj", 4: "Svibanj", 5: "Lipanj", 6: "Srpanj", 7: "Kolovoz", 8: "Rujan", 9: "Listopad", 10: "Studeni" })

# jaro winkler
dm3 = [[ textdistance.tanimoto(a, b) for b in data] for a in data]
textdistance_table3 ='\n'.join([''.join([f'{item:6.2f}' for item in row]) for row in dm3])

df3 = pd.read_csv(StringIO(re.sub(r'[-+|]', '', textdistance_table3)), sep='\s{2,}', engine='python', names = ["Siječanj", "Veljača", "Ožujak", "Travanj", "Svibanj", "Lipanj", "Srpanj", "Kolovoz", "Rujan", "Listopad", "Studeni"], header=None)
df3 = df3.rename(index = { 0 : "Siječanj", 1 : "Veljača", 2 : "Ožujak", 3: "Travanj", 4: "Svibanj", 5: "Lipanj", 6: "Srpanj", 7: "Kolovoz", 8: "Rujan", 9: "Listopad", 10: "Studeni" })

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
import pandas as pd
import json

# prikupljanje url-a stranica iz određenog perioda 01.01.2020 - 30.11.2020.
linkovi=[]

for i in range(1, 210):
    res = requests.get('https://www.baranjainfo.hr/page/'+str(i)+'/?s')
    soup = BeautifulSoup(res.text, 'lxml')

    glavni_blok = soup.find('div', class_='td-ss-main-content')
    blok = glavni_blok.find_all('h3', class_='entry-title td-module-title')
    for svaki in blok:
        link = svaki.find('a')
        linkovi.append(link['href'])

# otvaranje objekta za ispis sadržaja u csv file
csv_file = open('baranjainfo2.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Naslov', 'Tekst', 'Datum objave', 'Autor', 'Kategorija'])

# scrapanje svih stranica sa popisa url-a za naslov, sadržaj, datu, autora i kategoriju
contents = []
with open('linkovi.csv','r') as csvf: # otvaranje datoteke s linkovima
    urls = csv.reader(csvf)
    for url in urls:
        contents.append(url) # dodavanje svakog url-a u listu sadržaja - contents

for url in contents:  # parsanje svakog url-a
    page = urlopen(url[0]).read()
    soup = BeautifulSoup(page, 'html5lib')
    response = requests.get(url)

    article = soup.find('article')

    #naslov članka
    headline = article.h1.text

    #tekst
    paragraphs = ["".join(x.findAll(text=True)) for x in article.findAllNext("p")]
    
    #dohvaćanje datuma
    datum = article.find('div', class_='td-module-meta-info').time.text
    #print(datum)

    #tko je napisao članak
    autor = article.find('div', class_='td-post-author-name').a
    authors = autor.text
    #print(autor.text)

    #dohvaćanje kategorije
    category = article.find('div', class_='td-post-header').ul
    for cat in category.find_all('a'):
        cats = cat.text
        #print(cats)

    #print()
    csv_writer.writerow([headline, paragraphs, datum, authors, cats])

csv_file.close()
        

#datoteka.close()

# convert datoteke u json file
"""csvfile = open('baranjainfocsv.csv', 'r', encoding="utf-8")
jsonfile = open('baranjainfo3.json', 'w', encoding="utf-8")

fieldnames = ("Broj","Link","Naslov","Tekst", "Datum objave", "Autor", "Kategorija")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')"""

csv_file_path = 'baranjainfocsv.csv'
json_file_path = 'baranjainfojson.json'

podaci = {}
with open(csv_file_path, encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for rows in csv_reader:
        id = rows["ID"]
        podaci[id] = rows
with open(json_file_path, 'w', encoding="utf-8") as json_file:
    json_file.write(json.dumps(podaci, indent = 4, ensure_ascii=False))







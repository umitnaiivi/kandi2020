from flask import Flask, render_template, request
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from urllib import request as rq
from bs4 import BeautifulSoup
import re
import nltk
import random
import math
from nltk.stem.snowball import SnowballStemmer
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
from nltk.cluster import GAAClusterer

def remove_duplicate_links(x):
    return list(dict.fromkeys(x))


def doc_counter():
    return str(len(artikkelit))

# Ulkomaan uutislinkkien rapsuttelu
stemmer = SnowballStemmer("finnish")
url = "https://yle.fi/uutiset/18-34953"  # ulkomaan uutiset
html = rq.urlopen(url).read().decode('utf8')
soup = BeautifulSoup(html, 'html.parser')

ulkomaan_linkit = soup.find_all(class_="yle__article__listItem__link", attrs={"href": re.compile("uutiset")})
ulkomaan_linkit2 = remove_duplicate_links([link.get("href") for link in ulkomaan_linkit])
ulkomaan_linkit3 = [("https://yle.fi" + linkki) for linkki in ulkomaan_linkit2]
regex = re.compile(r'd?-d?')
lopulliset_ulkomaan_linkit = list(filter(regex.search, ulkomaan_linkit3))

# Kotimaan uutislinkkien rapsuttelu
url = "https://yle.fi/uutiset/18-34837"
html = rq.urlopen(url).read().decode('utf8')
soup = BeautifulSoup(html, 'html.parser')
kotimaan_linkit = soup.find_all(class_="yle__article__listItem__link", attrs={"href": re.compile("uutiset")})
kotimaan_linkit2 = remove_duplicate_links([link.get("href") for link in kotimaan_linkit])
kotimaan_linkit3 = [("https://yle.fi" + linkki) for linkki in kotimaan_linkit2]
lopulliset_kotimaan_linkit = list(filter(regex.search, kotimaan_linkit3))


kaikki_linkit = []
for item in lopulliset_ulkomaan_linkit:
    if item not in lopulliset_kotimaan_linkit:
        kaikki_linkit.append(item)
for item in lopulliset_kotimaan_linkit:
    if item not in kaikki_linkit:
        kaikki_linkit.append(item)


artikkelit = []
puhtaat_linkit = []
for linkki in kaikki_linkit:
    html = rq.urlopen(linkki).read().decode("utf8")
    soup = BeautifulSoup(html, "html.parser")
    teksti = soup.article.find_all("p")     # teksti = list of paragraphs
    kappaleet = []
    for kappale in teksti:
        kappaleet.append(kappale.get_text())
    kappaleet = " ".join(kappaleet)
    artikkelit.append(kappaleet)
    puhtaat_linkit.append(linkki)

kotimaan_artikkelit = []
for linkki in lopulliset_kotimaan_linkit:
    html = rq.urlopen(linkki).read().decode("utf8")
    soup = BeautifulSoup(html, "html.parser")
    teksti = soup.article.find_all("p")  # teksti = list of paragraphs
    kappaleet = []
    for kappale in teksti:
        kappaleet.append(kappale.get_text())
    kappaleet = " ".join(kappaleet)
    kotimaan_artikkelit.append(kappaleet)

ulkomaan_artikkelit = []
for linkki in lopulliset_ulkomaan_linkit:
    html = rq.urlopen(linkki).read().decode("utf8")
    soup = BeautifulSoup(html, "html.parser")
    teksti = soup.article.find_all("p")
    kappaleet = []
    for kappale in teksti:
        kappaleet.append(kappale.get_text())
    kappaleet = " ".join(kappaleet)
    ulkomaan_artikkelit.append(kappaleet)



links_and_articles = list(zip(puhtaat_linkit, artikkelit))


# make a vocabulary from the words in the articles, tokenize and stem
# first turn list of articles into a string

poistettavat_sanat = ['sähköpostiisi', 'saat', 'ylen', ",", '.', "on", "ja", "-", "että", "ei"]
vocabulary_0 = " ".join(artikkelit)
vocabulary_tokenized = nltk.word_tokenize(vocabulary_0)
vocabulary_stemmed = [stemmer.stem(word) for word in vocabulary_tokenized if word not in poistettavat_sanat]
vocabulary_stemmed2 = set(vocabulary_stemmed)


artikkelit_tokenized = [nltk.word_tokenize(article) for article in artikkelit]
artikkelit_stemmed = [[stemmer.stem(word.lower()) for word in article if word not in poistettavat_sanat] for article in artikkelit_tokenized]


def add_vector(a, b):
    for i, x in enumerate(b):
        a[i] += x

def normalize(a):
    total = math.sqrt(sum(x ** 2 for x in a))
    return [x / total for x in a]


d = 1000     # size of the index and context vectors
m = 10       # number of non-zero components in index vectors


index_vector = {word: [0] * d for word in vocabulary_stemmed2}
artikkelit_stemmed_tuple = tuple(tuple(x) for x in artikkelit_stemmed)
document_vectors = {item: [0.0] * d for item in artikkelit}

# random indexing for the index and document vectors

for word in vocabulary_stemmed:
    random_positions = list(range(0, d))
    random.shuffle(random_positions)
    for i in random_positions[:m]:
        index_vector[word][i] = 1

for fileid in document_vectors:
    random_positions = list(range(0, d))
    random.shuffle(random_positions)
    for i in random_positions[:m]:
        document_vectors[fileid][i] = 1


# document vector making

for i in range(len(artikkelit_stemmed_tuple)):
    for word in artikkelit_stemmed_tuple[i]:
        add_vector(document_vectors[artikkelit[i]], index_vector[word])


# clustering


n_clusters = 2  # number of clusters
vectors = [np.array(normalize(document_vectors[w])) for w in artikkelit]
clusterer = KMeansClusterer(n_clusters, cosine_distance, avoid_empty_clusters=True, repeats=10)
clusters = clusterer.cluster(vectors, assign_clusters=True, trace=False)

cluster_docs = [set() for i in range(0, n_clusters)]
for i in range(0, len(clusters)):
    cluster_docs[clusters[i]].add(artikkelit[i])
for i in range(0, n_clusters):
    print("\nCluster #{:d}: {:s}".format(i, "\n ".join(cluster_docs[i])))

for item in cluster_docs:
    print(len(item))

print("\n")
purity_calculation()


def purity_calculation():
    kotimaa_cluster1 = 0
    ulkomaa_cluster1 = 0
    kotimaa_cluster2 = 0
    ulkomaa_cluster2 = 0
    for item in cluster_docs[0]:
        if item in kotimaan_artikkelit:
            kotimaa_cluster1 += 1
        elif item in ulkomaan_artikkelit:
            ulkomaa_cluster1 += 1
    for item in cluster_docs[1]:
        if item in kotimaan_artikkelit:
            kotimaa_cluster2 += 1
        elif item in ulkomaan_artikkelit:
            ulkomaa_cluster2 += 1
    print(kotimaa_cluster1)
    print(ulkomaa_cluster1)
    print(kotimaa_cluster2)
    print(ulkomaa_cluster2)


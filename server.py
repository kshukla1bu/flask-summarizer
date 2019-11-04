# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:44:58 2019

@author: Kaushal
"""

from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

app = Flask(__name__)

def make_sentences(textdata):
    article = textdata.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences
    
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
    
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

@app.route('/')
def index():
    return render_template('Summarize.html');
    
@app.route('/getfile', methods=['POST'])
def submit_text():
    stop_words = stopwords.words('english')
    summarize_text = []
    text = request.form.get("text")
    sentences = make_sentences(text)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(5):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    new = "" 
     
    for x in summarize_text: 
        new += x
        new += ". "

    return render_template('Result.html', text = new)

    
if __name__ == '__main__':
    app.run()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:05:28 2024

@author: isabel_orlando
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def perform_lda(text_data, n_features=1000, n_components=10, n_top_words=20):
    # Set stopwords and punctuation
    swords = set(stopwords.words('english'))
    swords
    philo=['discrimination']
    morewords=['would', 'happen', 'also', 'dont', 'one', 'like', 'get', 'isnt', 
               'im', 'way', 'back', 'put, make', 'go', 'got', 'let', '\n', '“', '”', '’']
    lwt=['tonight', 'show', 'time', 'something', 'come', 'seem', 'may', 'give',
         'well', 'get', 'yeah', 'yeh', 'theyre', 'their', 'there', 'every', 'he',
         'she', 'they', 'them', 'though', 'that', 'say', 'even', 'thing', 'make', 
         'that', 'could', 'see', 'your']
    #alm=['alllivesmatter']
    #swords.update(morewords)
    swords.update(philo)
    puncs = string.punctuation

    # Prepare text data
    processed_texts = []
    for text in text_data:
        # Tokenize and clean text
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token.translate(str.maketrans('', '', puncs)) for token in tokens]
        tokens = [token for token in tokens if token not in swords and token.isalpha()]
        processed_texts.append(' '.join(tokens))  # Create a list of processed texts
    
    #was 0.95 and 2 for large df, 1.0 and 1 for small data set
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(processed_texts)

    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50.0, random_state=0)
    lda.fit(tf)

    
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, tf_feature_names, n_top_words, n_components, 'Topics in LDA model')

def plot_top_words(model, feature_names, n_top_words, n_components, title):
    rows = int(n_components / 5) + (n_components % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig('/Users/isabel_orlando/Desktop/GOVT3282/Previous files/topics.png', 
                dpi=300, bbox_inches='tight')
    plt.show()





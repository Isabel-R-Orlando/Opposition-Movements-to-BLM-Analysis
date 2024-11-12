#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:13:29 2024

@author: isabel_orlando
"""

#LDA topic modeling 
import matplotlib.pyplot as plt 
import pandas as pd
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 

#%%

#we are going to set some abritrary limits here 
n_features=1000
n_components=10
n_topwords=20
batch_size=128

#for processing text 
puncs=string.punctuation 
swords=set(stopwords.words('english'))

#%%
#generate one massive document per group 
#or interrows method from somewhere probs youtubeapi
x=''
mwdf.columns
for r in mwdf['transcript']:
    x+=''+r

#make each word into a list of just the words
x2=word_tokenize(x)

e1=[letter.lower() for letter in x2]
e2=[letter.translate(str.maketrans('','',puncs)) for letter in e1]
e3=[token for token in e2 if token not in swords]

ndf=pd.DataFrame({'newc':e3})
#%%
#set up a function to visualize topics 

def plot_top_words(model, feature_names, n_top_words, title):
    path='/Users/isabel_orlando/Desktop/GOVT3282'
    name='blurby'
    rows=int(n_components/5)
    fig, axes = plt.subplots(rows, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
            fig.suptitle(name+' - '+title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    fig.savefig(path+name+title+'.png', dpi=200)
    
#we are going to use just one topic model algorithm with just one 
#set number of toppics 

#%%

tf_vectorizer=CountVectorizer(
    max_df=0.95, min_df=2,
    max_features=n_features,
    stop_words='english',
    )


tf=tf_vectorizer.fit_transform(ndf['newc'])

lda=LatentDirichletAllocation(
    n_components=n_components,
    max_iter=5,
    learning_method='online',
    learning_offset=50.0,
    random_state=0)

lda.fit(tf)

tf_feature_names=tf_vectorizer.get_feature_names_out()

plot_top_words(lda, tf_feature_names, n_topwords, 'Topics in LDA model')







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:34:01 2024

@author: isabel_orlando
"""
#a is file name and b is column of text name
#https://www.datacamp.com/tutorial/text-analytics-beginners-nltk
def sentimentanalysis(a,b):


    #let's look at rudimentary text analysis 
    #reducing words to stems, finding commonly used words etc
    import pandas as pd
    import seaborn.objects as so
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize 
    from nltk.stem import PorterStemmer
    from wordcloud import WordCloud
    #nltk.download('punkt')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    from nltk.corpus import stopwords
    
    from nltk.tokenize import word_tokenize
    
    from nltk.stem import WordNetLemmatizer
    import os
    
    # download nltk corpus (first time only)
    import nltk
    
    nltk.download('all')
    
    #%%
    os.chdir("/Users/isabel_orlando/Desktop/GOVT3282")
    df=pd.read_csv(a)
    df
    
        #%%
        
    def preprocess_text(text):
        
        swords=set(stopwords.words('english'))
        swords
        morewords=['would', 'happen', 'also', 'dont', 'one', 'like', 'get', 'isnt', 
               'im', 'way', 'back', 'put, make', 'go', 'got', 'let']
        other=[ 'something', 'seem', 'may', 'give',
         'well', 'get', 'yeah', 'yeh', 'theyre', 'their', 'there', 'every', 'he',
         'she', 'they', 'them', 'though', 'that', 'say', 'even', 'thing', 'make', 
         'that', 'could', 'see', 'your']
        #alm=['alllivesmatter']
        swords.update(morewords)
        swords.update(other)
        #swords.update(alm)
        swords
        
        tokens=word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in swords]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text
    
    df[b]= df[b].apply(preprocess_text)
    df
    
    
    
    #%%
    analyzer = SentimentIntensityAnalyzer()
    
    # create get_sentiment function
    
    def get_sentiment(text):
    
        scores = analyzer.polarity_scores(text)
    
        #sentiment = 1 if scores['pos'] > 0 else 0
        
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
    
        return sentiment
    
    # apply get_sentiment function
    
    df['sentiment'] = df[b].apply(get_sentiment)
    
    return df
    
    
    


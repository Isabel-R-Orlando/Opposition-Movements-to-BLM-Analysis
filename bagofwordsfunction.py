#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:57:37 2024

@author: isabel_orlando
"""

#a is vector, b is choice of number of top words
def bagofwords(a,b,c,d):
    
    
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

    puncs=string.punctuation 
#let's make one large transcript document 
    text=''
    #a is the vector that contains the transcripts 
    for i in a:
        text+=i

#let's lower the cases
    t1=text.lower()
    t2=t1.translate(str.maketrans('','', puncs))
#create dictionary of stopwords from existing dictionary 
    swords=set(stopwords.words('english'))
    swords
    morewords=['would', 'happen', 'also', 'dont', 'one', 'like', 'get', 'isnt', 
               'im', 'way', 'back', 'put, make', 'go', 'got', 'let', "'", '"', '\n', '“', '”', '’']
    lwt=['tonight', 'show', 'time', 'something', 'come', 'seem', 'may', 'give',
         'well', 'get', 'yeah', 'yeh', 'theyre', 'their', 'there', 'every', 'he',
         'she', 'they', 'them', 'though', 'that', 'say', 'even', 'thing', 'make', 
         'that', 'could', 'see', 'your']
    #alm=['alllivesmatter']
    swords.update(morewords)
    #swords.update(lwt)
    swords
    #swords.update(alm)

#tokenize (separate document into obs of single words)
    t3=word_tokenize(t2)

#remove stop words
    t4=[token for token in t3 if token not in swords]
#now stemming (reducing words to their stems)
    porter = PorterStemmer()
    t5=[porter.stem(x) for x in t4]
    t6=nltk.FreqDist(t5)
    wdf=pd.DataFrame(t6.items(),columns=['word','frequency'])


    #have to be equal to an argument
    '''if b: 
        wdf['account']=b
    else: 
        wdf['account']=None'''
        
    wdfhigh=wdf.sort_values('frequency', ascending=False)
#top 25 words 
    wdfhigh2=wdfhigh.head(b)

    p=(
       so.Plot(wdfhigh2, y='word',x='frequency')
       .add(so.Bars(color='salmon'))
       .label(title='Top '+str(b)+' words by: '+c,
              y='Words', x = 'Frequency')
       )
    p.save('Top '+str(b)+' words '+c+'.png', dpi=200)
    #p.show()
#%%
    import os
    tpath1='Top '+str(b)+' words '+c+'.png'
    tpath1=str(tpath1)
    os.system('open '+tpath1)
    
    wc=WordCloud(background_color='White',
             max_words=d,
             width=1000, height=1000)

    wc.generate_from_frequencies(t6)
    wc.to_file('wordcloud for '+c+'.png')
    #wb.open('wordcloud for '+c+'.png')
              
    

#%%

#wordcloud, go to the documentation page 

    '''wc=WordCloud(background_color='White',
             max_words=d,
             width=1000, height=1000)

    wc.generate_from_frequencies(t6)
    wc.to_file('wordcloud for '+c+'.png')'''
    
    import webbrowser as wb
    wb.open('Top'+str(b)+' words '+c+'.png')
    wb.open('wordcloud for '+c+'.png')
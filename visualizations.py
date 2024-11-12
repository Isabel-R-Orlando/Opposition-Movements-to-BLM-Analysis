#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:36:31 2024

@author: isabel_orlando
"""

import os
import webbrowser as wb
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd

#%%

os.chdir("/Users/isabel_orlando/Desktop/GOVT3282")

os.listdir()

df=pd.read_csv("ALM_cleanedfile.csv")

df
df.text[1]

#retweet count 
df.retweet_count[1]
lretweets=df['retweet_count'].max()
lretweets
minretweets=df['retweet_count'].median()
minretweets
maxlength=df['length'].max()
maxlength
medianlength=df['length'].median()
medianlength
likes=df['like_count'].mode()
likes



#making variable of retweets
retweets=df.retweet_count
retweets

#%%
text=''
for i in df.text:
    text+=i

from bagofwordsfunction import bagofwords
bagofwords(text, 25, 'Opposition Movements to BLM', 150)

#%%
#length of tweets
p1=(
    so.Plot(df[df['length']<150], 'length')
    .add(so.Bars(color='salmon'),so.Hist(bins=20))
    .label(x='Length of Tweet in characters', 
           title='Length of Tweets')
    )
p1.show()
p1.save('p1-length.png', dpi=200)
#%%
#number of tweets per month 

df['date'] = pd.to_datetime(df['date'])

df['year_month'] = df['date'].dt.to_period('M')

df['count'] = 1  
monthly_tweets = df.groupby('year_month')['count'].sum().reset_index()

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=monthly_tweets, x='year_month', y='count', 
                 palette='coolwarm')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Number of Tweets Per Month')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Tweets')

textstr = "George Floyd's murder occurred on May 25, 2020"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.52, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


plt.savefig('/Users/isabel_orlando/Desktop/GOVT3282/Data Project/number_of_tweets_per_month.png', 
            dpi=300, bbox_inches='tight')
plt.show()

#%%

#most tweets got less than 10 likes
p4=(
    so.Plot(df[df['like_count']<10], 'like_count')
    .add(so.Bars(color='salmon'),so.Hist(bins=20))
    .label(x='Likes', 
           title='Likes of Tweets under 10')
    )
p4.show()
p4.save('likes.png', dpi=200)


#retweets histogram-most people get 0 retweets 
p2=(
    so.Plot(df[df['retweet_count']<30], 'retweet_count')
    .add(so.Bars(color='salmon'),so.Hist(bins=20))
    .label(x='Retweets', 
           title='Retweets per tweet')
    )

p2.show()
p2.save('p2-retweets.png', dpi=200)

p12 = (
    so.Plot(df, x='like_count', color='retweet_count')
    .add(so.Bar(), so.Hist(), so.Dodge())
    .label(title='Distribution of Likes and Retweets')
)

p12.show()

#%%
#another attempt - probably not showing anything of value though
plt.figure(figsize=(10, 6))

# Create a histogram for 'like_count'
plt.hist(df['like_count'], bins=50, alpha=0.5, label='Likes')

plt.title('Distribution of Likes and Retweets')
plt.xlabel('Like Count')
plt.ylabel('Frequency')
plt.legend()

plt.show()

#%%



#%%

from sentimentanalysis import sentimentanalysis
df=sentimentanalysis('ALM_cleanedfile.csv', 'text')


#%%
#essentially, negative is indicating that the words chosen are more harsh and negative, 
#neutral means it has more neutral langugage
#positive means thanking or complimentary or optimistic
#doesn't tell us support or condemnation for the movement ALM 
#we can assume more neg tweets are ANTI ALM while many pos are pro ALM (not all) - but yes
#neutral is not clear, generally pro ALM or ? 

sentiment_order = ['negative', 'neutral', 'positive']
sentiment_counts = df['sentiment'].value_counts().reindex(sentiment_order).reset_index()
sentiment_counts.columns = ['sentiment', 'count']

colors = ['coral', 'burlywood', 'darkkhaki']  # Colors corresponding to Negative, Neutral, Positive

plt.figure(figsize=(8, 4))
plt.bar(sentiment_counts['sentiment'], sentiment_counts['count'], color=colors)
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets by Sentiment')
plt.xticks(range(len(sentiment_counts['sentiment'])), ['negative or anti-ALM', 'neutral', 'positive or pro-ALM'])


plt.savefig('/Users/isabel_orlando/Desktop/GOVT3282/Data Project/sentiments.png', 
            dpi=300, bbox_inches='tight')
plt.show()

#making sure the values align with my graph 
neg = df['sentiment'].value_counts().get('negative', 0)
neg
neut = df['sentiment'].value_counts().get('neutral', 0)
neut
#%%

df_random_50=df.sample(n=50)

df_random_50.to_excel('random_50_cells.xlsx', index=False)

df.to_excel('sentiments.xlsx')


#%%
#topic modeling 
dfog=pd.read_csv("ALM_cleanedfile.csv")
os.chdir("/Users/isabel_orlando/Desktop/GOVT3282/Previous files")
#fightingwords using the #bluelivesmatter and #allbuildingsmatter


#contains 18000 tweets
blue_df = dfog[dfog['text'].str.contains('#bluelivesmatter', case=False, na=False)]
blue_df

from topicsfunction import perform_lda
perform_lda(blue_df['text'], n_features=1000, n_components=10, n_top_words=20)
#%%
#contains 1220 tweets for buildings people
build_df=dfog[dfog['text'].str.contains('#allbuildingsmatter', case=False, na=False)]

from topicsfunction import perform_lda
perform_lda(build_df['text'], n_features=1000, n_components=5, n_top_words=20)

#!!! WOW!!!!
#%%
#whitelivesmatter?
white_df=dfog[dfog['text'].str.contains('#whitelivesmatter', case=False, na=False)]
white_df

from topicsfunction import perform_lda
perform_lda(white_df['text'], n_features=1000, n_components=10, n_top_words=20)

text=''
for i in white_df.text:
    text+=i

from bagofwordsfunction import bagofwords
bagofwords(text, 25, 'White Lives Matter', 150)

#%%

#fighting words

from convokit import Corpus, FightingWords

#resetting my blue_df to be easier to handle 
blue_df.reset_index(drop=True, inplace=True)
blue_df = blue_df[['text']]
blue_df['name'] = 'blue_df'
blue_df = blue_df[['name', 'text']]

blue_df

#resetting build_df as well 
build_df.reset_index(drop=True, inplace=True)
build_df = build_df[['text']]
build_df['name'] = 'build_df'
build_df = build_df[['name', 'text']]

build_df

#%%
#process test into new data frames that remove stop words 
bluby = process_text(blue_df, 'text')
builby= process_text(build_df, 'text')

#%%
simpleuttdf=pd.concat([bluby, builby])

#adds together the data frames with video id, name and transcript, (name distinguishes the two)
#what is the reply to even doing? 
simpleuttdf['reply to']='None'
simpleuttdf['conversation id']='None'
#checking columsn for information 
#index resets at different youtuber 
#this is just adding a random counter called Time, idt it does antyhing 
simpleuttdf['time']=range(len(simpleuttdf))


simpleuttdf.columns 

#index gives unique identifier for each one per name 
simpleuttdf['index']=simpleuttdf.index

#changing the name of columns, index= timestamp, convo=vidid, speaker=name, 
#text=transcript, reply to = reply to, nothing = time 
simpleuttdf.columns=['speaker',
                     'text',
                     'reply_to',
                     'nothing',
                     'conversation_id',
                     'timestamp']

#list of the indexes for mark, then food ranger 
ids=list(simpleuttdf.index)

#just repeating another column that does same as timestamp but isn't the INDEX
simpleuttdf=simpleuttdf.reset_index()

#now creating a column at end called 'id' that has the index which was timestamp? 
simpleuttdf['id']=ids

#creating a corpus 
newcorpus=Corpus.from_pandas(simpleuttdf)
newcorpus.print_summary_stats()

#creating an empty set with no repeats 
tset=set()
for utt in newcorpus.iter_utterances():
    tset.add(utt.speaker.id)
    utt.meta['speaker']=utt.speaker.id
    
tset

newcorpus.iter_utterances()

#we are ready to impleemnt fighting words 
fw=FightingWords(ngram_range=(1,1))

fw.fit(
       newcorpus,
       class1_func=lambda utt: utt.meta['speaker']=='blue_df',
       class2_func=lambda utt: utt.meta['speaker']=='build_df')

vis=fw.summarize(newcorpus,
                 plot=True,
                 class1_name='#BlueLivesMatter',
                 class2_name='#AllBuildingsMatter')


#%%



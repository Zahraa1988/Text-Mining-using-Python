#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import require libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download(['stopwords',
              'punkt',
              'wordnet',
              'omw-1.4',
              'vader_lexicon'])
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


reviews = pd.read_csv('tourist_accommodation_reviews_org.csv')


# In[3]:


# display the first six rows
reviews.head()


# In[4]:


# display the last six rows
reviews.tail()


# In[5]:


reviews.describe()


# In[6]:


reviews.info()


# In[7]:


reviews.shape


# In[8]:


# check anu missing values
count= reviews.isnull().sum().sort_values(ascending=False)
percentage = ((reviews.isnull().sum()/len(reviews))*100).sort_values(ascending=False)
missing_data = pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])
print ('Count and Percentage of missing values for the columns:')
missing_data


# In[9]:


reviews['Hotel/Restaurant name'].unique()


# In[10]:


# chose 15 restaurants and 15 cafes
reviews = reviews[reviews["Hotel/Restaurant name"].isin(['Golden Paradise Restaurant','Ali Baba Restaurant',
                                                         'Dada Yura Restaurant','Albatross Cafe',
                                                         'Cafe del Mar Phuket','East Restaurant','Cafe Java',
                                                         'Archee Restaurant','Cafe de Bangtao','Capri Noi Restaurant',
                                                         'Coral Restaurant','3 Spices Restaurant','Atsumi Raw Cafe',
                                                         'Delish Cafe','Cafe Siam Breakfast Cafe','Audy Restaurant',
                                                         '2gether Restaurant','Gallery Cafe by Pinky','Madras Cafe',
                                                         'Rider Cafe','CatFish Cafe','Chai Thaifood Restaurant',
                                                         'Arabia Restaurant','Elephant Cafe by Tan','Acqua Restaurant',
                                                         'Madras Cafe 2 & Guest House','Eightfold Restaurant','Marriott Cafe',
                                                         'i-Kroon Cafe','Baan Noy Restaurant'])]


# In[11]:


reviews['Hotel/Restaurant name'].unique()


# In[12]:


reviews


# In[13]:


#  Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()


# In[14]:


# create four columns(compound,neg,neu,pos)
reviews['compound']= [sentiment.polarity_scores(review)['compound'] for review in reviews ['Review']]
reviews['neg']= [sentiment.polarity_scores(review)['neg'] for review in reviews ['Review']]
reviews['neu']= [sentiment.polarity_scores(review)['neu'] for review in reviews ['Review']]
reviews['pos']= [sentiment.polarity_scores(review)['pos'] for review in reviews ['Review']]


# In[15]:


reviews.head()


# In[16]:


reviews[['compound','neg','neu','pos']].describe()


# In[17]:


# review compound distrbution
sns.histplot(reviews['compound'])


# In[18]:


# review positive distrbution
sns.histplot(reviews['pos'])


# In[19]:


# review negative distrbution
sns.histplot(reviews['neg'])


# In[20]:


(reviews['compound']<=0).groupby(reviews['Hotel/Restaurant name']).sum()


# In[21]:


percent_negative = pd.DataFrame((reviews['compound']<=0).groupby(reviews['Hotel/Restaurant name']).sum()
                                /reviews['Hotel/Restaurant name'].groupby(reviews['Hotel/Restaurant name']).count()*100,
                                columns = ['% negative reviews']).sort_values(by='% negative reviews')
percent_negative


# In[22]:


sns.barplot(data=percent_negative,x='% negative reviews',y= percent_negative.index,color='purple')


# In[23]:


stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)


# In[24]:


def preprocess_text(text):
    tokenized_document= nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)
    cleaned_tokens =[word.lower() for word in tokenized_document if word.lower() not in stop_words]
    stemmed_text = [nltk.stem.PorterStemmer().stem(word) for word in cleaned_tokens]
    return stemmed_text


# In[25]:


reviews['processed_review']= reviews['Review'].apply(preprocess_text)
reviews_positive_subset= reviews.loc[(reviews['Hotel/Restaurant name']=='Dada Yura Restaurant')
                                    & (reviews['compound']>0),:]
reviews_negative_subset= reviews.loc[(reviews['Hotel/Restaurant name']=='Dada Yura Restaurant')
                                    & (reviews['compound']<=0),:]
reviews_positive_subset.head()


# In[26]:


neg_tokens =[word for review in reviews_negative_subset['processed_review'] for word in review]

wordcloud= WordCloud(background_color='white').generate_from_text(' '.join(neg_tokens))


plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[27]:


pos_tokens =[word for review in reviews_positive_subset['processed_review'] for word in review]

wordcloud= WordCloud(background_color='white').generate_from_text(' '.join(pos_tokens))


plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[28]:


#positive words
from nltk.probability import FreqDist
pos_freqdist = FreqDist(pos_tokens)
pos_freqdist.tabulate(10)


# In[29]:


#negative words
from nltk.probability import FreqDist
neg_freqdist = FreqDist(neg_tokens)
neg_freqdist.tabulate(10)


# In[30]:


pos_freqdist.plot(30)


# In[31]:


neg_freqdist.plot(30)


# In[32]:


reviews['processed_review']= reviews['Review'].apply(preprocess_text)
reviews_positive_subset= reviews.loc[(reviews['Hotel/Restaurant name']=='Cafe del Mar Phuket')
                                    & (reviews['compound']>0),:]
reviews_negative_subset= reviews.loc[(reviews['Hotel/Restaurant name']=='Cafe del Mar Phuket')
                                    & (reviews['compound']<=0),:]
reviews_positive_subset.head()


# In[33]:


neg_tokens =[word for review in reviews_negative_subset['processed_review'] for word in review]

wordcloud= WordCloud(background_color='white').generate_from_text(' '.join(neg_tokens))


plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[34]:


pos_tokens =[word for review in reviews_positive_subset['processed_review'] for word in review]

wordcloud= WordCloud(background_color='white').generate_from_text(' '.join(pos_tokens))


plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[35]:


#positive words
from nltk.probability import FreqDist
pos_freqdist = FreqDist(pos_tokens)
pos_freqdist.tabulate(10)


# In[36]:


#negative words
from nltk.probability import FreqDist
neg_freqdist = FreqDist(neg_tokens)
neg_freqdist.tabulate(10)


# In[37]:


pos_freqdist.plot(30)


# In[38]:


neg_freqdist.plot(30)


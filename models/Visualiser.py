#!/usr/bin/env python
# coding: utf-8

# In[6]:


import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


# In[9]:


# Sentiment Graph Creators
def get_multiple_graphs(reviews, label):
    row_start = int(input(f'Please enter starting point of range of max recommended length 20 between 0 and {len(reviews)} e.g 0: '))
    row_end = int(input(f'Please enter end point of range of max recommended length 20 between 0 and {len(reviews)} e.g 13: '))
    rows = (row_end-row_start)//3
    ax = reviews.iloc[row_start:row_end][label].unstack(level=0).plot(kind='bar', subplots=True, rot=0, 
                                                                          figsize=(9, 7), layout=(rows, 3), 
                                                    title="Number of Negative, Neutral and Positive Reviews")
    plt.tight_layout()

def get_course_graph(reviews, label):
    course_code = input(f'Please enter a course code e.g. "CS 115": ')
    ax = reviews.loc[course_code][label].plot(kind='bar', subplots=True, rot=0, 
                                                                          figsize=(5, 5), 
                                                    title="Number of Negative, Neutral and Positive Reviews")
    plt.tight_layout()


# In[7]:


# WordCloud Creator
def get_wordcloud(df, label):
    stopwords = set(STOPWORDS)
    course_code = input('Please enter a course code: ')
    if course_code in df.index:
        for ele in df.loc[course_code].index:
            text = df.loc[course_code, ele][label]
            wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
            plt.figure( figsize=(5,5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"{ele} Reviews for {course_code} Course")
            plt.tight_layout()
    else:
        print('Please enter a valid course code.')


# In[8]:


# Frequency of words Graph Creator
def word_frequency_graph(df, label):
    course_code = input('Please enter a course code: ')
    i = 0
    for ele in df.loc[course_code][label]:
        freq = nltk.FreqDist(ele)
        for key,val in freq.items():
            df.loc[i, 'frequency'] = (str(key) + ':' + str(val))
        freq.plot(20, cumulative=False,title=f"{df.loc[course_code].index[i]} Reviews for {course_code} Course")
        i += 1


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


# In[117]:


# This additional download includes pre-trained models, corpora, and other 
# resources that NLTK uses to perform various NLP tasks. 

# nltk.download('all');


# ### Method 1 (using Apply)

# In[67]:


# Text normalising: remove punctuations, lower case.
def text_normaliser(text):
    text = str(text).lower()
    text = str(text).translate(str.maketrans("", "", ".,-;!?:"))
    return text


# In[68]:


# Word normalising: Stemming and Lemmatising normalising: remove punctuations, lower case.

ps=PorterStemmer()
def nltk_stemmer(text):
    # Word normalising using Stemming | Using nltk PorterStemmer for Stemming
    text = ps.stem(text)
    return text

wnl = WordNetLemmatizer()
def nltk_lemmatiser(text):
    # Word normalising using Lemmatisation (converting words to their canonical forms) | Using nltk WordNetLemmatizer for lemmatisation
    text = wnl.lemmatize(text)
    return text


# In[69]:


# Removing stop words

sr= stopwords.words('english')
def stop_words_remover(text):
    # Remove Stop Words
    text = [word for word in text.split(' ') if word not in sr]
    text = " ".join(text)
    return text


# In[70]:


# Tokenising: breaking down the text into individual words or tokens

def tokeniser(text):
    # Tokenise Reviews: involves breaking down the text into individual words or tokens
    # Split the text into individual words and punctuation marks.
    text = [t for t in str(text).split()]
    return text

# Tokenization is typically performed using NLTK's built-in `word_tokenize` function, which 
# can split the text into individual words and punctuation marks.


# In[105]:


# Overall preprocessing function, excluding embedding

def text_preprocessor(text, word_normalising='lemmatisation', tokenise=False):
    text = text_normaliser(text)
    if word_normalising == 'lemmatisation':
        text = nltk_lemmatiser(text)
    elif word_normalising == 'stemming':
        text = nltk_stemmer(text)
    else:
        message = 'please enter a valid word_normalising technique ("lemmatisation" or "stemming")'
        return message
    text = stop_words_remover(text)
    if tokenise == True:
        text = tokeniser(text)
    elif tokenise == False:
        text = text
    return text


# In[112]:


# w2v embedding from Gensim Library

def w2v_embedding(df, label):
    all_reviews = [sentence.split(' ') for sentence in df[label] if sentence != 'nan']
    w2v = Word2Vec(all_reviews, vector_size=100, window=5, min_count=1) #no workers and negative parameters included
    embedded_vector = w2v.wv.vectors
    return embedded_vector



import gensim
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))   #deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]




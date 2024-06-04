#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries for LDA
import re
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

# from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Import libraries for NMF
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#Import libraries for ETS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

#Import libraries for ATS
import torch                
from transformers import AutoTokenizer, AutoModelWithLMHead

#Import Preprocessor library
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from models import Preprocessor

#Import normal libraries
import pandas as pd
import matplotlib.pyplot as plt

# In[2]:


### Topic Modelling:
# LDA (Latent Dirichlet Allocation)
# NMF (Non-Negative Matrix Factorization)


# In[3]:


# Functions
def make_bigrams(texts, bigram_mod_):
    return [bigram_mod_[doc] for doc in texts]

def make_trigrams(texts, trigram_mod_):
    return [trigram_mod_[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_corpus(data):
    data_words = list(Preprocessor.sent_to_words(data))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # Remove Stop Words
    data_words_nostops = Preprocessor.remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    data_words_trigrams = make_bigrams(data_words_nostops, trigram_mod)
    # # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    id2word = corpora.Dictionary(data_lemmatized)  
    # Create Corpus 
    texts = data_lemmatized  
    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]  
    
    return id2word, corpus, data_lemmatized


# In[4]:


def get_lda_coherence_chart(data):
    id2word, corpus, data_lemmatized = get_corpus(data)
#     topic_nums = list(np.arange(5, 76, 5))
    topic_nums = list(np.arange(1, 25, 1))
    coherence_scores = []
    for num in topic_nums:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

        cm = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

        coherence_scores.append(round(cm.get_coherence(), 5))
    scores = {'Number of Topics': topic_nums, 'Coherence Score': coherence_scores}
    # Get the number of topics with the highest coherence score
#     return scores

    # Plot the results
    fig = plt.figure(figsize=(15, 7))

    plt.plot(
        topic_nums,
        coherence_scores,
        linewidth=3,
        color='#4287f5'
    )

    plt.xlabel("Topic Num", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title('Coherence Score for LDA Model by Topic Number', fontsize=16)
    #  - Best Number of Topics: {}'.format(best_num_topics)
    plt.xticks(np.arange(2, max(topic_nums) + 1, 2), fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()


# In[5]:


def get_lda_topics(data, topic_number):
    id2word, corpus, data_lemmatized = get_corpus(data)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic_number, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    df = pd.DataFrame(columns=['Number','LDA Topics'])
    for idx, topic in lda_model.show_topics(formatted=False, num_topics=topic_number):
        df.loc[idx, 'Number'] = f"Topic {idx + 1}"
        df.loc[idx, 'LDA Topics'] =  " | ".join([w[0] for w in topic])
#     return lda_model.print_topics()
    return df


# In[12]:


def get_dominant_topic(data, topic_number):
    id2word, corpus, data_lemmatized = get_corpus(data)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic_number, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    pd.set_option("display.precision", 2)

    # Initialize lists to store the dominant topic and its probability for each comment
    dominant_topics = []
    dominant_probabilities = []

    # Iterate through each comment in the corpus
    for doc in corpus:
        # Get the topic distribution for the current comment
        topic_distribution = lda_model.get_document_topics(doc)

        # Find the dominant topic (topic with the highest probability)
        dominant_topic, dominant_probability = max(topic_distribution, key=lambda x: x[1])

        # Append the dominant topic and its probability to the lists
        dominant_topics.append(dominant_topic+1)
        dominant_probabilities.append(dominant_probability)

    return dominant_topics, dominant_probabilities



# In[6]:


def visualise_lda(data, topic_number):
    id2word, corpus, data_lemmatized = get_corpus(data)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic_number, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    return vis


# In[7]:


def get_nmf_topics(array, number_topics):
    data_nmf = [' '.join(array)]
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
    X = vectorizer.fit_transform(data_nmf)
    words = np.array(vectorizer.get_feature_names_out())
    # Applying Non-Negative Matrix Factorization
    nmf = NMF(n_components=number_topics, solver="mu")
    W = nmf.fit_transform(X)
    H = nmf.components_
    df = pd.DataFrame(columns=['Number','NMF Topics'])
    for i, topic in enumerate(H):
        df.loc[i, 'Number'] = f"Topic {i + 1}"
        df.loc[i, 'NMF Topics'] =  " | ".join([str(x) for x in words[topic.argsort()[-10:]]])
        # print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))        
    return df


# In[8]:


### Text Summary


# In[9]:


def get_parser(text):
    parser = PlaintextParser.from_string(text,Tokenizer("english"))
    return parser


# In[10]:


# Extractive Text Summary: Luhn Summariser and LexRank Summariser

def get_extractive_summaries(text):
    summarizer_lex = LexRankSummarizer()
    parser = get_parser(text)
    summary = summarizer_lex(parser.document, 5)
    df = pd.DataFrame(columns=['Sentences','LexRank Summary', 'Luhn Summary'])
    i = 0
    df['Sentences'] = range(1,6)
    for sentence in summary:
        df.loc[i, 'LexRank Summary'] = str(sentence)
        i += 1
    
    summarizer_luhn = LuhnSummarizer()
    summary_1 = summarizer_luhn(parser.document,5)
    j = 0
    for sentence in summary_1:
        df.loc[j, 'Luhn Summary'] = str(sentence)
        j += 1
    return df


# In[11]:


# Abstractive Text Summary: Google T5 Model

def get_t5_summary(text):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')                        
    model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)       
    inputs = tokenizer.encode("summarize: " + text,
                          return_tensors='pt',
                          max_length=len(text),             
                          truncation=True)  
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)          
    summary = tokenizer.decode(summary_ids[0])  
    summary = summary.replace('<pad>', '')
    summary = summary.replace('</s>', '')
    return summary


# In[ ]:





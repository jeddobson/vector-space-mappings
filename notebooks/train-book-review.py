#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, json
from glob import glob

# core nltk
import nltk
from nltk.tokenize import word_tokenize

# gensim magic
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


# In[2]:


reviews = []
for line in open('../data/goodreads_reviews_spoiler.json', 'r'):
    reviews.append(json.loads(line))


# In[ ]:


# select just review text
reviews = [x['review_sentences'] for x in reviews]


# In[ ]:


# now extract just 
extracted_reviews = list()
for r in reviews:
    text = str()
    for j in r:
        text = text + " " + j[1]
    extracted_reviews.append(text)


# In[ ]:


reviews=list()
for r in extracted_reviews:
    tokens = word_tokenize(r)    
    tokens = [word.lower() for word in tokens]
    
    tmp_text=list()

    for word in tokens:
        if len(word) == 1:
            if word.isalpha == True:
                tmp_text.append(word)
        else:
             tmp_text.append(word)           
    tokens = tmp_text

    # now remove leading and trailing quotation marks,      
    # hyphens and  dashes
    tmp_text=list()
    drop_list = ['“','"','”','-','—']
    for word in tokens:
        if word[0] in drop_list:
            word = word[1:]
        if word[-1:] in drop_list:
            word = word[:-1]

    # catch any zero-length words remaining
        if len(word) > 0:
            tmp_text.append(word)
    tokens = tmp_text
    reviews.append(tokens)


# In[ ]:


# source documents
# dimension of feature vectors 
# max distance   
# number of times a word must appear to be included in vocab
# for parallelization

print("starting training...")
book_review_model = gensim.models.Word2Vec(
    reviews, 
    sg=0,           # sg=1 is use skip-gram, sg=0 is cbow 
    size=200,        
    window=15,     
    max_vocab_size=25000,
    min_count=2,    # increase to limit vocab and find fewer rare words
    workers=10,     
    iter=10)


# In[ ]:


print("saving output")
fp = open("../models/book-reviews-vectors.w2v",'wb')
book_review_model.wv.save(fp)


# In[ ]:


book_review_model.wv.most_similar("l")


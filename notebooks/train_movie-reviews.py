#!/usr/bin/env python
# coding: utf-8

import os, sys
from glob import glob

# core nltk
import nltk
from nltk.tokenize import word_tokenize

# gensim magic
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

# read reviews
import gzip
print("loading gzipped reviews...")
raw_text = gzip.open("../data/movie-reviews/movie-reviews.txt.gz",'rt').read()

print("preprocessing...")
# tokenize
tokens = word_tokenize(raw_text)    

# drop to lowercase
tokens = [word.lower() for word in tokens]
        
# *step two* (default): remove non-alpha characters,
# punctuation, and as many other "noise" elements as
# possible. If dealing with a single character word,    
# drop non-alphabetical characters. This will remove 
# most punctuation but preserve many words containing
# marks such as the '-' in 'self-emancipated'

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

# simulate documents
print("segmenting...")
review_docs = list()
segment_length = int(len(tokens)/1000)
for j in range(1000):
     segment = tokens[segment_length*j:segment_length*(j+1)]
     review_docs.append(segment)

# source documents
# dimension of feature vectors 
# max distance   
# number of times a word must appear to be included in vocab
# for parallelization

print("starting training...")
movie_review_model = gensim.models.Word2Vec(
    review_docs, 
    sg=0,           # sg=1 is use skip-gram, sg=0 is cbow 
    size=200,        
    window=15,     
    min_count=2,    # increase to limit vocab and find fewer rare words
    workers=10,     
    iter=10)

print("saving output")
fp = open("../models/movie-reviews-vectors.w2v",'wb')
movie_review_model.wv.save(fp)

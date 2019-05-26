#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from textblob import Word
import numpy as np


def main():
    eng = "english"
    swe = "swedish"
    stop_eng = stopwords.words(eng)
    stop_swe = stopwords.words(swe)
    stem_eng = SnowballStemmer(eng)
    stem_swe = SnowballStemmer(swe)
    
    df = pd.read_csv('text_source/swe.txt', sep="\t", names=[eng, swe])
    
    df[eng] = pre_proc(df, eng, stop_eng, stem_eng)
    df[swe] = pre_proc(df, swe, stop_swe, stem_swe) 
    
    #df[eng] = df[eng].apply(lambda x: x.lower())
    #df[swe] = df[swe].apply(lambda x: x.lower())

    eng_df = df[eng]
    swe_df = df[swe]
    tf_eng = (eng_df).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    tf_eng.columns = ['words','tf']
    for i,word in enumerate(tf_eng['words']):
        tf_eng.loc[i, 'idf'] = np.log(df.shape[0]/(len(df[df[eng].str.contains(word)])))

    tf_eng['tfidf'] = tf_eng['tf'] * tf_eng['idf']
    #tf_swe = (swe_df).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    
    
    #tf_swe.columns = ['words','tf']
    print(tf_eng)

    
    #print(eng_df.to_string())
    #print(swe_df.to_string())


def pre_proc(df, name, stop=None, stemmer=None):
    pdat = df[name]
    # Lower all chars
    pdat = pdat.apply(lambda x: x.lower())
    # Remove punctations
    pdat = pdat.str.replace('[^\w\s]', '')
    # Remove stop words and digits
    if stop:
        pdat = pdat.apply(lambda x: " ".join([w for w in str(x).split() if w not in stop and not w.isdigit()]))
    # Remove rare words
    lowfreq = pd.Series(' '.join(pdat).split()).value_counts()[-1900:]
    lowfreq = list(lowfreq.index)
    pdat = pdat.apply(lambda x: " ".join(x for x in x.split() if x not in lowfreq))
    # Stem
    if stemmer:
        pdat = pdat.apply(lambda x: " ".join([stemmer.stem(w) for w in x.split()]))
    return pdat


#TODO
def tfidf():
    return True


if __name__ == "__main__":
    main()
    

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from nltk.corpus import stopwords
from stopwords import stopwords_eng
from nltk.stem.snowball import SnowballStemmer
from textblob import Word
import numpy as np
import re
import shorttext
import csv


def main():
    # Languages used
    eng = "english"
    swe = "swedish"
    
    # Stop words
    stop_eng = stopwords_eng()
    stop_swe = stopwords.words(swe)
    
    # Stemmers
    stem_eng = SnowballStemmer(eng)
    stem_swe = SnowballStemmer(swe)

    # Read data
    df = pd.read_csv('text_source/swe.txt', sep="\t", names=[eng, swe])

    # English corpus
    pre_eng = pre_proc(stop_eng, stem_eng)
    corpus_eng = [pre_eng(e).split() for e in df[eng]]
    dtm_eng = shorttext.utils.DocumentTermMatrix(corpus_eng, tfidf=True)
    # Matrix
    m_eng = dtm_eng.dtm
    print ("Shape of English document-to-text matrix is %s" % str(m_eng.get_shape()))
    # Dictionary
    d_eng = dtm_eng.dictionary

    # Swedish corpus
    pre_swe = pre_proc(stop_swe, stem_swe)
    corpus_swe = [pre_swe(s).split() for s in df[swe]]
    dtm_swe = shorttext.utils.DocumentTermMatrix(corpus_swe, tfidf=True)
    # Matrix
    m_swe = dtm_swe.dtm
    print ("Shape of Swedish document-to-text matrix is %s" % str(m_swe.get_shape()))
    # Dictionary
    d_swe = dtm_swe.dictionary
    
    # S12 matrix
    s12 = m_eng.transpose() * m_swe
    print ("Shape of S12 matrix is %s" % str(s12.get_shape()))

    # Max element index from each row
    m_eng_ind = m_eng.get_shape()[1]
    max_elem_ind = s12.argmax(axis=1)
    max_elem = s12.max(axis=1)
    mt = max_elem.todok()
    mt_val = list(mt.values())
    lexicon = []
    for i in range(m_eng_ind):
        if mt[i] > 1:
            lexicon.append([i, mt_val[i], d_eng[i], d_swe[max_elem_ind.item(i)]])
    print ("Writing lexicon to csv file")
    csv_write(sorted(lexicon, key=_sort_val, reverse=True))
    
    
def pre_proc(stop, stemmer):
    pipeline = [lambda s: re.sub('[\d]', '', s),
                lambda s: s.lower(),
                lambda s: ' '.join([w for w in str(s).split() if w not in stop]),
                lambda s: re.sub('[^\w\s]', '', s),
                lambda s: ' '.join([stemmer.stem(w) for w in s.split()])
    ]
    return shorttext.utils.text_preprocessor(pipeline)


def csv_write(lex):
    header = [["sep=,",]]
    header.append(["Index", "Frequency", "English", "Swedish"])
    with open('lexicon.csv', 'w') as cf:
        writer = csv.writer(cf, delimiter=',')
        writer.writerows(header)
        writer.writerows(lex)


def _sort_val(s):
    return s[1]


if __name__ == "__main__":
    main()
    

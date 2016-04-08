# NNBlocks is a Deep Learning framework for computational linguistics.
#
#   Copyright (C) 2015 Frederico Tommasi Caroli
#
#   NNBlocks is free software: you can redistribute it and/or modify it under
#   the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   NNBlocks is distributed in the hope that it will be useful, but WITHOUT ANY
#   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#   details.
#
#   You should have received a copy of the GNU General Public License along with
#   NNBlocks. If not, see http://www.gnu.org/licenses/.

import numpy as np
import theano
import nnb

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def is_year(s):
    return s.isdigit() and 1800 <= int(s) <= 2030

class WordVecsHelper:
    def __init__(self, lower=True):
        self.word_vecs = None
        self.word2index = None
        self.counter = {}
        self.lower = lower

    def add_sentences(self, sentences):
        for sentence in sentences:
            for word in sentence:
                token = word
                if self.lower:
                    token = word.lower()
                if is_year(token) or is_number(token):
                    continue
                if token not in self.counter:
                    self.counter[token] = 0
                self.counter[token] += 1

    def create(self, dim, threshold=0):
        word2index = {}
        word2index['NUMBER'] = 0
        word2index['YEAR'] = 1
        word2index['UNK'] = 2
        num_words = 3

        for token in self.counter:
            if self.counter[token] <= threshold:
                continue
            if is_year(token):
                token = 'YEAR'
            elif is_number(token):
                token = 'NUMBER'

            if token not in word2index:
                word2index[token] = num_words
                num_words += 1


        word_vecs = nnb.rng.normal(
            loc=0,
            scale=1.0,
            size=(num_words, dim)
        )
        self.word_vecs = word_vecs/10
        self.word2index = word2index
        return word_vecs,word2index

    def translate_word(self, s):
        if self.lower:
            s = s.lower()

        if is_year(s):
            s = 'YEAR'
        elif is_number(s):
            s = 'NUMBER'
        if s not in self.word2index:
            s = 'UNK'

        return self.word2index[s]
        
    def translate(self, l):
        if isinstance(l, str):
            return self.translate_word(l)

        r = []
        for s in l:
            r.append(self.translate_word(s))

        return r

    def read_file(self, filename, separator='\t'):
        line_counts = 1
        word_dim = None
        with open(filename) as fin:
            line = fin.readline()
            splits = line.split(separator)
            word_dim = len(splits) - 1
            for line in fin:
                line_counts += 1


        #UNK
        line_counts += 1
        word_vecs = np.empty(
            shape=(line_counts, word_dim),
            dtype=theano.config.floatX
        )
        word_vecs[0, :] = nnb.rng.normal(loc=0., scale=1., size=(1, word_dim)) / 10

        word2index = {}
        word2index['UNK'] = 0
        with open(filename) as fin:
            for line in fin:
                splits = line.split(separator)
                token = splits[0]
                vec = map(float, splits[1:])
                word_vecs[len(word2index), :] = vec
                word2index[token] = len(word2index)

        self.word2index = word2index
        self.word_vecs = word_vecs

        return word_vecs, word2index

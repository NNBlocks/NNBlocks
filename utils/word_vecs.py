import numpy as np

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
    def __init__(self, dim, lower=True):
        self.dim = dim
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

    def create(self, rng=np.random.RandomState(42), threshold=1):
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


        word_vecs = rng.normal(
            loc=0,
            scale=1.0,
            size=(num_words,self.dim)
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
        elif s not in self.word2index:
            s = 'UNK'

        return self.word2index[s]
        
    def translate(self, l):
        if isinstance(l,str):
            self.translate_word(l)

        r = []
        for s in l:
            r.append(self.translate_word(s))

        return r

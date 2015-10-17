import theano.tensor as T

def sigmoid(a):
    return T.nnet.sigmoid(a)

def tanh(a):
    return T.tanh(a)

def linear(a):
    return a

def threshold(t, yes=1., no=0.):
    def r(a):
        return T.switch(T.ge(a, t), yes, no)
    return r

def ReLU(a):
    return T.switch(T.lt(a, 0.), 0., a)

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

def leaky_ReLU(alpha):
    def r(a):
        f1 = 0.5 * (a + alpha)
        f2 = 0.5 * (a - alpha)
        return f1 * a + f2 * abs(a)
    return r

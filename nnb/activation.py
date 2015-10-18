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

#The ReLU functions are a copy of theano's recommended way to implement ReLU.
#theano.tensor.nnet.relu is not used here because it is only available in
#version 0.7.2 of theano

def ReLU(a):
    return 0.5 * (a + abs(a))

def leaky_ReLU(alpha, x=None):
    def r(a):
        f1 = 0.5 * (a + alpha)
        f2 = 0.5 * (a - alpha)
        return f1 * a + f2 * abs(a)
    if x is not None:
        return r(x)
    else:
        return r

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

def leaky_ReLU(alpha):
    def r(a):
        f1 = 0.5 * (a + alpha)
        f2 = 0.5 * (a - alpha)
        return f1 * a + f2 * abs(a)
    return r

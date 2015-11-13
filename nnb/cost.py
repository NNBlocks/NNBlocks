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

__all__ = [
    'cross_entropy_error',
    'CrossEntropyError',
    'mean_square_error',
    'MeanSquareError',
    'negative_log_likelihood_error',
    'NegativeLogLikelihoodError'
]

import theano.tensor as T
import theano
import nnb


def _check_inputs(prev):
    if len(prev) != 2:
        raise ValueError("Error models can only treat exactly 2 inputs")

def cross_entropy_error(p, y):
    if p.ndim != y.ndim:
        raise ValueError("Cross entropy can only be performed in outputs and" +
                        " expected outputs with the same dimensions")
    if p.ndim == 0:
        return -y * T.log(p)
    elif p.ndim <= 2:
        return -T.mean((y * T.log(p)).T.sum(axis=0))
    elif p.ndim == 3:
        def one_step(i, o):
            return (o * T.log(i)).T.sum(axis=0)
        errors, _ = theano.scan(fn=one_step, sequences=[p, y])
        return -T.mean(errors)
    else:
        raise NotImplemented("cross_entropy can only be performed on ndim up " +
                            "to 3")

class CrossEntropyError(nnb.Model):
    def apply(self, prev):
        _check_inputs(prev)
        return [cross_entropy_error(*prev)]

def mean_square_error(p, y):
    if p.ndim == 0 and y.ndim == 0:
        return T.sqr(p - y) / 2
    elif p.ndim == 1 and y.ndim == 1:
        return T.mean(T.sqr(p - y)) / 2
    elif p.ndim == 2 and y.ndim == 2:
        return T.mean(T.sqr(p - y).sum(axis=1)) / 2
    else:
        error_str = "Invalid dimensions for mean square error: {0} and {1}."
        error_str = error_str.format(p.ndim, y.ndim)
        raise ValueError(error_str)

class MeanSquareError(nnb.Model):
    def apply(self, prev):
        _check_inputs(prev)
        return [mean_square_error(*prev)]

def negative_log_likelihood_error(p, y):
    if p.ndim == 1 and y.ndim == 0:
        return -T.log(p[y])
    elif p.ndim == 2 and y.ndim == 1:
        return -T.mean(T.log(p)[T.arange(y.shape[0]), y])
    else:
        error_str = "Invalid dimensions for negative log likelihood: {0} and" +\
                    "{1}."
        error_str = error_str.format(p.ndim, y.ndim)
        raise ValueError(error_str)

class NegativeLogLikelihoodError(nnb.Model):
    def apply(self, prev):
        _check_inputs(prev)
        return [negative_log_likelihood_error(*prev)]

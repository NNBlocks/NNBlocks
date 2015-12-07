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

import nnb
import theano
import theano.tensor as T
import numpy as np
from trainer import Trainer

class SGDTrainer(Trainer):
    """A Trainer that minimizes a Model using a stochastic gradient descent

    :param learning_rate: Initial learning rate for the method. The learning
        rate can later be adjusted using the set_learning_rate method.
    :param momentum: The momentum parameter for the gradient descent. Default is
        0.
    """
    def init_options(self):
        opts = nnb.utils.Options()
        opts.add(
            name="learning_rate",
            value=0.1,
            value_type=float,
        )
        opts.add(
            name="momentum",
            value=0.,
            value_type=float
        )
        return opts
    def setup(self):
        opts = self.options
        model = opts.get('model')
        lr = opts.get('learning_rate')
        momentum = opts.get('momentum')
        params = model.params

        if len(params) == 0:
            raise ValueError("The model has no parameters to train")

        inputs, output, updates = self.get_io()
        cost = self.get_cost()

        lr = theano.shared(value=lr)
        self.__lr = lr

        velocity = [
            theano.shared(
                p.get_value() * np.asarray(0., dtype=theano.config.floatX)
            ) for p in params
        ]

        grads_hist = [
            theano.shared(
                p.get_value() * np.asarray(0., dtype=theano.config.floatX)
            ) for p in params
        ]

        grads = [T.grad(cost=cost, wrt=param) for param in params]

        for hist, grad in zip(grads_hist, grads):
            updates[hist] = hist + grad

        self.__update_grads = theano.function(inputs, [], updates=updates)

        batch_size = T.iscalar()

        grads_mean = [g / batch_size for g in grads_hist]

        updates = []
        for param, grad, v in zip (params, grads_mean, velocity):
            updates.append((v, momentum * v + lr * grad))
            updates.append((param, param - v))
        for grad in grads_hist:
            updates.append((grad, T.zeros_like(grad)))

        self.__update_params = theano.function([batch_size], [],
                                                updates=updates)

    def train(self, inputs):
        for inp in inputs:
            self.__update_grads(*inp)
        self.__update_params(len(inputs))

    def set_learning_rate(self, learning_rate):
        """Sets the learning rate
        """
        self.__lr.set_value(learning_rate)

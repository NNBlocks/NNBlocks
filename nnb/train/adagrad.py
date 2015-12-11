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
import theano
import numpy as np
import nnb
from nnb.train import Trainer
from nnb.utils import Options

class AdagradTrainer(Trainer):
    """A Trainer to minimize a Model using an adaptative gradient method.

    :param learning_rate: Initial learning rate for the method. The learning
        rate can later be adjusted using the set_learning_rate method.
    """

    def reset_history(self):
        """Resets the adagrad history
        """
        self.__reset_hist()

    @staticmethod
    def init_options():
        ops = Options()
        ops.add(
            name="hist",
            value_type=list
        )
        ops.add(
            name="learning_rate",
            value=0.1,
            value_type=float
        )
        ops.add(
            name="batch_size",
            value=20,
            value_type=int
        )
        ops.add(
            name="dataset",
            value_type=nnb.data.Dataset,
            required=True
        )
        return ops

    def setup(self):
        options = self.options
        model = options.get('model')
        batch_size = options.get('batch_size')
        dataset = options.get('dataset')

        params = model.params
        if params is None or len(params) == 0:
            raise ValueError("The model has no parameters to train")

        inputs, output, updates = self.get_io()
        index = T.lscalar("batch_index")
        cost = self.get_cost()

        batch_size = T.iscalar()

        params_grads = [T.grad(cost=cost, wrt=param) for param in params]

        adagrad_hist = options.get('hist')
        if adagrad_hist is None:
            adagrad_hist = [
                    np.zeros_like(param.get_value(borrow=True))
                for param in params
            ]
        adagrad_hist = [
            theano.shared(
                hist,
                name="adagrad_hist_{0}".format(param),
                borrow=True
            ) for hist, param in zip(adagrad_hist, params)
        ]

        new_hist = [ah + T.sqr(param_g)
                        for ah, param_g in zip(adagrad_hist, params_grads)]

        new_grad = [grad / (1e-6 + T.sqrt(ah))
                    for grad, ah in zip(params_grads, new_hist)]

        learning_rate = options.get('learning_rate')
        learning_rate = theano.shared(value=learning_rate)
        self.__lr = learning_rate

        for param, ng in zip(params, new_grad):
            updates[param] = param - learning_rate * ng

        for hist, nh in zip(adagrad_hist, new_hist):
            updates[hist] = nh

        givens = dataset.to_givens(inputs, index, batch_size)

        self.__train = theano.function(
            inputs=[index],
            outputs=None,
            updates=updates,
            givens=givens
        )

        adagrad_reset_update = [(hist, T.zeros_like(hist))
                                for hist in adagrad_hist]

        self.__reset_hist = theano.function(
            inputs=[],
            outputs=None,
            updates=adagrad_reset_update
        )

    def train(self, index):
        self.__train(index)

    def set_learning_rate(self, learning_rate):
        """Sets the learning rate
        """
        self.__lr.set_value(learning_rate)

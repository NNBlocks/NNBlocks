import nnb
import theano
import theano.tensor as T
import numpy as np
from trainer import Trainer

class SGDTrainer(Trainer):
    def init_options(self):
        opts = nnb.utils.Options()
        opts.add(
            name="learning_rate",
            value=0.1,
            value_type=float,
            description="Learning rate used for the training"
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

        inputs, output = self.get_io()
        cost = self.get_cost()

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

        grads_update = []
        for hist, grad in zip(grads_hist, grads):
            grads_update.append((hist, hist + grad))

        self.__update_grads = theano.function(inputs, [], updates=grads_update)

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

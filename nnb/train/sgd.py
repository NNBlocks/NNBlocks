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
        return opts
    def setup(self):
        opts = self.options
        model = opts.get('model')
        lr = opts.get('learning_rate')
        params = model.params

        if len(params) == 0:
            raise ValueError("The model has no parameters to train")

        inputs, output = self.get_io()
        cost = self.get_cost()
        expected_output = self.get_expected_output()

        grads_hist = [
            theano.shared(
                p.get_value() * np.asarray(0., dtype=theano.config.floatX)
            ) for p in params
        ]

        grads = [T.grad(cost=cost, wrt=param) for param in params]

        grads_update = []
        for hist, grad in zip(grads_hist, grads):
            grads_update.append((hist, hist + grad))

        self.__update_grads = theano.function(inputs + [expected_output], [],
                                                updates=grads_update)

        batch_size = T.iscalar()

        grads_mean = [g / batch_size for g in grads_hist]

        updates = []
        for param, grad in zip (params, grads_mean):
            updates.append((param, param - lr * grad))
        for grad in grads_hist:
            updates.append((grad, T.zeros_like(grad)))

        self.__update_params = theano.function([batch_size], [],
                                                updates=updates)

    def train(self, inputs):
        for inp in inputs:
            self.__update_grads(*inp)
        self.__update_params(len(inputs))

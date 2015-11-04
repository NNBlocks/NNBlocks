import theano.tensor as T
import theano
import numpy as np
from nnb.train import Trainer
from nnb.utils import Options

class AdagradTrainer(Trainer):
    reset_history = None

    @staticmethod
    def init_options():
        ops = Options()
        ops.add(
            name="hist",
            value_type=list,
            description="Starting adagrad history"
        )
        ops.add(
            name="learning_rate",
            value=0.1,
            value_type=float,
            description="Learning rate used for the training"
        )
        return ops

    def setup(self):
        options = self.options
        model = options.get('model')

        params = model.params
        if params is None or len(params) == 0:
            raise ValueError("The model has no parameters to train")

        inputs, output = self.get_io()
        cost = self.get_cost()

        batch_size = T.iscalar()

        params_grads = [T.grad(cost=cost, wrt=param) for param in params]
        grads_hist = [
            theano.shared(
                p.get_value() * np.asarray(0., dtype=theano.config.floatX)
            ) for p in params
        ]

        grads_mean = [g / batch_size for g in grads_hist]

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
                        for ah, param_g in zip(adagrad_hist, grads_mean)]

        new_grad = [grad / (1e-6 + T.sqrt(ah))
                    for grad, ah in zip(grads_mean, new_hist)]

        learning_rate = options.get('learning_rate')

        import collections
        updates = collections.OrderedDict()
        for param, ng in zip(params, new_grad):
            updates[param] = param - learning_rate * ng

        for hist, nh in zip(adagrad_hist, new_hist):
            updates[hist] = nh

        for g in grads_hist:
            updates[g] = T.zeros_like(g)

        adagrad_reset_update = [(hist, T.zeros_like(hist))
                                for hist in adagrad_hist]

        self.reset_history = theano.function(
            inputs=[],
            outputs=None,
            updates=adagrad_reset_update
        )


        update_grads = collections.OrderedDict()
        for g, pg in zip(grads_hist, params_grads):
            update_grads[g] = g + pg

        self.__compute_grads = theano.function(inputs, updates=update_grads)
        self.__train_with_grads = theano.function([batch_size], [],
                                                    updates=updates)

    def train(self, inputs):
        options = self.options

        for inp in inputs:
            self.__compute_grads(*inp)
        self.__train_with_grads(len(inputs))

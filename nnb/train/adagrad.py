import theano.tensor as T
import theano
import numpy as np
from nnb.train import Trainer

class AdagradTrainer(Trainer):
    reset_history = None

    @staticmethod
    def get_options():
        ops = Trainer.get_options()
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

        inputs, output = self.get_io()
        expected_output = self.get_expected_output()
        cost = self.get_cost()

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

        import collections
        updates = collections.OrderedDict()
        for param, ng in zip(params, new_grad):
            updates[param] = param - learning_rate * ng

        for hist, nh in zip(adagrad_hist, new_hist):
            updates[hist] = nh

        adagrad_reset_update = [(hist, T.zeros_like(hist))
                                for hist in adagrad_hist]

        self.reset_history = theano.function(
            inputs=[],
            outputs=None,
            updates=adagrad_reset_update
        )


        all_ = inputs + [expected_output]

        self.__get_grads = theano.function(all_, params_grads)
        self.__train_with_grads = theano.function(params_grads, [],
                                                    updates=updates)

    def train(self, inputs):
        options = self.options
        model = options.get('model')

        grads = [np.zeros_like(param.get_value(borrow=True)) 
                for param in model.params]
        for inp in inputs:
            grads_i = self.__get_grads(*inp)
            for g, gi in zip(grads, grads_i):
                g += gi

        for g in grads:
            g /= len(inputs)
        self.__train_with_grads(*grads)

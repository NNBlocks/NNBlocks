import theano.tensor as T
import theano
import numpy as np
import nn
import nn.cost as cost
from nn.train import Trainer

class AdagradTrainer(Trainer):
    reset_history = None

    @staticmethod
    def get_options():
        ops = Trainer.get_options()
        ops.add(
            name="model",
            required=True,
            description="The model to be trained.",
            value_type=nn.Model
        )
        ops.add(
            name="L2_reg",
            value=0.,
            value_type=[float, list],
            description="L2 regularization values. It can be a float that " + \
                        "will be applied for all parameters or a list of " + \
                        "floats, one for each parameter of the model."
        )
        ops.add(
            name="L1_reg",
            value=0.,
            value_type=[float, list],
            description="L1 regularization values. It can be a float that " + \
                        "will be applied for all parameters or a list of " + \
                        "floats, one for each parameter of the model."
        )
        ops.add(
            name="cost_func",
            value=cost.neg_log_likelihood,
            description="Cost function to be applied to the model's output " + \
                        "and expected output."
        )
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

        output = model.output
        t = T.TensorType(output.dtype, (False,) * output.ndim)
        expected_output = t('expected_output')

        cost_func = options.get('cost_func')
        cost = cost_func(output, expected_output)

        params = model.params

        L2_reg = options.get('L2_reg')

        if isinstance(L2_reg, float):
            L2_reg = [L2_reg for p in params]

        L1_reg = options.get('L1_reg')

        if isinstance(L1_reg, float):
            L1_reg = [L1_reg for p in params]

        for l1, l2, param in zip(L1_reg, L2_reg, params):
            cost += abs(param).sum() * l1
            cost += T.sqr(param).sum() * l2

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

        inputs = model.input

        all_ = inputs + [expected_output]

        self.__train = theano.function(all_, [], updates=updates)
        self.__get_grads = theano.function(all_, params_grads)
        self.__train_with_grads = theano.function(params_grads, [],
                                                    updates=updates)

    def train(self, inputs, expected_outputs):
        options = self.options
        model = options.get('model')
        supports_batch = False
        try:
            supports_batch = model.options.get('model_supports_batch')
        except KeyError:
            pass

        if not supports_batch:
            grads = [[] for param in model.params]
            for inp, outp in zip(inputs, expected_outputs):
                a = list(inp) + [outp]
                grads_i = self.__get_grads(*a)
                for g, gi in zip(grads, grads_i):
                    g.append(gi)
            grads = [np.mean(grad, axis=0) for grad in grads]
            self.__train_with_grads(*grads)
        else:
            a = np.append(inputs, expected_outputs, 1)
            self.__train(*a)


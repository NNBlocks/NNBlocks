import theano.tensor as T
import theano
import numpy as np
from NN import Model

class SoftmaxLayer(Model):

    @staticmethod
    def get_options():
        ops = Model.get_options()
        ops.add(
            name="model_supports_batch",
            value=True,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
        )
        ops.add(
            name="in_dim",
            required=True,
            value_type=int,
            description="""Number of input dimensions"""
        )
        ops.add(
            name="out_dim",
            required=True,
            value_type=int,
            description="""Number of classes this model is trying to predict"""
        )
        ops.add(
            name="W_softmax",
            value_type=np.ndarray,
            description="""The matrix used for the softmax layer"""
        )
        ops.add(
            name="b_softmax",
            value_type=np.ndarray,
            description="""Bias vector used for the softmax layer"""
        )
        return ops

    def init_params(self):
        options = self.options
        rng = np.random.RandomState(32523)
        in_dim = options.get('in_dim')
        out_dim = options.get('out_dim')

        W_softmax = options.get('W_softmax')
        if W_softmax is None:
            W_softmax = np.asarray(
                rng.uniform(
                    low=-1/np.sqrt(in_dim),
                    high=1/np.sqrt(in_dim),
                    size=(in_dim, out_dim)
                ),
                dtype=theano.config.floatX
            )
        W_softmax = theano.shared(
            value=W_softmax,
            name='W_softmax',
            borrow=True
        )

        b_softmax = options.get('b_softmax')
        if b_softmax is None:
            b_softmax = np.asarray(
                rng.uniform(
                    low=0,
                    high=1,
                    size=(out_dim,)
                ),
                dtype=theano.config.floatX
            )
        b_softmax = theano.shared(
            value=b_softmax,
            name='b_softmax',
            borrow=True
        )

        return [W_softmax, b_softmax]

    def generate_input(self):
        return [T.matrix("softmax_in")]

    def generate_output(self, inputs):
        inps = inputs[0]
        W_softmax = self.params[0]
        b_softmax = self.params[1]

        return T.nnet.softmax(T.dot(inps, W_softmax) + b_softmax)

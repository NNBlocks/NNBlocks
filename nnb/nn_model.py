from nnb import Model
import numpy as np
import nnb.utils as utils
import theano
import theano.tensor as T

class PerceptronLayer(Model):
    """A Perceptron layer
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="insize",
            description="The size of the input of the layer.",
            value_type=int,
            required=True
        )
        ops.add(
            name="outsize",
            description="The size of the output of the layer.",
            value_type=int,
            required=True
        )
        ops.add(
            name='W',
            description="Matrix of the network's weights, with dimensions "
                        "equal to (insize, outsize)"
        )
        ops.add(
            name='b',
            description="Vector of the network's biases, with length outsize"
        )
        return ops

    def init_params(self):
        rng = np.random.RandomState(1234)
        options = self.options
        insize = options.get('insize')
        outsize = options.get('outsize')


        W = options.get('W')
        if W is None:
            W = np.asarray(
                rng.uniform(
                    low=-1/np.sqrt(insize),
                    high=1/np.sqrt(insize),
                    size=(insize, outsize)
                ),
                dtype=theano.config.floatX
            )
        W = theano.shared(
            value=W,
            name='W',
            borrow=True
        )

        b = options.get('b')
        if b is None:
            b = np.zeros(outsize, dtype=theano.config.floatX)
        b = theano.shared(
            value=b,
            name='b',
            borrow=True
        )

        return [W, b]

    def apply(self, prev):
        W = self.params[0]
        b = self.params[1]
        inp = prev[0]

        return [T.tanh(inp.dot(W) + b)]


class SoftmaxLayer(Model):
    """A Softmax layer
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="model_supports_batch",
            value=True,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
        )
        ops.add(
            name="insize",
            required=True,
            value_type=int,
            description="""Input vectors' length"""
        )
        ops.add(
            name="outsize",
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
        in_dim = options.get('insize')
        out_dim = options.get('outsize')

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

    def apply(self, prev):
        inps = prev[0]
        W_softmax = self.params[0]
        b_softmax = self.params[1]

        return [T.nnet.softmax(T.dot(inps, W_softmax) + b_softmax)]


class RecursiveNeuralNetwork(Model):
    """
    A Recursive Neural Network. 
    Each composition in the recursive layer is made with the model passed by
    options.
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="comp_model",
            description="The composition model to be used in the tree",
            value_type=Model
        )
        ops.add(
            name="insize",
            required=True,
            value_type=int
        )

        return ops

    def init_params(self):
        options = self.options
        word_dim = options.get('insize')
        comp_model = options.get('comp_model')

        if comp_model is None:
            comp_model = PerceptronLayer(insize=word_dim * 2, outsize=word_dim)
            options.set('comp_model', comp_model)

        return comp_model.params

    def apply(self, inputs):
        x = inputs[0]
        comp_tree = inputs[1]
        comp_model = self.options.get('comp_model')

        #Stanford NLP does this. Not described in the paper.
        x = T.tanh(x)

        #Composition function for two word_vecs
        def compose(u, v):
            stack = T.concatenate([u, v], axis=0)
            out = comp_model.apply([stack])[0]
            if out.ndim == 2:
                import warnings
                warnings.warn("The composition model's output is 2 " +
                            "dimensional. Using the first row of it as " +
                            "composition.", RuntimeWarning)
                out = out[0]
            return out

        #One theano.scan step in the RNN feedforward
        def one_step(children, index, partial):
            return T.set_subtensor(
                        partial[index],
                        compose(
                            partial[children[0]],
                            partial[children[1]]
                        )
                    )

        #Allocate partial results matrix. Each line will hold a node's value
        partial = T.alloc(0., x.shape[0] + comp_tree.shape[0], x.shape[1])

        #Set the first n nodes to be the phrase's word_vecs
        partial = T.set_subtensor(partial[:x.shape[0]], x)

        #Execute the scan
        h, _ = theano.scan(
            fn=one_step,
            outputs_info=partial,
            sequences=[
                comp_tree,
                T.arange(
                    x.shape[0],
                    x.shape[0] + comp_tree.shape[0]
                )
            ]
        )

        #Get the last iteration's values
        h = h[-1]

        return [h]


class RecurrentNeuralNetwork(Model):
    """A simple recurrent neural network
    """
    @staticmethod
    def init_options():
        ops = utils.Options()

        ops.add(
            name='insize',
            value_type=int,
            required=True,
            description="Size of each input of the network."
        )

        ops.add(
            name='outsize',
            value_type=int,
            required=True,
            description="Size of each output of the network."
        )

        ops.add(
            name='h0',
            value_type=np.ndarray,
            description="Initial 'memory' input."
        )

        ops.add(
            name='W',
            value_type=np.ndarray,
            description="Weight matrix for x, the features vector."
        )

        ops.add(
            name='b',
            value_type=np.ndarray,
            description="Bias vector for the transformation."
        )

        ops.add(
            name='W_h',
            value_type=np.ndarray,
            description="Weight matrix for h_t-1, the memory input."
        )

        return ops

    def init_params(self):
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')
        h0 = self.options.get('h0')
        W = self.options.get('W')
        b = self.options.get('b')
        W_h = self.options.get('W_h')

        rng = np.random.RandomState(456)

        if h0 is None:
            h0 = np.zeros(shape=(outsize,), dtype=theano.config.floatX)

        if W is None:
            W = np.asarray(
                rng.uniform(
                    low=-1/np.sqrt(insize),
                    high=1/np.sqrt(insize),
                    size=(insize, outsize)
                ),
                dtype=theano.config.floatX
            )

        if b is None:
            b = np.zeros(shape=(outsize,), dtype=theano.config.floatX)

        if W_h is None:
            W_h = np.asarray(
                rng.uniform(
                    low=-1/np.sqrt(outsize),
                    high=1/np.sqrt(outsize),
                    size=(outsize, outsize)
                ),
                dtype=theano.config.floatX
            )
        h0 = theano.shared(value=h0, borrow=True)
        W = theano.shared(value=W, borrow=True)
        b = theano.shared(value=b, borrow=True)
        W_h = theano.shared(value=W_h, borrow=True)

        return [h0, W, b, W_h]

    def apply(self, inputs):
        options = self.options
        h0 = self.params[0]
        W = self.params[1]
        b = self.params[2]
        W_h = self.params[3]
        x = inputs[0]

        def one_step(x_t, h_tm1):
            z = x_t.dot(W) + b
            m = h_tm1.dot(W_h)
            return T.tanh(z + m)

        h, updates = theano.scan(
            fn=one_step,
            sequences=x,
            outputs_info=[h0],
            n_steps=x.shape[0]
        )

        return [h]

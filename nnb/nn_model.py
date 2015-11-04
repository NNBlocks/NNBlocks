from nnb import Model
import warnings
import numpy as np
import nnb
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
        ops.add(
            name='activation_func',
            value=T.nnet.sigmoid
        )

        return ops

    def init_params(self):
        options = self.options
        insize = options.get('insize')
        outsize = options.get('outsize')


        W = options.get('W')
        if W is None:
            W = np.asarray(
                nnb.rng.uniform(
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

        return [self.options.get('activation_func')(inp.dot(W) + b)]


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
        in_dim = options.get('insize')
        out_dim = options.get('outsize')

        W_softmax = options.get('W_softmax')
        if W_softmax is None:
            W_softmax = np.asarray(
                nnb.rng.uniform(
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
                nnb.rng.uniform(
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
            value_type=int
        )

        return ops

    def init_params(self):
        options = self.options
        word_dim = options.get('insize')
        comp_model = options.get('comp_model')

        if comp_model is not None and word_dim is not None:
            warning.warn("You only have to set either the 'insize' or the " +
                        "'comp_model' option, not both. Using just the " +
                        "'comp_model'.")

        if comp_model is None:
            if word_dim is None:
                raise ValueError("Either the 'insize' or the 'comp_model' " +
                                "option should be set.")
            comp_model = nnb.ConcatenationModel(axis=0)
            comp_model |= PerceptronLayer(insize=word_dim * 2, outsize=word_dim)
            options.set('comp_model', comp_model)

        return comp_model.params

    def _get_inputs(self):
        return self.options.get('comp_model')._get_inputs()

    def apply(self, inputs):
        comp_tree = inputs[0]
        x = inputs[1:]
        comp_model = self.options.get('comp_model')

        #One theano.scan step in the RNN feedforward
        def one_step(children, index, *partials):
            inputs1 = []
            inputs2 = []
            for partial in partials:
                inputs1.append(partial[children[0]])
                inputs2.append(partial[children[1]])
            model_out = comp_model.apply(inputs1 + inputs2)
            new_partials = []
            for p, o in zip(partials, model_out):
                new_partials.append(T.set_subtensor(p[index], o))

            return tuple(new_partials)

        partials = []
        for o in x:
            shape = []
            for i in range(1, o.ndim):
                shape.append(o.shape[i])

            partial = T.alloc(0., o.shape[0] + comp_tree.shape[0], *shape)
            partial = T.set_subtensor(partial[:o.shape[0]], o)
            partials.append(partial)

        #Execute the scan
        h, _ = theano.scan(
            fn=one_step,
            outputs_info=partials,
            sequences=[
                comp_tree,
                T.arange(
                    x[0].shape[0],
                    x[0].shape[0] + comp_tree.shape[0]
                )
            ]
        )

        #Get the last iteration's values
        if isinstance(h, list):
            h = [o[-1] for o in h]
        else:
            h = [h[-1]]

        return h


class SimpleRecurrence(Model):
    """A simple recurrence for a RecurrentNeuralNetwork
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

        ops.add(
            name='activation_func',
            value=T.nnet.sigmoid
        )

        return ops

    def init_params(self):
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')
        W = self.options.get('W')
        b = self.options.get('b')
        W_h = self.options.get('W_h')

        if W is None:
            W = np.asarray(
                nnb.rng.uniform(
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
                nnb.rng.uniform(
                    low=-1/np.sqrt(outsize),
                    high=1/np.sqrt(outsize),
                    size=(outsize, outsize)
                ),
                dtype=theano.config.floatX
            )
        W = theano.shared(value=W, borrow=True, name='W')
        b = theano.shared(value=b, borrow=True, name='b')
        W_h = theano.shared(value=W_h, borrow=True, name='W_h')

        return [W, b, W_h]

    def apply(self, inputs):
        W = self.params[0]
        b = self.params[1]
        W_h = self.params[2]
        x_t = inputs[0]
        h_tm1 = inputs[1]

        z = x_t.dot(W) + b
        m = h_tm1.dot(W_h)
        return [self.options.get('activation_func')(z + m)]

class LSTMRecurrence(Model):
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name='insize',
            required=True,
            value_type=int
        )
        opts.add(
            name='outsize',
            required=True,
            value_type=int
        )
        opts.add(
            name='Wi',
            value_type=np.ndarray
        )
        opts.add(
            name='Wf',
            value_type=np.ndarray
        )
        opts.add(
            name='Wc',
            value_type=np.ndarray
        )
        opts.add(
            name='Wo',
            value_type=np.ndarray
        )
        opts.add(
            name='Ui',
            value_type=np.ndarray
        )
        opts.add(
            name='Uf',
            value_type=np.ndarray
        )
        opts.add(
            name='Uc',
            value_type=np.ndarray
        )
        opts.add(
            name='Uo',
            value_type=np.ndarray
        )
        opts.add(
            name='Vo',
            value_type=np.ndarray
        )
        opts.add(
            name='bi',
            value_type=np.ndarray
        )
        opts.add(
            name='bf',
            value_type=np.ndarray
        )
        opts.add(
            name='bc',
            value_type=np.ndarray
        )
        opts.add(
            name='bo',
            value_type=np.ndarray
        )

        return opts

    def init_params(self):
        opts = self.options
        Wi = opts.get('Wi')
        Wf = opts.get('Wf')
        Wc = opts.get('Wc')
        Wo = opts.get('Wo')
        Ui = opts.get('Ui')
        Uf = opts.get('Uf')
        Uc = opts.get('Uc')
        Uo = opts.get('Uo')
        Vo = opts.get('Vo')
        bi = opts.get('bi')
        bf = opts.get('bf')
        bc = opts.get('bc')
        bo = opts.get('bo')
        insize = opts.get('insize')
        outsize = opts.get('outsize')

        def make_shared_matrix(p):
            if p is None:
                p = np.asarray(
                    nnb.rng.uniform(
                        low=-1. / np.sqrt(insize),
                        high=1. / np.sqrt(insize),
                        size=(insize, outsize)
                    ),
                    dtype=theano.config.floatX
                )
            return theano.shared(value=p, borrow=True)

        def make_shared_vec(p):
            if p is None:
                p = np.zeros(outsize, theano.config.floatX)
            return theano.shared(value=p, borrow=True)

        matrices = [make_shared_matrix(p)
                    for p in [Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, Vo]]
        vectors = [make_shared_vec(p)
                    for p in [bi, bf, bc, bo]]

        return matrices + vectors

    def apply(self, inputs):
        x_t = inputs[0]
        h_tm1 = inputs[1]
        C_tm1 = inputs[2]
        Wi = self.params[0]
        Wf = self.params[1]
        Wc = self.params[2]
        Wo = self.params[3]
        Ui = self.params[4]
        Uf = self.params[5]
        Uc = self.params[6]
        Uo = self.params[7]
        Vo = self.params[8]
        bi = self.params[9]
        bf = self.params[10]
        bc = self.params[11]
        bo = self.params[12]

        it = T.nnet.sigmoid(x_t.dot(Wi) + h_tm1.dot(Ui) + bi)
        _Ct = T.tanh(x_t.dot(Wc) + h_tm1.dot(Uc) + bc)
        ft = T.nnet.sigmoid(x_t.dot(Wf) + h_tm1.dot(Uf) + bf)
        Ct = it * _Ct + ft * C_tm1
        ot = T.nnet.sigmoid(x_t.dot(Wo) + h_tm1.dot(Uo) + bo)
        ht = ot * T.tanh(Ct)
        return [ht, Ct]


class RecurrentNeuralNetwork(Model):
    """A recurrent neural network.
    The model inputs will be:
        0-n: x_t
        rest: h_tm1
    where n is the number of inputs per time t
    """
    @staticmethod
    def init_options():
        ops = utils.Options()

        ops.add(
            name='model',
            value_type=Model,
            description="Model to be used for the recurrence."
        )

        ops.add(
            name='h0',
            value_type=[np.ndarray, list],
            description="Initial 'memory' input."
        )

        ops.add(
            name='insize',
            value_type=int,
            description="Size of each input vector of the model. This is " +
                        "only used if the option 'model' is not set."
        )

        ops.add(
            name='outsize',
            value_type=int,
            description="Size of each output vector of the model. This is " +
                        "required so the initial memory input h0 can be " +
                        "initialized."
        )

        return ops

    def init_params(self):
        model = self.options.get('model')
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')
        h0 = self.options.get('h0')

        if model is None:
            if insize is None or outsize is None:
                raise ValueError("Either the option 'model' or 'insize'" +
                                "+'outsize' should be set in " +
                                "RecurrentNeuralNetwork.")
            if h0 is None:
                h0 = np.zeros(shape=(outsize,), dtype=theano.config.floatX)
            h0 = [theano.shared(name='h0', value=h0, borrow=True)]
            model = SimpleRecurrence(insize=insize, outsize=outsize)
            self.options.set('model', model)
        else:
            if h0 is None:
                raise ValueError("The option 'h0' should be set if you are " +
                                "setting your own model for recurrence.")
            if not isinstance(h0, list):
                h0 = [theano.shared(name='h0', value=h0, borrow=True)]
            else:
                h0_n = []
                for i in range(len(h0)):
                    h0_n.append(
                        theano.shared(
                            name='h0_{0}'.format(i),
                            value=h0[i],
                            borrow=True
                        )
                    )
                h0 = h0_n


        return h0 + model.params

    def _get_inputs(self):
        return self.options.get('model')._get_inputs()

    def apply(self, inputs):
        options = self.options
        model = options.get('model')
        h0 = self.params[:len(self.params) - len(model.params)]

        def one_step(*args):
            return tuple(model.apply(list(args)))

        h, updates = theano.scan(
            fn=one_step,
            sequences=inputs,
            outputs_info=h0
        )

        if not isinstance(h, list):
            h = [h]

        return h

class ConvolutionalNeuralNetwork(Model):
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name='stride',
            value_type=int,
            value=1
        )
        opts.add(
            name='window',
            value_type=int,
            required=True
        )
        opts.add(
            name='insize',
            value_type=int
        )
        opts.add(
            name='outsize',
            value_type=int
        )
        opts.add(
            name='activation_func',
            value=nnb.activation.sigmoid
        )

        return opts

    def init_params(self):
        opts = self.options
        insize = opts.get('insize')
        outsize = opts.get('outsize')
        window = opts.get('window')

        W = np.asarray(nnb.rng.uniform(
            low=-1 / np.sqrt(insize),
            high=1 / np.sqrt(insize),
            size=(outsize, insize, window)
        ), dtype=theano.config.floatX)
        W = theano.shared(value=W, name='W', borrow=True)

        b = np.asarray(np.zeros(shape=(outsize,)), dtype=theano.config.floatX)
        b = theano.shared(value=b, name='b', borrow=True)

        return [W, b]

    def apply(self, prev):
        W = self.params[0]
        b = self.params[1]
        stride = self.options.get('stride')
        window = self.options.get('window')
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')

        conv = T.nnet.conv2d(
            prev[0].dimshuffle('x', 'x', 0, 1),
            W.dimshuffle(0, 'x', 1, 2),
            filter_shape=(outsize, 1, insize, window),
            image_shape=(1, 1, None, insize),
            subsample=(1, stride)
        )
        act = self.options.get('activation_func')
        output = act(conv + b.dimshuffle('x', 0, 'x', 'x'))

        return [output.dimshuffle(1, 2, 3).flatten(ndim=2).dimshuffle(1, 0)]

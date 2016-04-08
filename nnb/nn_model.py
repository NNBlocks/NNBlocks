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

from nnb import Model
import warnings
import numpy as np
import nnb
import nnb.utils as utils
import nnb.init as init
import theano
import theano.tensor as T

class PerceptronLayer(Model):
    """A Perceptron layer
    This Model implements a Perceptron Layer for Neural Networks. The horizontal
    joining of several of these layers forms a Multilayer Perceptron.
    This Model will take a single input x and output
    activation_func(x.dot(W) + b).

    :param insize: Required int. The input size. If the input of this model is
        a vector, insize will be the vector's length. If the input is a matrix,
        insize will be the length of each row.
    :param outsize: Required int. The output size. This can be thought as the
        layer's number of neurons. If the input is a vector, outsize is the
        length of the output vector. If the input is a matrix, outsize is the
        length of each row of the output matrix.
    :param activation_func: Optional callable object. This is the activation
        function used in the weighted average of the input vector. This function
        should use only theano operations. Default is nnb.activation.sigmoid
    :param init: The weight initializer for the layer weights. Default is
        XavierInitializer.
    :param W: Optional numpy ndarray with ndim=2. If set, the weights of this
        layer are not initialized with `init`. Instead they are set to this
        parameter's value.
    :param b: Optional numpy ndarray with ndim=1. If set, the bias vector of
        this layer is not initialized with zeros. Instead it is set to this
        parameter's value.

    Inputs:
        A single input x with x.ndim=1 or x.ndim=2. x.shape[-1] should be equal
            to the insize parameter

    Outputs:
        A single output y with y.ndim=x.ndim and y.shape[-1] equal to the
            outsize parameter.

    Tunable Parameters:
        W - Weight matrix
        b - Bias vector
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="insize",
            value_type=int,
            required=True
        )
        ops.add(
            name="outsize",
            value_type=int,
            required=True
        )
        ops.add(
            name="init",
            value_type=init.Initializer,
            value=init.XavierInitializer()
        )
        ops.add(
            name='W',
        )
        ops.add(
            name='b',
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
        init = options.get('init')

        W = options.get('W')
        if W is None:
            W = init((insize, outsize))
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
    This Model is very similar to the PerceptronLayer Model. The only big
    difference is that the activation_func is set to a softmax function.

    :param insize: Required int. The input size. If the input of this model is
        a vector, insize will be the vector's length. If the input is a matrix,
        insize will be the length of each row.
    :param outsize: Required int. The output size. This can be thought as the
        layer's number of neurons. If the input is a vector, outsize is the
        length of the output vector. If the input is a matrix, outsize is the
        length of each row of the output matrix.
    :param init: The weight initializer for the layer weights. Default is
        XavierInitializer.
    :param W_softmax: Optional numpy ndarray with ndim=2. If set, the weights of
        this layer are not initialized with `init`. Instead they are set to this
        parameter's value.
    :param b: Optional numpy ndarray with ndim=1. If set, the bias vector of
        this layer are not initialized with zeros. Instead they are set to this
        parameter's value.

    Inputs:
        A single input x with x.ndim=1 or x.ndim=2. x.shape[-1] should be equal
            to the insize parameter

    Outputs:
        A single output y with y.ndim=x.ndim and y.shape[-1] equal to the
            outsize parameter.

    Tunable Parameters:
        W_softmax - Weight matrix
        b_softmax - Bias vector
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="insize",
            required=True,
            value_type=int,
        )
        ops.add(
            name="outsize",
            required=True,
            value_type=int,
        )
        ops.add(
            name="init",
            value_type=init.Initializer,
            value=init.XavierInitializer()
        )
        ops.add(
            name="W_softmax",
            value_type=np.ndarray,
        )
        ops.add(
            name="b_softmax",
            value_type=np.ndarray,
        )
        return ops

    def init_params(self):
        options = self.options
        in_dim = options.get('insize')
        out_dim = options.get('outsize')
        init = options.get('init')

        W_softmax = options.get('W_softmax')
        if W_softmax is None:
            W_softmax = init((in_dim, out_dim))
        W_softmax = theano.shared(
            value=W_softmax,
            name='W_softmax',
            borrow=True
        )

        b_softmax = options.get('b_softmax')
        if b_softmax is None:
            b_softmax = np.zeros(out_dim, dtype=theano.config.floatX)
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

        y = T.nnet.softmax(T.dot(inps, W_softmax) + b_softmax)
        if inps.ndim == 1:
            y = y.reshape((y.shape[1],))

        return [y]


class RecursiveNeuralNetwork(Model):
    """A Recursive Neural Network.
    Each composition in the recursive NN is made with the model passed by
    parameter. This makes this model very flexible.
    One important detail about this model is that every output of the
    composition model should have the same shape as its inputs, since they are
    used recursively.

    :param comp_model: Optional Model. Composition Model to be used throught the
        tree. This Model can have any number of inputs/outputs. If the
        comp_model has 2 outputs, it should also have 2 inputs of the same
        shapes. This is necessary to perform the recursive steps. If no
        comp_model is set, the default composition is a single PerceptronLayer
        applied to the concatenated childrens' vectors.
    :param insize: Optional int. This parameter is only used when no comp_model
        is set. In this case the insize parameter is required and used to
        instantiate the default composition Model (a PerceptronLayer).

    Inputs:
        The first input is a matrix that defines the composition tree to be
            followed. This matrix specifies the composition tree in the
            following way:
                All nodes of the tree are numbered. The leafs of the tree are
                numbered from left to right, starting from 0. Then all the
                internal nodes are numbered sequentialy in any order. We will
                call these nodes numbers as nodes ids. We will also reference
                the number of leafs as leafs_nr from now on.
                Each column of the tree matrix represents an internal node of
                the tree, where the first column represents the node with id
                leafs_nr, the second column represents the node with id
                leafs_nr+1 and so on.
                Each element of each column vector of the tree matrix specifies
                a child of that column. This element specifies the child using
                the child's node id. This node id can be from a leaf node or
                from another internal node.
                Let's have a look at an example:
                comp_tree = [
                    [1, 2],
                    [4, 3],
                    [0, 5]
                ]
                This tree matrix with 4 leaf nodes can be visualized as:

                      6
                     / \ 
                    /   5
                   /   / \ 
                  /   4   \ 
                 /   / \   \ 
                0   1   2   3

            This Model can also have other inputs. These inputs will be treated
            as leaf nodes of the tree. The basic requirement for these inputs is
            that the shape[0] property of every input should be equal. This way
            they are all passed to the composition model in the right way.
            For example, imagine a composition tree that composes words
            representions into phrases representations. If every word is
            represented by a matrix and a vector (just like the MV-RNN model
            from Socher et al. [2013]), then the inputs of this Model should be
            a matrix and a 3DTensor, both with shape[0] equal to the sentence
            length.

    Outputs:
        This Model outputs all nodes outputs, including the leaf nodes. For the
            MV-RNN example, this Model would have 2 outputs. One output would be
            a matrix representing every vector representation of the tree. The
            other output would be a 3DTensor representing every matrix
            representation of the tree.

    Tunable Parameters:
        The tunable parameters of this Model are just the comp_model's tunable
            parameters. If no comp_model is given, the tunable parameters are
            the same as the PerceptronLayer
    """

    @staticmethod
    def init_options():
        ops = utils.Options()
        ops.add(
            name="comp_model",
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
            updates = theano.updates.OrderedUpdates()
            if isinstance(model_out, tuple):
                updates += model_out[1]
                model_out = model_out[0]

            new_partials = []
            for p, o in zip(partials, model_out):
                new_partials.append(T.set_subtensor(p[index], o))

            return new_partials, updates

        partials = []
        for o in x:
            shape = []
            for i in range(1, o.ndim):
                shape.append(o.shape[i])

            partial = T.alloc(0., o.shape[0] + comp_tree.shape[0], *shape)
            partial = T.set_subtensor(partial[:o.shape[0]], o)
            partials.append(partial)

        #Execute the scan
        h, updates = theano.scan(
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

        return h, updates

class Recurrence(object):
    """An abstract class to Models that implement a recurrence function.
    This is useful to let the RecurrentNeuralNetwork know what initial inputs
    the recurrence model is expecting
    """

    def get_h0(self):
        """This method lets the RecurrentNeuralNetwork know what the initial
        inputs are.

        :returns: A theano shared variable or a list of theano shared variables
        """
        raise NotImplemented("Abstract method not implemented")


class SimpleRecurrence(Model, Recurrence):
    """A simple recurrence for a RecurrentNeuralNetwork.
    This Recurrence Model passes the last output and the current input in a
    Perceptron Layer, summing them up after.

    :param insize: The size of the vectors that will be used as input
    :param outsize: The size of the output vector
    :param W: Weight matrix for the current input vector. This matrix has shape
        (insize, outsize). If not specified, the matrix will be randomly
        initialized with the `init` parameter.
    :param b: Vector of shape (outsize,) representing the bias vector. If not
        specified, the vector will be initialized with zeros
    :param W_h: The weight matrix for the last output. This matrix has shape
        (outsize, outsize). If not specified, the matrix will be randomly
        initialized with the `init` parameter.
    :param init: The weight initializer used for the tunable parameters. The
        default initializer is a XavierInitializer.

    Inputs:
        Two vectors. The first with shape (insize,) is the current input vector.
            The second vector, with shape (outsize,) is the last output vector.
    Outputs:
        A single vector with shape (outsize,)

    Tunable Parameters:
        W - Weight matrix for the current input
        b - Bias vector
        W_h - Weight matrix for the last output
    """
    @staticmethod
    def init_options():
        ops = utils.Options()

        ops.add(
            name='insize',
            value_type=int,
            required=True,
        )

        ops.add(
            name='outsize',
            value_type=int,
            required=True,
        )

        ops.add(
            name='W',
            value_type=np.ndarray,
        )

        ops.add(
            name='b',
            value_type=np.ndarray,
        )

        ops.add(
            name='W_h',
            value_type=np.ndarray,
        )

        ops.add(
            name='activation_func',
            value=T.nnet.sigmoid
        )
        ops.add(
            name='init',
            value_type=init.Initializer,
            value=init.XavierInitializer()
        )

        return ops

    def get_h0(self):
        outsize = self.options.get('outsize')
        h0 = np.asarray(np.zeros(shape=(outsize,)), dtype=theano.config.floatX)
        h0 = theano.shared(value=h0, name='h0')
        return h0

    def init_params(self):
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')
        W = self.options.get('W')
        b = self.options.get('b')
        W_h = self.options.get('W_h')
        init = self.options.get('init')

        if W is None:
            W = init((insize, outsize))

        if b is None:
            b = np.zeros(shape=(outsize,), dtype=theano.config.floatX)

        if W_h is None:
            W_h = init((outsize, outsize))

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

class LSTMRecurrence(Model, Recurrence):
    """The LSTM recurrence Model
    This Model implements the Long Short-Term Memory. For more information on
    this kind of recurrence model, please read
    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    This Model only operates on vectors.
    Every gate for the LSTM has two weight matrices. One that operates on the
    current input and other that operates on the previous output. One can tell
    the difference between them by their names. A parameter starting with W
    operates on the current input, while a parameter starting with U operates on
    the previous output.

    :param insize: The length of the input vectors
    :param outsize: Optional length of the output vectors. If not specified,
        the Model assumes outsize==insize
    :param init: The Initializer used to initialize all the weights. The default
        initializer is a XavierInitializer.
    :param Wi: The weight matrix for the input gate. This matrix has shape
        (insize, outsize). If not specified, the matrix is randomly initialized
    :param Wf: The weight matrix for the forget gate. This matrix has shape
        (insize, outsize). If not specified, the matrix is randomly initialized
    :param Wc: The weight matrix for the internal representation. This matrix
        has shape (insize, outsize). If not specified, the matrix is randomly
        initialized
    :param Wo: The weight matrix for the output gate. This matrix has shape
        (insize, outsize). If not specified, the matrix is randomly initialized
    :param Ui: The weight matrix for the input gate. This matrix has shape
        (outsize, outsize). If not specified, the matrix is randomly initialized
    :param Uf: The weight matrix for the forget gate. This matrix has shape
        (outsize, outsize). If not specified, the matrix is randomly initialized
    :param Uc: The weight matrix for the internal representation. This matrix
        has shape (outsize, outsize). If not specified, the matrix is randomly
        initialized
    :param Uo: The weight matrix for the output gate. This matrix has shape
        (outsize, outsize). If not specified, the matrix is randomly initialized
    :param Vo: The weight matrix for the output gate that operates on the
        internal state. This matrix has shape (outsize, outsize). If not
        specified, the matrix is randomly initialized
    :param bi: Bias vector for the input gate, with shape (outsize,). If not
        specified, the vector is initialized with zeros.
    :param bf: Bias vector for the forget gate, with shape (outsize,). If not
        specified, the vector is initialized with zeros.
    :param bc: Bias vector for the internal representation, with shape
        (outsize,). If not specified, the vector is initialized with zeros.
    :param bo: Bias vector for the output gate, with shape (outsize,). If not
        specified, the vector is initialized with zeros.

    Inputs:
        Three vectors. The first with shape (insize,) is the current input. The
            second with shape (outsize,) is the previous output from the
            recurrence. The third with shape (outsize,) is the previous internal
            state from the recurrence.

    Outputs:
        Two vectors, both with shape (outsize,). The first is the output from
        the LSTM cell. The second is the internal state of the LSTM cell.

    Tunable Parameters:
        (For explanations of what each tunable parameter is, read this Model's
        parameters list)
        [Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, Vo, bi, bf, bc, bo]
    """
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
            value_type=int
        )
        opts.add(
            name='init',
            value_type=init.Initializer,
            value=init.XavierInitializer()
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

    def get_h0(self):
        insize = self.options.get('insize')
        outsize = self.options.get('outsize')
        h0 = np.asarray(np.zeros(shape=(outsize,)), dtype=theano.config.floatX)
        h0 = theano.shared(value=h0, name='h0')
        C0 = np.asarray(np.zeros(shape=(outsize,)), dtype=theano.config.floatX)
        C0 = theano.shared(value=C0, name='C0')
        return [h0, C0]

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
        init = opts.get('init')

        if outsize is None:
            outsize = insize
            opts.set('outsize', outsize)

        def make_shared_matrix(p, ins, outs):
            if p is None:
                p = init((ins, outs))
            return theano.shared(value=p, borrow=True)

        def make_shared_vec(p):
            if p is None:
                p = np.zeros(outsize, theano.config.floatX)
            return theano.shared(value=p, borrow=True)

        matrices = [make_shared_matrix(p, insize, outsize)
                    for p in [Wi, Wf, Wc, Wo]]
        matrices += [make_shared_matrix(p, outsize, outsize)
                    for p in [Ui, Uf, Uc, Uo, Vo]]
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
        ot = T.nnet.sigmoid(x_t.dot(Wo) + Ct.dot(Vo) + h_tm1.dot(Uo) + bo)
        ht = ot * T.tanh(Ct)
        return [ht, Ct]


class RecurrentNeuralNetwork(Model):
    """A Recurrent Neural Network
    The purpose of this Model is to apply another Model's 'apply' method
    recurrently on this Model's inputs.
    This Model assumes that the first dimension of its inputs represents the
    passing of time. For example, if this Model has a tensor x as input, x[0]
    represents the input at time 1, x[1] the input at time 2 and so on. For this
    Model to work, every 'shape[0]' of each input must be the same.
    A Model that serves as the Recurrence Model for the RecurrentNeuralNetwork
    can extend the nnb.Recurrence abstract class. This way the
    RecurrentNeuralNetwork can extract the output for time 0. If this can't be
    detected at runtime, the RecurrentNeuralNetwork will request the 'h0'
    parameter to be passed to the constructor.
    The order of the inputs passed to the Recurrence Model is:
        1 - The current time inputs, in order they are given to the
            RecurrentNeuralNetwork.
        2 - The previous time outputs, int the same order the Recurrence Model
            outputs them.

    :param model: The Recurrence Model to be used.
        So the RecurrentNeuralNetwork can detect the h0 for the model, this
        Recurrence Model should extend the nnb.Recurrence class.
        If the 'model' parameter is not specified, the RecurrentNeuralNetwork
        will need the 'insize' and 'outsize' parameters so it can build a
        SimpleRecurrence to be used as the Recurrence Model.
        This Recurrence Model should be able to handle the
        RecurrentNeuralNetwork inputs and the Recurrence Model previous outputs
    :param h0: The output for time 0, i.e. the first value to be used in the
        recurrence. This parameter can be a numpy ndarray or a list of ndarrays.
        This parameter is only needed if the Recurrence Model used doesn't
        extend the nnb.Recurrence class.
    :param insize: Same as the 'insize' for the nnb.SimpleRecurrence Model. This
        is only used if no 'model' parameter is specified. In this case this
        parameter is required.
    :param outsize: Same as the 'outsize' for the nnb.SimpleRecurrence Model.
        This is only used if no 'model' parameter is specified. In this case
        this parameter is required.

    Inputs:
        Any number of inputs with any number of dimensions > 0, as long as the
            length of the first dimension of all inputs are equal. This is
            required because the first dimension of all inputs are understanded
            as the passing of time.

    Outputs:
        The same outputs as the Recurrence Model with an extra dimension. This
            extra dimension puts all outputs from the Recurrence Model in the
            same tensor

    Tunable Parameters:
        h0 - The first values used for the recurrence, in the same order they
            are given to the RecurrentNeuralNetwork, either by the 'h0'
            parameter or the 'get_h0(self)' method from the Recurrence Model.
        The rest of the parameters are the same as the Recurrence Model.
    """
    @staticmethod
    def init_options():
        ops = utils.Options()

        ops.add(
            name='model',
            value_type=Model,
        )

        ops.add(
            name='h0',
            value_type=[np.ndarray, list],
        )

        ops.add(
            name='insize',
            value_type=int,
        )

        ops.add(
            name='outsize',
            value_type=int,
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
                if not isinstance(model, Recurrence):
                    raise ValueError("Unable to infer the initial output " +
                                    "value for the recurrence model. Either " +
                                    "specify the 'h0' parameter or make the " +
                                    "Model extend the Recurrence class, " +
                                    "implementing the 'get_h0(self)' method")
                h0 = model.get_h0()

                if not isinstance(h0, list):
                    h0 = [h0]

            else:
                if isinstance(h0, list):
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
                else:
                    h0 = [theano.shared(value=h0, name='h0', borrow=True)]


        return h0 + model.params

    def _get_inputs(self):
        return self.options.get('model')._get_inputs()

    def apply(self, inputs):
        options = self.options
        model = options.get('model')
        h0 = self.params[:len(self.params) - len(model.params)]

        def one_step(*args):
            return model.apply(list(args))

        h, updates = theano.scan(
            fn=one_step,
            sequences=inputs,
            outputs_info=h0
        )

        if not isinstance(h, list):
            h = [h]

        return h, updates

class ConvolutionalLayer(Model):
    """A simple convolutional layer for a neural network
    This layer implements a convolutional layer that operates on word vectors.
    To implement the convolution neural network described in
    http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf one just needs to define
    some number of instances of this layer, each with a different window size.
    This Model uses the very optimized convolution implemented by theano.

    :param window: The window size. If the convolutional layer is analyzing
        n-grams of size n, this parameter should be set to n. This is a
        required parameter
    :param insize: The size of each word vector. This is a required parameter
    :param outsize: The number of filters to be aplied. This will make the
        output vectors to have this size. This is a required parameter
    :param stride: The 'step size' when performing each filter. The default
        value for this parameter is 1
    :param activation_func: The activation function to be used. The default
        value for this parameter is nnb.activation.sigmoid
    :param W: The matrix of weights to be used. If not specified the matrix will
        be initialized randomly
    :param b: The bias vector. If not specified the vector will be initialized
        with zeros
    :param init: The Initializer used to initialize the weights. The default
        initializer is a XavierInitializer.

    Inputs:
        A matrix, where each line is a word vector of size 'insize'

    Outputs:
        A matrix where each line is a vector of size 'outsize'. Each column is
            the result of a filter applied to 'window' word vectors

    Tunable Parameters:
        W - Matrix of weights
        b - Bias vector
    """
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
            value_type=int,
            required=True
        )
        opts.add(
            name='outsize',
            value_type=int,
            required=True
        )
        opts.add(
            name='activation_func',
            value=nnb.activation.sigmoid
        )
        opts.add(
            name='W',
            value_type=np.ndarray
        )
        opts.add(
            name='b',
            value_type=np.ndarray
        )
        opts.add(
            name='init',
            value_type=init.Initializer,
            value=init.XavierInitializer()
        )

        return opts

    def init_params(self):
        opts = self.options
        insize = opts.get('insize')
        outsize = opts.get('outsize')
        window = opts.get('window')
        init = opts.get('init')

        W = opts.get('W')
        b = opts.get('b')

        if W is None:
            W = init((outsize, insize, window))
        W = theano.shared(value=W, name='W', borrow=True)

        if b is None:
            b = np.zeros(outsize, dtype=theano.config.floatX)
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
            prev[0].dimshuffle('x', 'x', 1, 0),
            W.dimshuffle(0, 'x', 1, 2),
            filter_shape=(outsize, 1, insize, window),
            image_shape=(1, 1, insize, None),
            subsample=(1, stride)
        )
        act = self.options.get('activation_func')
        output = act(conv + b.dimshuffle('x', 0, 'x', 'x'))

        return [output.dimshuffle(1, 2, 3).flatten(ndim=2).dimshuffle(1, 0)]

class MaxPoolingLayer(Model):
    """A max pooling layer
    This layer performs a max pooling layer over a matrix of word vectors. This
    layer is not supposed to work like the max pooling layer for images, because
    one can't set a window size for the height. This is supposed to work on
    entire vectors always.

    :param window: The number of rows to analize in each max operation. This
        parameter is required
    :param ignore_border: If set to True, the last vectors when sliding the
        window will be ignored if they don't fit in a full `window` size. The
        default value for this parameter is False.
        NOTE: If you know the number of line of the input matrix will be a
        multiple of `window`, set this parameter to True. This can result in
        some performance boost.

    Inputs:
        A matrix

    Outputs:
        A matrix. Each row is the vector obtained by the max pooling over a
            window.
    """
    @staticmethod
    def init_options():
        opts = nnb.utils.Options()
        opts.add(
            name='window',
            value_type=int,
            required=True
        )
        opts.add(
            name='ignore_border',
            value_type=bool,
            value=False
        )
        return opts

    def apply(self, prev):
        window = self.options.get('window')
        ignore_border = self.options.get('ignore_border')

        x = prev[0]

        rest = x.shape[0] % window

        if ignore_border:
            x = x[:-rest]
        else:
            pad = T.alloc(float('-inf'), window - rest, x.shape[1])
            x = T.concatenate([x, pad], axis=0)

        x = x.reshape((x.shape[0] // window, window, x.shape[1]))
        return [T.max(x, axis=1)]

class DropoutLayer(PerceptronLayer):
    """Dropout layer
    This Model functions exactly like the Perceptron layer. All the differences
    are listed here. For the whole functionality, read the nnb.PerceptronLayer
    documentation.
    This Model drops the output from a neuron with probability `p`. Dropping a
    neuron is equivalent to setting an output to 0.
    To evaluate a Model using this layer, create another Model that substitutes
    this layer with a PerceptronLayer and do the following:

    perceptron_layer.params = [param / 2 for param in dropout_layer.params]

    Now the Model containing this perceptron_layer SHOULD NOT be trained. The
    attempt to do so will result in a difficult to understand theano error.
    Maybe in the future the action to create an evaluation Model from one that
    contains a DropoutLayer will be automatic.

    :param p: The probability of dropping a neuron. This should be a float. The
        default value is 0.5
    """

    @staticmethod
    def init_options():
        opts = PerceptronLayer.init_options()
        opts.add(
            name='p',
            value_type=float,
            value=0.5
        )
        return opts

    def apply(self, prev):
        #Based on https://github.com/mdenil/dropout/blob/master/mlp.py
        o = super(DropoutLayer, self).apply(prev)
        p = self.options.get('p')
        srng = theano.tensor.shared_randomstreams.RandomStreams(
            nnb.rng.randint(999999))
        mask = srng.binomial(n=1, p=(1 - p), size=o[0].shape)

        return [o[0] * T.cast(mask, theano.config.floatX)]

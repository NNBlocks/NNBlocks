from NN import Model
import numpy as np
import theano
import theano.tensor as T

class PerceptronLayer(Model):

    @staticmethod
    def get_options():
        ops = Model.get_options()
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
            name="model_supports_batch",
            value=True,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
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
            #TODO Take this next line out. It's only for the Socher's model
            W += np.vstack([np.eye(insize/2), np.eye(insize/2)])
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

    def generate_input(self):
        inp = T.vector("layer_input")
        return [inp]

    def generate_output(self, inputs):
        W = self.params[0]
        b = self.params[1]
        inp = inputs[0]

        return T.tanh(inp.dot(W) + b)


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

    def generate_input(self):
        return [T.matrix("softmax_in")]

    def generate_output(self, inputs):
        inps = inputs[0]
        W_softmax = self.params[0]
        b_softmax = self.params[1]

        return T.nnet.softmax(T.dot(inps, W_softmax) + b_softmax)


class RecursiveNeuralNetwork(Model):

    @staticmethod
    def get_options():
        ops = Model.get_options()
        ops.add(
            name="comp_model",
            description="The composition model to be used in the tree",
            value_type=Model
        )
        ops.add(
            name="word_vecs",
            required=True,
            value_type=np.ndarray,
            description="""Word vectors used for the compositionality"""
        )
        ops.add(
            name="model_supports_batch",
            value=False,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
        )

        return ops
    """
    A Recursive Neural Network. 
    Each composition in the recursive layer is made with the model passed by
    options.
    """
    def init_params(self):
        options = self.options
        word_vecs = options.get('word_vecs')
        word_dim = word_vecs.shape[1]
        comp_model = options.get('comp_model')

        if comp_model is None:
            comp_model = PerceptronLayer(insize=word_dim * 2, outsize=word_dim)
            options.set('comp_model', comp_model)

        word_vecs = theano.shared(
            value=word_vecs,
            name="word_vecs",
            borrow=True
        )

        return [word_vecs] + comp_model.params

    def generate_input(self):
        sentence = T.ivector("sentence")
        comp_tree = T.imatrix("comp_tree")
        return [sentence, comp_tree]

    def generate_output(self, inputs):
        sentence = inputs[0]
        comp_tree = inputs[1]
        word_vecs = self.params[0]
        comp_model = self.options.get('comp_model')

        x = word_vecs[sentence, :]

        #Stanford NLP does this. Not described in the paper.
        x = T.tanh(x)

        #Composition function for two word_vecs
        def compose(u, v):
            stack = T.concatenate([u, v], axis=0)
            out = comp_model.generate_output(stack)
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

        return h


class SiameseNetwork(Model):
    @staticmethod
    def get_options():
        ops = Model.get_options()

        ops.add(
            name="model",
            required=True,
            value_type=Model,
            description="Model to be used for the siamese network"
        )

        return ops
    def init_params(self):
        options = self.options

        return options.get('model').params

    def generate_input(self):
        inputs = self.options.get('model').input
        inputs += self.options.get('model').generate_input()
        return inputs

    def generate_output(self, inputs):
        out1 = self.options.get('model').output
        out2 = self.options.get('model').generate_output(inputs[len(inputs)/2:])

        return T.dot(out1, out2) / (out1.norm(2) * out2.norm(2))

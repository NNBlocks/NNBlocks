import theano
import theano.tensor as T
import numpy as np
from nn import Model

class RecursiveNeuralTensorNetwork(Model):

    @staticmethod
    def get_options():
        ops = Model.get_options()
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
        ops.add(
            name="V",
            value_type=[np.ndarray],
            description="""The tensor used for the compositionality model"""
        )
        ops.add(
            name="W",
            value_type=[np.ndarray],
            description="""The matrix used for the compositionality model"""
        )
        ops.add(
            name="b",
            value_type=[np.ndarray],
            description="""Bias vector used for the compositionality model"""
        )

        return ops
    """
    A Recursive Neural Network. 
    The NN consists of a recursive layer and a softmax layer. 
    Each composition in the recursive layer is made with sigmoid([a,b]*W), 
    where a and b are word vecs.
    """
    def init_params(self):
        rng = np.random.RandomState(1234)
        options = self.options
        word_vecs = options.get('word_vecs')
        word_dim = word_vecs.shape[1]

        word_vecs = theano.shared(
            value=word_vecs,
            name="word_vecs",
            borrow=True
        )

        V = options.get('V')
        if V is None:
            V = np.asarray(
                rng.uniform(
                    low=-1/np.sqrt(4*word_dim),
                    high=1/np.sqrt(4*word_dim),
                    size=(word_dim, 2*word_dim, 2*word_dim)
                ),
                dtype=theano.config.floatX
            )
        V = theano.shared(
            value=V,
            name='V',
            borrow=True
        )

        W = options.get('W')
        if W is None:
            W = np.asarray(
                    rng.uniform(
                        low=-1/np.sqrt(2*word_dim),
                        high=1/np.sqrt(2*word_dim),
                        size=(2*word_dim, word_dim)
                    ),
                    dtype=theano.config.floatX
                )
            W += np.vstack([np.eye(word_dim), np.eye(word_dim)])
        W = theano.shared(
            value=W,
            name='W',
            borrow=True
        )

        b = options.get('b')
        if b is None:
            b = np.zeros(word_dim, dtype=theano.config.floatX)
        b = theano.shared(
            value=b,
            name='b',
            borrow=True
        )

        return [word_vecs, V, W, b]

    def generate_input(self):
        sentence = T.ivector("sentence")
        comp_tree = T.imatrix("comp_tree")
        return [sentence, comp_tree]

    def generate_output(self, inputs):
        sentence = inputs[0]
        comp_tree = inputs[1]
        word_vecs = self.params[0]
        V = self.params[1]
        W = self.params[2]
        b = self.params[3]
        
        x = word_vecs[sentence, :]
        x = T.tanh(x)

        #Composition function for two word_vecs
        def compose(u, v, V, W, b):
            stack = T.concatenate([u, v], axis=0)
            tensor_part = stack.dot(V).dot(stack.T)
            matrix_part = stack.dot(W) + b
            return T.tanh(tensor_part + matrix_part)

        #One theano.scan step in the RNN feedforward
        def one_step(children, index, partial, V, W, b):
            return T.set_subtensor(
                        partial[index],
                        compose(
                            partial[children[0]],
                            partial[children[1]],
                            V,
                            W,
                            b
                        )
                    )

        #Allocate partial results matrix. Each line will hold a node's value
        partial = T.alloc(0., x.shape[0] + comp_tree.shape[0], x.shape[1])
        
        #Set the first n nodes to be the phrase's word_vecs
        partial = T.set_subtensor(partial[:x.shape[0]], x)
        
        #Execute the scan
        h_1, _ = theano.scan(
                        fn = one_step,
                        outputs_info=partial,
                        sequences=[
                            comp_tree,
                            T.arange(
                                x.shape[0],
                                x.shape[0] + comp_tree.shape[0]
                            )
                        ],
                        non_sequences=[V, W, b]
                    )

        #Get the last iteration's values
        h_1 = h_1[-1]

        return h_1

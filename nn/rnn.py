import theano
import theano.tensor as T
import numpy as np

class RNN:
    """
    A Recursive Neural Network. 
    The NN consists of a recursive layer and a softmax layer. 
    Each composition in the recursive layer is made with sigmoid([a,b]*W), 
    where a and b are word vecs.
    """
    def __init__(self, out_dim, word_vecs, learning_rate=0.1, V=None, W=None,
                b=None, W_softmax=None, b_softmax=None, L1_reg=None, 
                L2_reg=None):
        """
        Initializes the network.
        This initialization can take some time, since some theano functions will
        be compiled.
        Parameters:
        - `out_dim`: The output dimension, i.e. the number of classes the NN 
                will classify
        - `word_vecs`: Matrix-like object containing all word vectors, one per 
                line
        - `learning_rate`: Learning rate used for training
        - `W`,`W_softmax` and `b_softmax`: Pre-trained network parameters

        """
        rng = np.random.RandomState(1234)
        word_dim = word_vecs.shape[1]
        if L2_reg is None:
            L2_reg = [0., 0., 0., 0., 0., 0.]
        if L1_reg is None:
            L1_reg = [0., 0., 0., 0., 0., 0.]
        
        #RNN inputs
        phrase = T.ivector("phrase")
        comp_tree = T.imatrix("comp_tree")

        #Initialize RNN parameters
        word_vecs = theano.shared(
                    value=word_vecs,
                    name="word_vecs",
                    borrow=True
                )
        if V is None:
            V = np.asarray(
                    rng.uniform(
                        low = -1/np.sqrt(4*word_dim),
                        high = 1/np.sqrt(4*word_dim),
                        size = (word_dim, 2*word_dim, 2*word_dim)
                    ),
                    dtype=theano.config.floatX
                )
        V = theano.shared(
                value=V,
                name='V',
                borrow=True
            )
        if W is None:
            W  = np.asarray(
                    rng.uniform(
                        low =  -1/np.sqrt(2*word_dim),
                        high = 1/np.sqrt(2*word_dim),
                        size = (2*word_dim, word_dim)
                    ),
                    dtype=theano.config.floatX
                )
            W += np.vstack([np.eye(word_dim),np.eye(word_dim)])
        W = theano.shared(
                value=W,
                name='W',
                borrow=True
            )
        if b is None:
            b = np.zeros(word_dim,dtype=theano.config.floatX)
        b = theano.shared(
                value=b,
                name='b',
                borrow=True
            )
        if W_softmax is None:
            W_softmax = np.asarray(
                    rng.uniform(
                        low =  -1/np.sqrt(word_dim),
                        high = 1/np.sqrt(word_dim),
                        size = (word_dim, out_dim)
                    ),
                    dtype=theano.config.floatX
                )
        W_softmax = theano.shared(
                value=W_softmax,
                name='W_softmax',
                borrow=True
            )
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

        #Now x will be the matrix of word_vecs for that phrase
        x = word_vecs[phrase,:]
        #Stanford NLP code does this...
        x = T.tanh(x)


        #Composition function for two word_vecs
        def compose(u,v,V,W,b):
            stack = T.concatenate([u,v], axis=0)
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

        #Execute the softmax layer for each node
        h_2 = T.nnet.softmax(T.dot(h_1, W_softmax) + b_softmax)


        #Compile feedforward function
        self.__do_feedforward = theano.function([phrase, comp_tree], h_2)

        #Now for the training part
        from nn.optimize import adagrad
        from nn.cost import neg_log_likelihood_summed

        #- `y`: Int label representing the expected result for each node
        y = T.ivector('y')

        #Stack the parameters
        params = [V, W, b, W_softmax, b_softmax, word_vecs]

        cost = neg_log_likelihood_summed(h_2, y)
        for param, l1, l2 in zip(params,L1_reg,L2_reg):
            cost += l1 * abs(param).sum()
            cost += l2 * T.sqr(param).sum()

        params_grad = [
            T.grad(
                cost = cost,
                wrt = param
            )
            for param in params
        ]


        #Compile the online learning function
        self.__do_train, adagrad_hist = adagrad(
                        inputs=[phrase,comp_tree,y],
                        grads=params_grad,
                        params=params,
                        learning_rate=learning_rate
                    )

        adagrad_reset_update = [(hist, T.zeros_like(hist))
                                for hist in adagrad_hist]

        #Reset adagrad's history
        self.reset_adagrad = theano.function(
            inputs=[],
            outputs=[],
            updates=adagrad_reset_update
        )


        #Compile a function that returns the grads of the params. Usefull for
        #batch learning
        self.__do_grads = theano.function(
            inputs=[phrase,comp_tree,y],
            outputs=params_grad
        )

        self.__do_cost = theano.function(
            inputs=[phrase, comp_tree, y],
            outputs=cost
        )

        #Compile a function that, given the params grads, apply adagrad with 
        #the same adagrad history as self.train. Usefull for batch learning
        self.__do_train_using_grads, _ = adagrad(
            inputs=params_grad,
            params=params,
            grads=params_grad,
            hist=adagrad_hist,
            learning_rate=learning_rate
        )

        #Keep the params
        self.params = params

    def batch_train(self, phrases, comp_trees, ys):
        """
        This function takes lists of input parameters for the network (as 
        described in #feedforward(phrase,comp_tree) and a list of labels 
        and trains the network using a batch learning approach with adagrad.
        """
        grads = [[] for param in self.params]
        for phrase, comp_tree, y in zip(phrases,comp_trees,ys):
            grads_i = self.__do_grads(phrase, comp_tree, y)
            for g, gi in zip(grads,grads_i):
                g.append(gi)


        grads = [np.mean(grad, axis=0) for grad in grads]
        self.__do_train_using_grads(*grads)

    def cost(self, sentence, comp_tree, y):
        return self.__do_cost(sentence, comp_tree, y)

    def feedforward(self, phrase, comp_tree):
        """Do a feedforward pass in the entire network.
        - `phrase`: Vector containing the index of the $n$ word vectors 
                (according to the `word_vecs` parameter) of a phrase
        - `comp_tree`: Matrix of shape($n$-1,2) that represents the composition
                tree to be followed. Each row of this matrix represents an
                internal node of the tree. Each column of a row represents a 
                child of that node.
                For example, the row with values [0,1] on it's columns 
                represents a node father of word vec 0 and word vec 1.
                The indexes used in the columns are distributed in the 
                following way:
                    [0,$n$-1] = word vecs of the phrase
                    [$n$,(2*$n$)-1] = internal nodes in the order they appear in
                                        `comp_tree`.
                OBS: The order of the nodes in `comp_tree` is mostly not 
                important, but the last row MUST contain the highest node in the
                tree.
        - Returns: Class predicted
        """
        r = np.argmax(self.__do_feedforward(phrase,comp_tree), axis=1)
        return r

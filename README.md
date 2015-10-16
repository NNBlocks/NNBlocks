#NNBlocks
NNBlocks is a framework to build and train any kind of exotic neural networks architectures. It is made to be:
* Easy to use
* Easy to expand with new model blocks
* Full of auxiliary tools for your neural network

###Example:
Let's say you have 3 input vectors as matrix lines. This will be the NN input.
You want to pass it through a multilayer perceptron, then a recurrent neural network and  finnaly get the last vector of the result. i.e.:
![](http://i.imgur.com/W7pr3iL.png?1)
####Code:
```python
import nnb
import numpy as np

#The input vectors will have size 3
input_vecs = np.arange(9)
input_vecs.resize(3,3)

input_layer = nnb.InputLayer(ndim=2)
p_layer1 = nnb.PerceptronLayer(insize=3, outsize=5)
p_layer2 = nnb.PerceptronLayer(insize=5, outsize=3)
rnn = nnb.RecurrentNeuralNetwork(insize=3, outsize=3)

#Join everything
final_model = input_layer | p_layer1 | p_layer2 | rnn[-1]

#Compiling the feedforward function
feedforward = final_model.compile()
feedforward(input_vecs)
```

###Another (more advanced) example:
Now let's implement a Matrix-Vector Recursive Neural Network (MV-RNN), just as described in Socher's work (http://www.aclweb.org/anthology/D12-1110). This example is way more complicated than the last, but is an awesome example to demonstrate the power of NNBlocks.
First of all, let's understand the architecture:
* The inputs will be a composition tree, represented by a matrix of children indexes, and a vector of word indexes, that will be later translated to word vectors and word matrices
* Each node in a recursive tree will generate it's word vector using the vectors and matrices of it's children. Each node will also produce a new matrix for the vector generated. For details on this step please read the original paper.
* A softmax classifier can be put on top of each node. This classifier will be fed with the generated vector of that node.

Now a step-by-step for the code:

```python
import nnb
import numpy as np

#15 word vectors of size 5
word_vecs = np.random.random(size=(15,5))
#15 word matrices of size 5x5
word_mats = np.random.random(size=(15,5,5))
```
Here we just initialize some of our network's parameters, where
* _word_vecs_ is all of our vocabulary vectors
* _word_mats_ is all of our vocabulary matrices

The rest of the parameters will be handled by NNBlocks' models. We need to declare these parameters here because they don't fit in any of NNBlocks already implemented models. This sounds like a bad thing, but it's actually a big plus that you can plug in your own parameters and use them in the network easily.

```python
comp_tree = nnb.InputLayer(ndim=2, dtype='int32')
word_index = nnb.InputLayer(ndim=1, dtype='int32')
```

Now we initialize our network's inputs.
* _comp_tree_ is the composition tree to be followed by the recursive tree. Each line of the matrix represents a node in the tree and each column represents a child of that node. Since each node has two children, the matrix has shape (n, 2), where n is the number of internal nodes
* _word_index_ is a vector representing the sentence to be composed. Each element in the vector is a word index

Now we shall start to build the actual model. We will first focus on building the model that will do the composition on each node of the tree.

```python
words_comp = \
    nnb.CustomModel(fn=lambda x, X, y, Y: [x.dot(Y), y.dot(X)]) | \
    nnb.ConcatenationModel(axis=0) | \
    nnb.PerceptronLayer(insize=10, outsize=5)
    
matrix_comp = \
    nnb.SliceModel(slice=[1, 3]) | \
    nnb.ConcatenationModel(axis=1) | \
    nnb.PerceptronLayer(insize=10, outsize=5, activation_func=lambda x: x)

comp_model = words_comp & matrix_comp
```
Now _comp_model_, receiving two word vectors and matrices, will output a composed word vector and a composed word matrix. The RecursiveNeuralNetwork will take care of passing these inputs to our model. Again, if you want to understand how the composition is made, read the original paper.

Now the rest of the code:

```python
rnn = nnb.RecursiveNeuralNetwork(comp_model=comp_model)
vec_picker = nnb.Picker(choices=word_vecs)
matrix_picker = nnb.Picker(choices=word_mats)

rnn_inputs = comp_tree & (word_index | vec_picker) & (word_index | matrix_picker)
all = rnn_inputs | rnn
#We are interested only in the composed vectors
composed_vectors = all[0]

#If we wanted to put a softmax layer on top of each node:
#classifications = composed_vectors | nnb.SoftmaxLayer(insize=5, outsize=NUM_CLASSES)
```

At last we finished our model. Now we can use NNBlocks' tools to train our network!

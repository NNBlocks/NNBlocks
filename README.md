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

NNBlocks is built on top of theano.

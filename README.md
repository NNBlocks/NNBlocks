#NNBlocks
W.I.P.

Example:
```python
#Implementing a recursive neural network for sentences
import nnb
import numpy as np

#The word vec size will be 3
word_vecs = np.random.random(size=(15,3))

#Using indexes of word vecs as input. The Picker will take care of getting them
sentence = nnb.InputLayer(ndim=1, dtype='int32')
comp_tree = nnb.InputLayer(ndim=2, dtype='int32')

#Picker to get the right word vecs, given the indexes
picker = nnb.Picker(choices=word_vecs)

#Recursive Neural Network
rnn = nnb.RecursiveNeuralNetwork(insize=3)

#Join everything
final_model = ((sentence | picker) & comp_tree) | rnn

#Compiling the feedforward function
feedforward = final_model.compile()
```

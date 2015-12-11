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

import numpy as np
import theano

class Input(object):
    """Represents a collection user inputs for a Model
    """
    def __init__(self, data, name=None):
        """
        :param data: A list or a ndarray with dtype=object. Each ndarray in the
            list will be padded to the maximum length found in the data. After,
            the whole data will be turned into a theano shared variable.
        :param name: Name of the user input. This can be useful to identify this
            input later in the training
        """
        if not isinstance(data, list) or (isinstance(data, np.ndarray)
                                            and data.dtype != 'object'):
            raise ValueError("The data parameter should either be a list or " +
                            "a numpy ndarray with dtype=object")

        self.name = name

        max_size = 0
        for inp in data:
            if not isinstance(inp, np.ndarray) or inp.dtype == 'object':
                raise ValueError("The data parameter should contain only " +
                                "ndarrays with dtype!=object. " +
                                "{0} found".format(type(inp)))
            size = len(inp)
            if size > max_size:
                max_size = size

        padding_matrix = np.zeros(shape=(len(data), max_size), dtype='int8')

        for i in range(len(data)):
            inp = data[i]
            padding_matrix[i, :len(inp)] = 1
            if len(inp) < max_size:
                pad = max_size - len(inp)
                ndarray_pad = np.zeros(shape=(pad,) + inp.shape[1:],
                                        dtype=inp.dtype)
                data[i] = np.concatenate([inp, ndarray_pad], axis=0)
            print data[i]

        data = np.asarray(data)
        self.data = theano.shared(value=data, borrow=True)
        self.padding = theano.shared(value=padding_matrix, borrow=True)

    def __str__(self):
        string = "Input object \"{0}\" =\n{1}".format(self.name,
                                                    self.data.get_value())
        return string

    def __len__(self):
        return self.data.get_value().shape[0]

class Dataset(object):
    """Represents a dataset for training a Model.
    The dataset consists basically of collections of user inputs
    """

    def __init__(self, inputs):
        """
        :param inputs: A list of nnb.data.Input instances.
        """
        if not isinstance(inputs, list):
            raise ValueError("The inputs parameter should be a list")

        import collections
        inputs_dict = collections.OrderedDict()
        names_counter = 0
        inputs_are_named = True
        size = None
        for inp in inputs:
            if not isinstance(inp, Input):
                raise ValueError("The inputs parameter should only contain " +
                                "nnb.data.Input instances")

            if size is not None:
                if len(inp) != size:
                    raise ValueError("The nnb.data.Dataset needs the same " +
                                    "number of every user input")
            else:
                size = len(inp)

            if inp.name is not None:
                inputs_dict[inp.name] = inp
            else:
                inputs_dict["input_{0}".format(names_counter)] = inp
                names_counter += 1
                inputs_are_named = False

        self._inputs = inputs_dict
        self._all_inputs_are_named = inputs_are_named

    def get_input(self, name):
        return self._inputs[name]

    def all_inputs_are_named(self):
        return self._all_inputs_are_named

    def get_inputs(self):
        return list(self._inputs.values())

    def to_givens(self, inputs, index=0, batch_size=None):
        if batch_size:
            batch_size = len(self)
        givens = {}
        si = index * batch_size
        ei = si + batch_size
        if self.all_inputs_are_named():
            for inp in inputs:
                print inp
                print self.get_input(inp.name).data
                givens[inp] = self.get_input(inp.name).data[si:ei]
        else:
            givens = {i: d.data[si:ei] for i, d in
                    zip(inputs, self.get_inputs())}

        return givens

    def shuffle(self):
        order = nnb.rng.shuffle(np.arange(len(self)))
        for inp in self.get_inputs():
            inp.data = inp.data[order]

    def __len__(self):
        return len(self._inputs.values()[0])

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

import nnb
import nnb.utils as utils
import theano.tensor as T

def _reg_dict(d):
    r = {}
    for key, value in d.items():
        if isinstance(key, tuple):
            for param in key:
                r[param] = value
        else:
            r[key] = value

    return r

def _reg_opt(param, model):
    if isinstance(param, dict):
        param = _reg_dict(param)
    elif isinstance(param, list):
        #pad missing model params
        param = param + [0.] * (len(model.params) - len(param))
        param = dict((p, v) for (p, v) in zip(model.params, param))
    else:
        param = dict((p, param) for p in model.params)

    return param

class Trainer(object):
    """An abstract class for adjusting tunable parameters
    Any class that adjusts a Model's tunable parameters extends this class.
    The way Trainers adjust tunable parameters is by trying to minimize the
    Model's single output. That way the Model's cost is actually part of the
    Model itself.
    Just like the nnb.Model class, this class also implements the Options
    scheme, where a subclass should not override the __init__ method, but
    override the static method init_options() and return a nnb.utils.Options
    instance. This instance will dictate what the instantiation parameters are.
    """
    options = None
    __io = None
    __expected_output = None

    def __init__(self, **kwargs):
        """Initialization method
        Unlike the nnb.Model class, this __init__ method declares some of its
        own initialization parameters that will serve all Trainers. These are:

        :param model: The Model to be minimized. This parameter is required
        :param L1_reg: A float, dict or list that sets L1 regularization
            parameters. Here is what each possibility does:
                float: All tunable parameters have the same L1 regularization
                    set by this float.
                list: This should be a list of floats with length less or equal
                    than the number of tunable parameters of the Model. The
                    Trainer will assign each of the floats in the list to a
                    tunable parameter in the same order they appear in the
                    `model.params` property. Then these floats are used as the
                    L1 regularization for the tunable parameters.
                    If the length of the list is less than the number of tunable
                    parameters, than the Trainer will fill the list with the
                    missing 0's.
                dict: This should be a (tuple -> float) dict, where the tuple
                    should contain tunable parameters. The float will tell what
                    the L1 regularization should be for each tunable parameter
                    in the tuple.
        :param L2_reg: A float, dict or list that sets L2 regularization
            parameters. See the above explanation on L1 regularization, as the
            same rules apply.
        """
        options = self.init_options()
        if not isinstance(options, utils.Options):
            raise TypeError("Options should be a NN.utils.Options instance." + \
                            " Got {0} instead".format(type(options)))
        options.add(
            name='model',
            value_type=nnb.Model,
            required=True
        )
        options.add(
            name='L1_reg',
            value_type=[float, dict, list],
            value=0.
        )
        options.add(
            name='L2_reg',
            value_type=[float, dict, list],
            value=0.
        )

        options.set_from_dict(kwargs)
        options.check()
        self.options = options
        self.setup()

    def get_io(self):
        """Returns the Model's inputs and outputs
        This is preferable to calling the Model's get_io method, as this method
        caches its returns.
        """
        if self.__io is not None:
            return self.__io

        self.__io = self.options.get('model').get_io()
        return self.__io

    def get_cost(self):
        """Returns the Model's output plus the specified regularizations
        This method returns a theano variable, so it is normally handled by
        Trainers only.
        """
        options = self.options
        model = options.get('model')
        L1_reg = options.get('L1_reg')
        L2_reg = options.get('L2_reg')

        inputs, output, updates = self.get_io()

        L1_reg = _reg_opt(L1_reg, model)
        L2_reg = _reg_opt(L2_reg, model)

        for param in model.params:
            reg1 = 0.
            reg2 = 0.

            if param in L1_reg:
                reg1 = L1_reg[param]
            if param in L2_reg:
                reg2 = L2_reg[param]

            output += abs(param).sum() * reg1
            output += T.sqr(param).sum() * reg2

        return output

    @staticmethod
    def init_options():
        """Method that declares initialization parameters of the Trainer
        A class that extends the nnb.train.Trainer class should override this
        method to set initialization parameters of the Trainer. This method
        should always be static.
        This method should return a nnb.utils.Options instance. For more info on
        this class, see its documentation.
        """
        return utils.Options()

    def setup(self):
        """Method that initializes a Trainer
        A class that extends the nnb.train.Trainer class should do any
        initialization needed in this method.
        When this method is called, all the initialization parameters set in the
        init_options() method are already checked, so it is safe to access them
        via `self.options` property.
        This is where the theano functions for the training should be built.
        """
        pass

    def train(self, inputs):
        """Method that adjusts the Model's tunable parameters
        Every class that extends the nnb.train.Trainer class MUST override this
        method, as it is the one the does the training.
        After this method returns, the tunable parameters will be adjusted
        according to the inputs cost.

        :param inputs: A list of lists. The outer list is the collection of
            examples to the Model, whereas the inner lists are the user inputs
            the Model expects. For correct execution, these user inputs should
            always be numpy ndarrays.
            For example, let's say the Model being trained here expects the user
            inputs x,y and z. This parameter should be something like:
            [
                [x1, y1, z1],
                [x2, y2, z2],
                #...
                [xn, yn, zn]
            ]
        """
        raise NotImplementedError("The train method is not implemented in " + \
                                    "{0}".format(type(self)))


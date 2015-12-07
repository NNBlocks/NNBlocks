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

import theano
import theano.tensor as T
import nnb.utils as utils

class Model(object):
    """The Model class.
    Everything that has an input and/or generates an output extends this class.
    When creating a new Model, extend this class for maximum reusability.
    """

    options = None
    """nnb.utils.Options instance. These are the Models' parameters
    specification and values.
    """

    params = None
    """List of theano's shared variables. These are the Model's tunable
    parameters.
    """

    def __init__(self, **kwargs):
        """The initialization method for Models
        Models shouldn't override this method, as it takes care of validating
        options and initializing the Model.
        Any initialization should take place in the init_params method.
        When instantiating a Model class, the instance of nnb.utils.Options
        returned by the init_options method will dictate what the Model's
        parameters are. For more info on this, see the init_options method.
        """
        self.options = self.init_options()
        if not isinstance(self.options, utils.Options):
            raise ValueError("The init_options methd should return an " +
                            "initialized instance of nnb.utils.Options")
        self.options.set_from_dict(kwargs)
        self.options.check()

        self.params = self.init_params()
        if not isinstance(self.params, list):
            raise ValueError("The init_params method should return a list of" +
                            " theano shared variables.")

    @staticmethod
    def init_options():
        """The method that defines the Model's parameters.
        When a class overrides this method, it should build a nnb.utils.Options'
        instance with the parameters specifications. This instance should be
        returned by this method.
        To access the nnb.utils.Options instance returned by this method, use
        the options property.
        For example:

            import nnb
            class FooModel(nnb.Model):
                @staticmethod
                def init_options():
                    opts = nnb.utils.Options()
                    opts.add(
                        name='bar_param',
                        value_type=int,
                        required=True
                    )
                    return opts

            f = FooModel()
            #ValueError! Should pass the bar_param parameter
            f = FooModel(bar_param=2.3)
            #ValueError! bar_param should be an int
            f = FooModel(bar_param=2)
            #OK!

        If this method is not overriden, and empty nnb.utils.Options instance
        will be used.
        For more info on options, see the nnb.utils.Options documentation.
        """
        return utils.Options()

    def init_params(self):
        """The method that initializes the Model's tunable parameters.
        Any class that overrides this method should return a list of theano
        shared variables. These shared variables will be the Model's tunable
        parameters during training. To access the parameters returned by this
        method, use the params property.
        If a Model uses another Model for internal computations, this method
        should also include the internal Model's parameters in the parameters
        list, since they won't be accessible during training in any other way
        (see RNNs).
        If this method is not overriden, a empty list will be used as
        parameters.
        For a concrete example, see any Model in the nnb/nn_model.py file.
        """
        return []

    def apply(self, prev):
        """The method that computes the Model's outputs
        This method is responsable for computing the Model's output given a
        number of inputs. Any checks on the number of inputs or input types
        should be made by the class overriding this method.
        If the Model is placed several times in a composed Model, this method
        will be called several times with different inputs, once for each Model
        placement.
        :param prev: A list of theano variables. These are the Model's inputs
        :returns: A list of theano variables. These are the Model's outputs. If
            the Model has some sort of update that it wants to perform, this
            method can return a tuple (list of outputs, updates dict).
        """
        return prev

    def _get_inputs(self):
        """Method that defines the Model's user inputs
        When compiling the function for a composed Model, this method will tell
        what are the function's parameters.
        This method is rarely overriden, because the nnb.InputLayer already
        takes care of defining the user inputs, this is why it has an underscore
        on it's name. The only exception is when a Model wants another Model to
        do some internal computations (see RNNs).
        See this example on how an internal Model should be treated:

            import nnb

            class OuterModel(nnb.Model):
                @staticmethod
                def init_options():
                    opts = nnb.utils.Options()
                    opts.add(
                        name='aux_model',
                        value_type=nnb.Model,
                        required=True
                    )
                    return opts

                def _get_inputs(self):
                    return self.options.get('aux_model')._get_inputs()
        """
        raise NotImplementedError("Couldn't find ways to get the inputs for " +
                                    "{0}.".format(type(self)))

    def get_io(self):
        """Method that scans the Model for inputs and computes the outputs.
        This method should not be overriden.
        :returns: Tuple of length 3, which the first element is a list of theano
            variables representing the Model's user inputs, the second element
            is a theano variable or a list of theano variables representing the
            Model's outputs and the third element is an updates dict, used for
            the theano function.
        """
        inputs = self._get_inputs()
        outputs = self.apply(None)
        updates = theano.updates.OrderedUpdates()
        if isinstance(outputs, tuple):
            updates = outputs[1]
            outputs = outputs[0]
        if len(outputs) == 1:
            outputs = outputs[0]
        return (inputs, outputs, updates)

    def compile(self, **kwargs):
        """Compiles the Model into a function.
        This method should not be overriden.

        :param **kwargs: Every key=value parameter will be passed along to the
            `theano.function` compiler.

        :returns: A callable object that computes outputs, given inputs, the way
        the Model specifies.
        """
        i, o, u = self.get_io()
        return theano.function(inputs=i, outputs=o, updates=u, **kwargs)

    def __and__(self, other):
        """Concatenates the Model with another vertically.
        This is a crucial method to declare complex Models.
        This method should not be overriden.
        For more info on this, see nnb.VerticalJoinModel
        Example:

            m1 = FooModel()
            m2 = BarModel()
            m3 = m1 & m2
            #m3 will now output m1 and m2's outputs

        :param other: The adjacent model
        :returns: A new Model that will pass it's inputs to this and the other
            model and outputs both Models' outputs.
        """
        return VerticalJoinModel(m1=self, m2=other)

    def __or__(self, other):
        """Concatenates the Model with another horizontally.
        This is a crucial method to declare complex Models.
        This method should not be overriden.
        For more info on this, see nnb.HorizontalJoinModel
        Example:

            m1 = FooModel()
            m2 = BarModel()
            m3 = m1 | m2
            #m3 will now output m2's outputs

        :param other: The next Model in the computation pipeline
        :returns: A new Model that will take this Model's outputs and pass them
            on to the other Model inputs.
        """
        return HorizontalJoinModel(m1=self, m2=other)

    def __getitem__(self, val):
        """Slices this Model's output.
        If this Model has a single output, the slice will be applied to the
        theano variable output. If this Model has more than one output, the
        slice will be applied to the output list.
        For more info, see the SliceModel documentation.
        For example:

            m1 = FooModel() #Has a single output
            m2 = BarModel() #Has 3 outputs

            m3 = m1[:5] #Has a single output: A theano variable subtensor
            m4 = m2[:2] #Has 2 outputs: The first 2 outputs of m2
        """
        return self | SliceModel(slice=val)

    def __iter__(self):
        """Method declared to certify that Models are not iterables.
        """
        raise TypeError('NNBlocks Models are not iterable')

class SliceModel(Model):
    """Model that slices it's inputs
    :param slice: Required slice, list, tuple or int object. This will be used
        to determine this Model's outputs
    Inputs:
        Any input of any shape, as long as the the shapes are compatible with
            the slice parameter.
    Outputs:
        If the previous Model outputs a list of length greater than 1, the
            slice parameter will be used to index the list and output the
            result. If the previous Model outputs a list of length 1, the slice
            parameter will be used in the only member of this list, outputing a
            theano subtensor variable.
    """
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name="slice",
            required=True
        )
        return opts

    def apply(self, prev):
        sli = self.options.get('slice')
        if len(prev) == 1:
            return [prev[0][sli]]

        if isinstance(sli, list):
            out = []
            for index in sli:
                out.append(prev[index])
            return out

        if isinstance(sli, int):
            return [prev[sli]]

        return prev[sli]

class InputLayer(Model):
    """Model that declares user inputs
    When put on a composed Model, this Model will add a new user input to the
    Model's compiled function.
    For example:
        import nnb

        inp = nnb.InputLayer(ndim=0)
        custom = nnb.CustomModel(fn=lambda x: x*2)
        model = inp | custom
        func = model.compile()
        func(4)
        # = 8

    :param ndim: Required int parameter. This tells the Model the number of
        dimensions of the new user input.
    :param dtype: Optional parameter that follows theano and numpy's way to
        specify array types. The default value is specified by
        theano.config.floatX
    :param name: Optional parameter that specifies an input name. This is useful
        to identify the user input. Default is 'input'.

    Inputs:
        Any number of inputs of any shape

    Outputs:
        Outputs the user input specified plus any other inputs of this Model.
    """

    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name="ndim",
            required=True
        )
        opts.add(
            name="dtype",
            value=theano.config.floatX
        )
        opts.add(
            name="name",
            value="input"
        )
        return opts

    def _get_inputs(self):
        if hasattr(self, '_inp'):
            return [self._inp]

        ndim = self.options.get('ndim')
        dtype = self.options.get('dtype')
        t = T.TensorType(dtype, (False,) * ndim)

        self._inp = t(self.options.get('name'))

        return [self._inp]

    def apply(self, prev):
        if prev is None:
            prev = []
        return [self._inp] + prev

class VerticalJoinModel(Model):
    """Model that joins two Models vertically
    Joining a Model vertically means that they will both receive the same inputs
    and their outputs will be concatenated in a single list.
    Example:

        m1 = Model1()
        m2 = Model2()
        vertical = VerticalModel(m1=m1, m2=m2)

    Now vertical is a Model that can be visualized as:

                           --> m1 --
                         /           \ 
        previous_model --             --> next_model
                         \           /
                           --> m2 --

    :param m1: Required Model. This will be the first Model to be joined
    :param m2: Required Model. This will be the second Model to be joined

    Inputs:
        Any number of inputs of any shape

    Outputs:
        Concatenation of m1 and m2 outputs, when given this Model's inputs.

    IMPORTANT: This Model is never instantiated directly. The Model '&' notation
                creates a VerticalJoinModel
    """

    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name="m1",
            required=True
        )
        opts.add(
            name="m2",
            required=True
        )

        return opts

    def init_params(self):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        return _uniq_list(m1.params + m2.params)

    def _get_inputs(self):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        inp1 = []
        try:
            inp1 = m1._get_inputs()
        except NotImplementedError:
            pass
        inp2 = []
        try:
            inp2 = m2._get_inputs()
        except NotImplementedError:
            pass

        inps = _uniq_list(inp1 + inp2)

        return inps

    def apply(self, prev):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        out1 = m1.apply(prev)
        updates = theano.updates.OrderedUpdates()

        if isinstance(out1, tuple):
            updates += out1[1]
            out1 = out1[0]

        if not isinstance(out1, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m1)))
        out2 = m2.apply(prev)

        if isinstance(out2, tuple):
            updates += out2[1]
            out2 = out2[0]

        if not isinstance(out2, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m2)))
        if len(updates) == 0:
            return out1 + out2 #lists
        else:
            return out1 + out2, updates

class HorizontalJoinModel(Model):
    """Model that joins two Models horizontally
    Joining a Model horizontally means that the output of the first Model will
    be used as input of the second Model.
    Example:

        m1 = Model1()
        m2 = Model2()
        horizontal = HorizontalModel(m1=m1, m2=m2)

    Now horizontal is a Model that can be visualized as:

        m1 --> m2

    :param m1: Required Model. This will be the first Model to be joined
    :param m2: Required Model. This will be the second Model to be joined

    Inputs:
        Any number of inputs of any shape

    Outputs:
        m2's outputs, when given m1's outputs, when given this Model's inputs

    IMPORTANT: This Model is never instantiated directly. The Model '|' notation
                creates a HorizontalJoinModel
    """

    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name="m1",
            required=True
        )
        opts.add(
            name="m2",
            required=True
        )

        return opts

    def init_params(self):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        return _uniq_list(m1.params + m2.params)

    def _get_inputs(self):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        inp1 = []
        try:
            inp1 = m1._get_inputs()
        except NotImplementedError:
            pass
        inp2 = []
        try:
            inp2 = m2._get_inputs()
        except NotImplementedError:
            pass

        inps = _uniq_list(inp1 + inp2)

        return inps

    def apply(self, prev):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        updates = theano.updates.OrderedUpdates()
        out1 = m1.apply(prev)
        if isinstance(out1, tuple):
            updates += out1[1]
            out1 = out1[0]

        if not isinstance(out1, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m1)))
        out2 = m2.apply(out1)
        if isinstance(out2, tuple):
            updates += out2[1]
            out2 = out2[0]

        if not isinstance(out2, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m2)))
        if len(updates) == 0:
            return out2
        else:
            return out2, updates

class Picker(Model):
    """Model that uses its input to slice a set of choices
    This Model takes its input, being any kind of theano variable, and uses it
    to slice a tensor passed as the parameter choices. This choices parameter
    will be turned into a tunable parameter.
    This Model is very useful to index a matrix of word vectors. For example:

        import numpy as np
        import nnb

        #15 word vecs of size 5
        word_vecs = np.random.uniform(size=(15,5))

        word_indexes = nnb.InputLayer(ndim=1 dtype='int32')
        word_vecs_picker = nnb.Picker(choices=word_vecs)
        model = word_indexes | word_vecs_picker

        model_func = model.compile()
        model_func([1, 6, 12, 7]) #Results in a 4x5 matrix of 4 word vecs

    Now when training the model, the word vecs will also be tunable

    :param choices: Required numpy ndarray. This will be turned into a theano
        shared variable so it can be tunable

    Inputs:
        A single input of any dimension or size, but with dtype intX (X=8, 16,
            32...). This will be used to slice the choices tensor.

    Outputs:
        A single output being the choices sliced using the input

    Tunable Parameters:
        choices - The ndarray passed as parameter when initializing the model.
    """
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name='choices',
            required='True'
        )
        return opts

    def init_params(self):
        choices = self.options.get('choices')
        return [theano.shared(value=choices, borrow=True, name='choices')]

    def apply(self, prev):
        return [self.params[0][prev[0]]]

class ConcatenationModel(Model):
    """Model that concatenate its inputs.
    This model follows the same syntax as theano.tensor.concatenate

    :param axis: Optional int. Inputs will be joined along this axis. The
        default value is 0.

    Inputs:
        Any number of inputs of any shape, as long as they can be concatenated
            along the chosen axis

    Outputs:
        Concatenated inputs
    """
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name='axis',
            value=0,
            value_type=int
        )
        return opts

    def apply(self, prev):
        return [T.concatenate(prev, axis=self.options.get('axis'))]

class CustomModel(Model):
    """Model that uses an arbitrary function to process its inputs
    This Model is the easiest way to create a Model. It takes a function and
    uses it on its inputs.
    An important detail is that the function should only build more of the
    theano's computation graph, i.e. should not use for loops to process the
    inputs directly and should not use functions that are not ready to handle
    theano variables.

    :param params: Optional list of custom tunable parameters. These can be
        simply numpy ndarrays or theano shared variables. Any ndarrays in the
        list will be turned into theano shared variables so they can be tuned
        during training.
    :param fn: Required callable object. This function will be used to process
        this Model's inputs and return its outputs. The parameters' order for
        this function is:
            1 - This Model's inputs, one per parameter.
            2 - The params parameter turned into theano shared variables, one
                per paramer.
        For example:
            import numpy as np
            import nnb

            p = np.asarray([1,3])

            inp1 = nnb.InputLayer(ndim = 1)
            inp2 = nnb.InputLayer(ndim = 1)

            def custom_func(inp1, inp2, p):
                return (inp1 + inp2) / p, (inp1 + inp2) * p

            custom = nnb.CustomModel(params=[p], fn=custom_func)
            final_model = (inp1 & inp2) | custom
            model_func = final_model.compile()
            model_func([2, 1], [5, 2])
            # = ([7, 1], [7, 6])

    Inputs:
        Any number of inputs of any shape

    Outputs:
        The fn's outputs, when given this Model's inputs

    Tunable Parameters:
        Any parameter passed in the params option when this Model was
            initialized. These parameters will be nameless.
    """
    @staticmethod
    def init_options():
        opts = utils.Options()
        opts.add(
            name='fn',
            required=True
        )
        opts.add(
            name='params',
            value=[],
            value_type=list
        )

        return opts

    def init_params(self):
        params = self.options.get('params')
        params_out = []
        for p in params:
            if not isinstance(p, theano.compile.sharedvalue.SharedVariable):
                p = theano.shared(value=p, borrow=True)
            params_out.append(p)
        return params_out

    def apply(self, prev):
        fn = self.options.get('fn')
        a = prev + self.params
        o = fn(*a)
        if not isinstance(o, list):
            if isinstance(o,tuple):
                o = list(o)
            else:
                o = [o]
        return o

def _uniq_list(l):
    seen = set()
    return [x for x in l if not (x in seen or seen.add(x))]

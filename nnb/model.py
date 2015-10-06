import theano
import nnb.utils as utils
import abc

class Model(object):
    """The Model class.
    Everything that has an input that generates an output extends this class.
    """
    __metaclass__ = abc.ABCMeta
    options = None
    params = None

    def __init__(self, options=None, **kwargs):
        if options is None:
            options = self.get_options()
        if options is None:
            raise ValueError("The get_options method should return an Options" +
                            " object.")
        if not isinstance(options, utils.Options):
            raise TypeError("Options should be a NN.utils.Options instance." + \
                            " Got {0} instead".format(type(options)))
        options.set_from_dict(kwargs)
        options.check()
        self.options = options
        self.params = self.init_params()
        if self.params is None or not isinstance(self.params, list):
            raise ValueError("The init_params method should return a list of " +
                            "theano shared variables."
        inputs = self.generate_input()
        if inputs is None or not isinstance(inputs, list):
            raise ValueError("The generate_input method should return a list " +
                            "of theano variables.")
        output = self.generate_output(inputs)

        self.feedforward = theano.function(inputs, outputs)

    @abc.abstractmethod
    def init_params(self):
        """Returns the parameters of the model.
        A child class of Model must implement this method.
        This method should return a list of theano shared variables that will
        be used as parameters for the model.
        These parameters can be accessed later from self.params member.
        """
        return []

    @abc.abstractmethod
    def generate_input(self):
        """Generated a list of suggested inputs for the model.
        A child class of Model must implement this method.
        This method should return a list of theano variables. These variables
        will be used as suggestion for the input of the model. When no
        information about the input of this model is provided by a composite
        model, the inputs returned by this method will be used.
        """
        return []

    @abc.abstractmethod
    def generate_output(self, inputs):
        """Generates the desired output of the model, given an inputs list.
        A child class of Model must implement this method.
        This method should return a theano variable derived from the inputs
        passed as parameters.
        Args:
            inputs (List[theano.TensorType]): List of inputs of the model.
                These inputs won't necessarily be of the same type as returned
                by the generate_input method. This happend when the output of
                another model is used as input for this model.
        """
        return None

    @staticmethod
    def get_options():
        """Configure and returns the Options instance for this model.
        For more information see the nnb.utils.Options docs.
        """
        return utils.Options()

    def __getitem__(self, val):
        return SliceModel(model=self, slice=val)


class SliceModel(Model):
    @staticmethod
    def get_options():
        opts = Model.get_options()
        opts.add(
            name="model",
            required=True,
            value_type=Model,
            description="Model to be sliced."
        )
        opts.add(
            name="slice",
            required=True,
            description="The slice to be used."
        )
        return opts

    def init_params(self):
        opts = self.options
        return opts.get("model").params

    def generate_input(self):
        return self.options.get("model").generate_input()

    def generate_output(self, inputs):
        model_out = self.options.get("model").generate_output(inputs)
        model_slice = self.options.get("slice")
        return model_out[model_slice]


class CompositeModel(Model):
    @staticmethod
    def get_options():
        ops = Model.get_options()
        ops.add(
            name="models",
            required=True,
            value_type=list,
            description="""The models to be composed."""
        )
        return ops

    def init_params(self):
        models = self.options.get('models')

        #The model_supports_batch has to be inferred after the models are set
        supports_batch = True
        for model in models:
            try:
                model_supports_batch = model.options.get('model_supports_batch')
                if not model_supports_batch:
                    supports_batch = False
                    break
            except KeyError:
                supports_batch = False
                break
        self.options.add(
            name="model_supports_batch",
            value=supports_batch,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
        )

        params = []

        for model in models:
            params += model.params

        return params

    def generate_input(self):
        return self.options.get('models')[0].generate_input()

    def generate_output(self, inputs):
        models = self.options.get('models')

        result = models[0].generate_output(inputs)

        for i in xrange(1, len(models)):
            result = models[i].generate_output([result])

        return result

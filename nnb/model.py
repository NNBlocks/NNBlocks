import theano
import nnb.utils as utils
import abc

class Model(object):
    __metaclass__ = abc.ABCMeta
    input = None
    output = None
    options = None
    params = None

    def __init__(self, options=None, **kwargs):
        if options is None:
            options = self.get_options()
        if not isinstance(options, utils.Options):
            raise TypeError("Options should be a NN.utils.Options instance." + \
                            " Got {0} instead".format(type(options)))
        options.set_from_dict(kwargs)
        options.check()
        self.options = options
        self.params = self.init_params()
        self.input = self.generate_input()
        self.output = self.generate_output(self.input)

        self.feedforward = theano.function(self.input, self.output)

    @abc.abstractmethod
    def init_params(self):
        return []

    @abc.abstractmethod
    def generate_input(self):
        return []

    @abc.abstractmethod
    def generate_output(self, inputs):
        return None

    @staticmethod
    def get_options():
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

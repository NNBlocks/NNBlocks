import theano
import NN.utils as utils

class Model(object):
    input = None
    output = None
    options = None
    params = None

    def __init__(self, options=None, **kwargs):
        if options is None:
            options = self.get_options()
        if not isinstance(options,utils.Options):
            raise TypeError("Options should be a NN.utils.Options instance." + \
                            " Got {0} instead".format(type(options)))
        options.set_from_dict(kwargs)
        options.check()
        self.options = options
        self.params = self.init_params()
        self.input = self.generate_input()
        self.output = self.generate_output(self.input)

        self.feedforward = theano.function(self.input, self.output)

    def init_params(self):
        return []

    def generate_input(self):
        return []

    def generate_output(self, inputs):
        return None
    
    @staticmethod
    def get_options():
        return utils.Options()

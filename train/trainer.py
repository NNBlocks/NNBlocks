import NN.utils as utils

class Trainer(object):
    options = None

    def __init__(self, options=None, **kwargs):
        if options is None:
            options = self.get_options()
        if not isinstance(options, utils.Options):
            raise TypeError("Options should be a NN.utils.Options instance." + \
                            " Got {0} instead".format(type(options)))
        options.set_from_dict(kwargs)
        options.check()
        self.options = options
        self.setup()

    @staticmethod
    def get_options():
        return utils.Options()

    def setup(self):
        pass

    def train(self, inputs, expected_outputs):
        raise NotImplementedError("The train method is not implemented in " + \
                                    "{0}".format(type(self)))

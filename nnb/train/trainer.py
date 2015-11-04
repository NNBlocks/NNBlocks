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
    options = None
    __io = None
    __expected_output = None

    def __init__(self, **kwargs):
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
        if self.__io is not None:
            return self.__io

        self.__io = self.options.get('model').get_io()
        return self.__io

    def get_cost(self):
        options = self.options
        model = options.get('model')
        L1_reg = options.get('L1_reg')
        L2_reg = options.get('L2_reg')

        inputs, output = self.get_io()

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
        return utils.Options()

    def setup(self):
        pass

    def train(self, inputs):
        raise NotImplementedError("The train method is not implemented in " + \
                                    "{0}".format(type(self)))


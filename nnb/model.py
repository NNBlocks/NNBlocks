import theano
import theano.tensor as T
import nnb.utils as utils
import abc

class Model(object):
    """The Model class.
    Everything that has an input that generates an output extends this class.
    """
    def __init__(self, **kwargs):
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
        return utils.Options()

    def init_params(self):
        return []

    def apply(self, prev):
        return prev

    def _get_inputs(self):
        raise NotImplementedError("Couldn't find ways to get the inputs for " +
                                    "{0}.".format(type(self)))

    def get_io(self):
        self.params = self.init_params()
        inputs = self._get_inputs()
        outputs = self.apply(None)
        if len(outputs) == 1:
            outputs = outputs[0]
        return (inputs, outputs)

    def compile(self, *args):
        a = self.get_io() + args
        return theano.function(*a)

    def __and__(self, other):
        return VerticalJoinModel(m1=self, m2=other)

    def __or__(self, other):
        return HorizontalJoinModel(m1=self, m2=other)

    def __getitem__(self, val):
        return self | SliceModel(slice=val)

    def __iter__(self):
        raise TypeError('NNBlocks Models are not iterable')

class SliceModel(Model):
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
        return list(set(m1.params + m2.params))

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

        with_duplicates = inp1 + inp2
        without_duplicates = []
        check_set = set()

        for inp in with_duplicates:
            if inp in check_set:
                continue
            check_set.add(inp)
            without_duplicates.append(inp)

        return without_duplicates

    def apply(self, prev):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        out1 = m1.apply(prev)
        if not isinstance(out1, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m1)))
        out2 = m2.apply(prev)
        if not isinstance(out2, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m2)))
        return out1 + out2 #lists

class HorizontalJoinModel(Model):

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
        return list(set(m1.params + m2.params))

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

        with_duplicates = inp1 + inp2
        without_duplicates = []
        check_set = set()

        for inp in with_duplicates:
            if inp in check_set:
                continue
            check_set.add(inp)
            without_duplicates.append(inp)

        return without_duplicates

    def apply(self, prev):
        m1 = self.options.get('m1')
        m2 = self.options.get('m2')
        out1 = m1.apply(prev)
        if not isinstance(out1, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m1)))
        out2 = m2.apply(out1)
        if not isinstance(out2, list):
            raise ValueError(("The model {0} didn't return a list of theano" +
                                " variables.").format(type(m2)))
        return out2

class Picker(Model):
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
        return [theano.shared(value=choices, borrow=True)]

    def apply(self, prev):
        return [self.params[0][prev[0]]]

class ConcatenationModel(Model):
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
            o = [o]
        return o

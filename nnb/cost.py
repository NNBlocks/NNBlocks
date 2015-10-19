import theano.tensor as T
import theano

def cross_entropy(p, y):
    if p.ndim != y.ndim:
        raise ValueError("Cross entropy can only be performed in outputs and" +
                        " expected outputs with the same dimensions")
    if p.ndim == 0:
        return -y * T.log(p)
    elif p.ndim <= 2:
        return -T.mean((y * T.log(p)).T.sum(axis=0))
    elif p.ndim == 3:
        def one_step(i, o):
            return (o * T.log(i)).T.sum(axis=0)
        errors, _ = theano.scan(fn=one_step, sequences=[p, y])
        return -T.mean(errors)
    else:
        raise NotImplemented("cross_entropy can only be performed on ndim up " +
                            "to 3")

def cross_entropy_summed(p, y):
    return -T.sum((y * T.log(p)))

def mean_square_error(p, y):
    if p.ndim == 0 and y.ndim == 0:
        return T.sqr(p - y) / 2
    elif p.ndim == 1 and y.ndim == 1:
        return T.mean(T.sqr(p - y)) / 2
    elif p.ndim == 2 and y.ndim == 2:
        return T.mean(T.sqr(p - y).sum(axis=1)) / 2
    else:
        error_str = "Invalid dimensions for mean square error: {0} and {1}."
        error_str = error_str.format(p.ndim, y.ndim)
        raise ValueError(error_str)

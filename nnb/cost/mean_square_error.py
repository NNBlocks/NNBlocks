import theano.tensor as T

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

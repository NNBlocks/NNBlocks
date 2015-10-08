import theano.tensor as T

def neg_log_likelihood(p, y):
    return -T.mean((y * T.log(p)).T.sum(axis=0))

def neg_log_likelihood_summed(p, y):
    return -T.sum((y * T.log(p)))

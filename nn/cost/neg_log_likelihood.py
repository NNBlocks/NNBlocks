import theano.tensor as T

def neg_log_likelihood(p, y):
    return -T.mean(T.log(p)[T.arange(y.shape[0]),y])

def neg_log_likelihood_summed(p, y):
    return -T.sum(T.log(p)[T.arange(y.shape[0]),y])

import theano.tensor as T

def neg_log_likelihood(p, y):
    return -T.mean(T.log(p).dot(y.T))

def neg_log_likelihood_summed(p, y):
    return -T.sum(T.log(p).dot(y.T))

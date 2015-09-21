import theano.tensor as T
import theano
import numpy as np

def adagrad(inputs, 
            params, 
            cost=None, 
            grads=None, 
            learning_rate=0.01, 
            hist=None):

    params_grad = []
    if cost is not None:
        params_grad = [T.grad(cost=cost,wrt=param) for param in params]
    elif grads is not None:
        params_grad = grads
    else:
        raise ValueError("Either cost or grads should not be None")

    adagrad_hist = []

    if hist is None:
        adagrad_hist = [
            theano.shared(
                np.zeros_like(param.get_value(borrow=True)), 
                name="adagrad_hist_%s"%param,
                borrow=True
            ) for param in params
        ]
    else:
        adagrad_hist = hist

    new_hist = [ah + T.sqr(param_g) 
                    for ah,param_g in zip(adagrad_hist,params_grad)]

    new_grad = [grad / (1e-6 + T.sqrt(ah)) 
                for grad,ah in zip(params_grad,new_hist)]

    import collections
    updates = collections.OrderedDict()
    for param,ng in zip(params,new_grad):
        updates[param] = param - learning_rate * ng

    for hist, nh in zip(adagrad_hist,new_hist):
        updates[hist] = nh

    return theano.function(inputs,cost,updates=updates), adagrad_hist

# NNBlocks is a Deep Learning framework for computational linguistics.
#
#   Copyright (C) 2015 Frederico Tommasi Caroli
#
#   NNBlocks is free software: you can redistribute it and/or modify it under
#   the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   NNBlocks is distributed in the hope that it will be useful, but WITHOUT ANY
#   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#   details.
#
#   You should have received a copy of the GNU General Public License along with
#   NNBlocks. If not, see http://www.gnu.org/licenses/.

import numpy as np
import nnb
import theano

class Initializer(object):
    """Abstract Initializer class
    A class that implements a way to initialize weights should extend this class
    and override the __call__(self, shape) method.
    """

    def __call__(self, shape):
        """The class that extends the Initializer class should override this
        method

        :param shape: Shape of the returned tensor of weights

        :returns: An initialized tensor with the shape specified. This tensor
            should have dtype=theano.config.floatX
        """
        raise NotImplemented("Abstract method not implemented")

class ConstantInitializer(Initializer):
    """Initializes a tensor with a pre-defined value
    """

    def __init__(self, value=0.):
        """
        :param value: Optional float. This parameter will determine what is the
            value the tensor will be filled. Default is 0.
        """
        self.value = value

    def __call__(self, shape):
        """Returns a tensor initialized with the constant `value` and the shape
        specified
        """
        r = np.full(shape, self.value)
        return np.asarray(r, dtype=theano.config.floatX)

class NormalInitializer(Initializer):
    """Initializes a tensor with a Gaussian distribution
    """

    def __init__(self, mean=0., std=1.):
        """
        :param mean: The mean of the Gaussian distribution. Default is 0
        :param std: Standard deviation of the Gaussian distribution. Default is
            1
        """
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        """Returns a tensor initialized with a Gaussian distribution
        """
        r = nnb.rng.normal(loc=self.mean, scale=self.std, size=shape)
        return np.asarray(r, dtype=theano.config.floatX)

class TruncatedNormalInitializer(Initializer):
    """Initializes a tensor with a truncated Gaussian distribution
    The difference between this Initializer and the NormalInitializer is that
    every value drawn from the Gaussian distribution that is further than 2
    times the standard deviation from the mean is discarded and re-drawn.
    """

    def __init__(self, mean=0., std=1.):
        """
        :param mean: The mean of the Gaussian distribution. Default is 0
        :param std: Standard deviation of the Gaussian distribution. Default is
            1
        """
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        """Returns a tensor initialized with a truncated Gaussian distribution
        """
        r = nnb.rng.normal(loc=self.mean, scale=self.std, size=shape)
        r = np.asarray(r, dtype=theano.config.floatX)
        r = np.reshape(r, -1)

        for i in range(len(r)):
            while abs(r[i] - self.mean) > 2 * self.std:
                r[i] = nnb.rng.normal(loc=self.mean, scale=self.std)

        return np.reshape(r, shape)

class UniformInitializer(Initializer):
    """Initializes a tensor with an uniform distribution
    """

    def __init__(self, low=0., high=1.):
        """
        :param low: The lower boundry for the uniform distribution
        :param high: The upper boundry for the uniform distribution
        """
        self.low = low
        self.high = high

    def __call__(self, shape):
        """Returns a tensor initialized with an uniform distribution
        """
        r = nnb.rng.uniform(low=self.low, high=self.high, size=shape)
        return np.asarray(r, dtype=theano.config.floatX)

class XavierInitializer(Initializer):
    """Initializes a tensor with a Xavier initialization function
    This function draws every weight from a Gaussian distribution with mean 0
    and standard deviation 1/(n_in + n_out), where n_in and n_out are the
    shape[-2] and shape[-1] of the tensor, respectively. As a consequence, this
    Initializer doesn't handle tensor with ndim < 2
    """

    def __init__(self, factor=1.):
        """
        :param factor: A scaling factor for the distribution. If this
            Initializer is used for a NN layer, this factor should be adjusted
            depending on the activation function. For sigmoid this should be 1,
            for ReLU this should be sqrt(2). The default value is 1
        """
        self.factor = factor

    def __call__(self, shape):
        """Returns a tensor initialized with the Xavier initialization function
        """
        if len(shape) <= 1:
            raise ValueError("The XavierInitializer only handles tensors with" +
                            " ndim >= 2")
        r = nnb.rng.normal(loc=0., scale=1. / (shape[-1] + shape[-2]),
                            size=shape)
        r *= self.factor
        r = np.asarray(r, dtype=theano.config.floatX)

        return r

class EyeInitializer(Initializer):
    """Initializes a matrix with an identity matrix plus a Gaussian Noise drawn
    from the Xavier initialization function
    """

    def __init__(self, factor=1.):
        """
        :param factor: The scaling factor for the Xavier initialization
            function. If this is 0, the matrix is initialized with an identity
            matrix. Default is 1.
        """
        self.xavier = XavierInitializer(factor=factor)

    def __call__(self, shape):
        """Returns a tensor initialized with an identity matrix plus a Xavier
        initialization function.
        The shape parameter for this Initializer should have ndim=2 and
        shape[0]==shape[1]
        """
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("The shape for the EyeInitializer should have " +
                            "ndim=2 and shape[0]==shape[1]")
        r = np.eye(shape[0], shape[1])
        r += self.xavier(shape)
        return np.asarray(r, dtype=theano.config.floatX)

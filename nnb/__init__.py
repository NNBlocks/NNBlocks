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
rng = np.random.RandomState(1337)

import init
import utils
from model import (
    Model,
    InputLayer,
    Picker,
    ConcatenationModel,
    CustomModel,
    SliceModel
)
import activation
import cost
import train
from nn_model import (
    PerceptronLayer,
    SoftmaxLayer,
    RecursiveNeuralNetwork,
    RecurrentNeuralNetwork,
    SimpleRecurrence,
    LSTMRecurrence,
    ConvolutionalLayer,
    MaxPoolingLayer,
    DropoutLayer
)

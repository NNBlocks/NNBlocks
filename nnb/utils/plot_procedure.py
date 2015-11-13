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

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise type(e)(e.message + '\n\tmatplotlib is needed to run plot ' +
                'procedures')
import numpy as np

plt.ion()
class plot_line(object):
    _figure = None
    _ax = None
    _line = None
    _fn = None

    def __init__(self, fn, label=None, title=None):
        self._figure = plt.figure()
        self._ax = self._figure.add_subplot(111)
        if title is not None:
            self._ax.set_title(title)
        self._ax.grid(True)
        if label is not None:
            self._line, = self._ax.plot([], label=label)
            self._ax.legend(handles=[self._line])
        else:
            self._line, = self._ax.plot([])
        self._fn = fn

    def __call__(self, descriptor):
        plot_result = self._fn(descriptor.last_eval_results)
        old_x = self._line.get_xdata()
        new_x = np.append(old_x, descriptor.epoch_num)
        old_y = self._line.get_ydata()
        new_y = np.append(old_y, plot_result)
        self._line.set_xdata(new_x)
        self._line.set_ydata(new_y)
        self._ax.relim()
        self._ax.autoscale_view()

        self._figure.canvas.draw()
        plt.pause(0.0000000001)

    def get_data(self):
        return self._line.get_xdata(), self._line.get_ydata()

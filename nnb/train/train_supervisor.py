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

import nnb
import time
import nnb.utils as utils
import numpy as np
import theano
import sys

class TrainSupervisor(object):
    
    @staticmethod
    def init_options():
        opts = utils.Options()

        opts.add(
            name='dataset',
            required=True,
            value_type=[np.ndarray, list]
        )
        opts.add(
            name='trainer',
            required=True,
            value_type=nnb.train.Trainer
        )
        opts.add(
            name='eval_dataset',
            value_type=[np.ndarray, list]
        )
        opts.add(
            name='eval_interval',
            value=1,
            value_type=int
        )
        opts.add(
            name='max_no_improve',
            value_type=int
        )
        opts.add(
            name='epochs_num',
            value_type=int,
            required=True
        )
        opts.add(
            name='permute_train',
            value=True,
            value_type=bool
        )
        opts.add(
            name='custom_procedures',
            value=[],
            value_type=list
        )
        opts.add(
            name='batch_size',
            value_type=int
        )
        opts.add(
            name='eval_model'
        )
        opts.add(
            name='eval_model_is_cost',
            value=False
        )
        opts.add(
            name='plot',
            value_type=bool,
            value=False
        )
        return opts

    def __init__(self, **kwargs):
        self.options = self.init_options()
        self.options.set_from_dict(kwargs)
        self.options.check()
        trainer = self.options.get('trainer')
        eval_model = self.options.get('eval_model')

        if eval_model is None:
            eval_model = trainer.options.get('model')
            self.options.set('eval_model_is_cost', True)

        io = eval_model.get_io()
        inp = io[0]
        outp = io[1]

        self.__eval = theano.function(inp, outp)

        eval_dataset = self.options.get('eval_dataset')
        dataset = self.options.get('dataset')

        #This avoids some problems with the shuffle and custom procedures.
        #For example, the user might want to do some evaluation himself with a
        #non-shuffled copy of the outputs. This would generate bugs really hard
        #to detect.
        if eval_dataset is dataset:
            import copy
            eval_dataset = copy.deepcopy(dataset)
            self.options.set('eval_dataset', eval_dataset)

        plot_cost = self.options.get('plot')
        eval_model_is_cost = self.options.get('eval_model_is_cost')

        if plot_cost == True:
            if eval_dataset is None:
                raise ValueError("Can't plot the metric function without an " +
                                "evaluation dataset")
            if not eval_model_is_cost:
                raise ValueError("Can't plot a cost line if the eval_model is" +
                                " not a cost. To set the eval_model as a cost" +
                                ", set the eval_model_is_cost parameter to " +
                                "True")
            import nnb.utils.plot_procedure as plot

            def get_cost(last_eval_results):
                accum_cost = sum(last_eval_results)
                return accum_cost / len(last_eval_results)

            plot_func = plot.plot_line(fn=get_cost, title="Cost")
            self.options.get('custom_procedures').append(plot_func)

    def eval(self, dataset):
        all_out = []
        for ex in dataset:
            cost = self.__eval(*ex)
            all_out.append(cost)

        return all_out

    def train(self):
        opts = self.options
        dataset = opts.get('dataset')
        eval_dataset = opts.get('eval_dataset')
        eval_interval = opts.get('eval_interval')
        trainer = opts.get('trainer')
        model = trainer.options.get('model')
        patience = opts.get('max_no_improve')
        epochs_num = opts.get('epochs_num')
        permute = opts.get('permute_train')
        custom_procedures = opts.get('custom_procedures')
        batch_size = opts.get('batch_size')
        eval_model_is_cost = opts.get('eval_model_is_cost')

        descriptor = TrainingDescriptor()

        if batch_size is None:
            batch_size = len(dataset)

        no_improve = 0
        descriptor.best_eval_error = float('Inf')
        best_params = []
        for epoch in xrange(epochs_num):
            descriptor.epoch_num = epoch + 1
            print '~Epoch {0}~'.format(epoch + 1)
            init_time = time.time()
            if permute:
                nnb.rng.shuffle(dataset)
            iterations = len(dataset) / batch_size
            for i in xrange(iterations):
                si = i * batch_size
                ei = (i + 1) * batch_size
                trainer.train(dataset[si:ei])
                fracs = iterations / 10
                if fracs > 0 and i % fracs == 0:
                    frac = i / fracs
                    print '\r[{0}{1}]'.format('-' * frac, ' ' * (10 - frac)),
                    sys.stdout.flush()
            print ''
            took_time = time.time() - init_time
            print 'Finished. Took {0} minutes.'.format(took_time / 60)
            if eval_dataset is not None and \
                    eval_interval > 0 and (epoch + 1) % eval_interval == 0:
                print 'Evaluating...'.format(epoch + 1)
                descriptor.last_eval_results = self.eval(eval_dataset)
                if eval_model_is_cost:
                    descriptor.last_eval_error = \
                        sum(descriptor.last_eval_results)
                    descriptor.last_eval_error /= \
                        len(descriptor.last_eval_results)
                    print 'Error = {0}'.format(descriptor.last_eval_error)
                    if descriptor.last_eval_error < descriptor.best_eval_error:
                        print 'New best!'
                        descriptor.best_eval_error = descriptor.last_eval_error
                        best_params = [p.get_value().copy()
                                            for p in model.params]
                        no_improve = 0
                    else:
                        no_improve += 1
            try:
                for proc in custom_procedures:
                    if isinstance(proc, tuple):
                        if (epoch + 1) % proc[1] == 0:
                            proc[0](descriptor)
                    else:
                        proc(descriptor)
            except StopTraining:
                break

            if no_improve == patience:
                break

        print 'Finished!'
        if eval_model_is_cost:
            print 'Best error: {0}'.format(descriptor.best_eval_error)
            for p, new_p in zip(model.params, best_params):
                p.set_value(new_p)

        return descriptor.best_eval_error


class StopTraining(Exception):
    pass

class TrainingDescriptor(object):
    """The epoch number"""
    epoch_num = None
    """The list of the last model's output when evaluating"""
    last_eval_results = None
    """The last evaluation error"""
    last_eval_error = None
    """The best evaluation error up to this point"""
    best_eval_error = None

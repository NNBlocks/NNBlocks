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
            value_type=int
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
            name='eval_metric'
        )
        return opts

    def __init__(self, **kwargs):
        self.options = self.init_options()
        self.options.set_from_dict(kwargs)
        self.options.check()
        trainer = self.options.get('trainer')
        cost_func = self.options.get('eval_metric')
        if cost_func is None:
            cost_func = trainer.options.get('cost_func')
        io = trainer.get_io()
        exp = trainer.get_expected_output()
        cost = cost_func(io[1], exp)

        inp = io[0] + [exp]
        outp = [io[1]] + [cost]

        self.__out_and_cost = theano.function(inp, outp)

    def eval(self, dataset):
        total_cost = 0.
        all_out = []
        for ex in dataset:
            out, cost = self.__out_and_cost(*ex)
            all_out.append(out)
            total_cost += cost

        total_cost /= len(dataset)

        return all_out, total_cost

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

        if eval_dataset is dataset:
            import copy
            eval_dataset = copy.deepcopy(dataset)

        descriptor = TrainingDescriptor()

        if batch_size is None:
            batch_size = len(dataset)

        rng = np.random.RandomState(1337)
        no_improve = 0
        descriptor.best_eval_error = float('Inf')
        best_params = []
        for epoch in xrange(epochs_num):
            descriptor.epoch_num = epoch + 1
            print '~Epoch {0}~'.format(epoch + 1)
            init_time = time.time()
            if permute:
                rng.shuffle(dataset)
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
                descriptor.last_eval_results, descriptor.last_eval_error = \
                                                        self.eval(eval_dataset)
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

        print 'Finished! Best error: {0}'.format(descriptor.best_eval_error)
        for p, new_p in zip(model.params, best_params):
            p.set_value(new_p)


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

import matplotlib.pyplot as plt
import numpy as np

class Dashboard:

    def __init__(self, model, acc_func, cost_func):
        plt.ion()
        self.fig = plt.figure()
        self.acc_ax = self.fig.add_subplot(211)
        self.error_ax = self.fig.add_subplot(212)

        self.model = model
        self.acc_func = acc_func
        self.cost_func = cost_func

        self.acc_ax.set_title("Accuracy")
        self.error_ax.set_title("Cost mean")

        self.acc_ax.grid(True)
        self.error_ax.grid(True)

        self.acc_ax.set_xlim((0,4))
        self.error_ax.set_xlim((0,4))

        self.fig.canvas.manager.window.showMaximized()
        self.fig.show()
        plt.pause(0.01)
        self.datasets = {}

    def add_dataset(self, features, labels, name=None):
        if name is None:
            name = len(self.datasets)
        
        acc_line, = self.acc_ax.plot([], label=name)
        error_line, = self.error_ax.plot([])
        self.datasets[name] = (features, labels, [acc_line, error_line])
        self.acc_ax.legend(
            handles=self.acc_ax.get_lines(), 
            bbox_to_anchor=(0.9, 0.4), 
            loc=2, 
            borderaxespad=0.)

    def evaluate(self):
        for dataset_name in self.datasets:
            features, labels, lines = self.datasets[dataset_name]

            error_mean = 0
            acc_mean = 0
            for i in range(len(features)):
                results = self.model.feedforward(*features[i])
                error_mean += self.cost_func(
                    results,
                    labels[i]
                )
                acc_mean += self.acc_func(
                    results,
                    labels[i]
                )
            error_mean /= np.float64(len(features))
            acc_mean /= np.float64(len(features))

            old_data = lines[0].get_ydata()
            lines[0].set_ydata(np.append(old_data, acc_mean))
            lines[0].set_xdata(range(len(lines[0].get_ydata())))

            old_data = lines[1].get_ydata()
            lines[1].set_ydata(np.append(old_data,error_mean))
            lines[1].set_xdata(range(len(lines[1].get_ydata())))

        old_xlim = self.acc_ax.get_xlim()
        self.acc_ax.set_xlim(old_xlim[0],old_xlim[1]+1)
        self.acc_ax.relim()
        self.acc_ax.autoscale_view()
        old_xlim = self.error_ax.get_xlim()
        self.error_ax.set_xlim(old_xlim[0],old_xlim[1]+1)
        self.error_ax.relim()
        self.error_ax.autoscale_view()
        plt.tight_layout()
        plt.pause(0.1)

    def save_pic(self, f):
        self.fig.saveFig(f)



import matplotlib.pyplot as plt
import numpy as np

class Dashboard:

    def __init__(self, class_nr):
        plt.ion()
        self.fig = plt.figure()
        self.acc_ax = self.fig.add_subplot(211)
        self.conf_ax = self.fig.add_subplot(224)
        self.error_ax = self.fig.add_subplot(223)

        self.acc_ax.set_title("Accuracy")
        self.error_ax.set_title("Error mean")

        self.acc_ax.grid(True)
        self.error_ax.grid(True)

        self.class_nr = class_nr

        self.acc_ax.set_xlim((0,4))
        self.error_ax.set_xlim((0,4))

        res = self.conf_ax.imshow(
                            np.zeros(shape=(class_nr,class_nr)), 
                            cmap=plt.cm.jet,
                            interpolation='nearest',
                            vmin=0,
                            vmax=1
                        )
        self.fig.colorbar(
            res, 
            ax=self.conf_ax, 
            ticks=np.array([1,.75,.5,.25,0])
        )
        self.conf_annotations = np.ndarray(
            shape=(self.class_nr, self.class_nr),
            dtype=object
        )
        for i in xrange(self.class_nr):
            for j in xrange(self.class_nr):
                self.conf_annotations[i,j] = self.conf_ax.annotate(
                    str(0), xy=(j,i),
                    horizontalalignment='center',
                    verticalalignment='center'
                )

        self.fig.canvas.manager.window.showMaximized()
        self.fig.show()
        plt.pause(0.01)
        self.datasets = {}
        self.heatmap_name = None

    def add_dataset(self, features, labels, name=None, set_heatmap=False):
        if name is None:
            name = len(self.datasets)
        
        acc_line, = self.acc_ax.plot([], label=name)
        error_line, = self.error_ax.plot([])
        self.datasets[name] = (features, labels, [acc_line,error_line])
        self.acc_ax.legend(
            handles=self.acc_ax.get_lines(), 
            bbox_to_anchor=(0.9, 0.4), 
            loc=2, 
            borderaxespad=0.)
        if set_heatmap:
            self.set_heatmap_name(name)

    def set_heatmap_name(self, name):
        if name not in self.datasets:
            raise ValueError(str(name) + " is not a valid dataset")
        self.heatmap_name = name

        self.conf_ax.set_title(name + ("'s" if name[-1] != "s" else "'") 
                            + " confusion matrix")


    def evaluate(self, network):
        for dataset_name in self.datasets:
            features, labels, lines = self.datasets[dataset_name]
            confusion_matrix = np.zeros(
                (self.class_nr, self.class_nr), 
                dtype=int
            )
            error_mean = 0
            for i in range(len(features)):
                results = network.feedforward(*features[i])
                error_mean += network.cost(features[i][0],features[i][1],labels[i])
                for l, r in zip(labels[i], results):
                    confusion_matrix[l,r] += 1
            error_mean /= np.float64(len(features))
            precision = []
            recall = []
            F1 = []
            for i in range(self.class_nr):
                precision.append(
                    np.float64(confusion_matrix[i,i])
                    / np.float64(confusion_matrix[:,i].sum())
                )
                recall.append(
                    np.float64(confusion_matrix[i,i])
                    / np.float64(confusion_matrix[i,:].sum())
                )
                F1.append(
                    (2*precision[-1]*recall[-1])/(precision[-1]+recall[-1])
                )

            accuracy = (np.float64(np.diag(confusion_matrix).sum())
                        / confusion_matrix.sum())
            old_data = lines[0].get_ydata()
            lines[0].set_ydata(np.append(old_data,accuracy))
            lines[0].set_xdata(range(len(lines[0].get_ydata())))

            old_data = lines[1].get_ydata()
            lines[1].set_ydata(np.append(old_data,error_mean))
            lines[1].set_xdata(range(len(lines[1].get_ydata())))

            if dataset_name == self.heatmap_name:
                self.conf_ax.get_images()[0].set_data(
                    np.float64(confusion_matrix) 
                    / confusion_matrix.sum(axis=1)[:,None]
                )
                for i in xrange(self.class_nr):
                    for j in xrange(self.class_nr):
                        self.conf_annotations[i,j].set_text(
                            str(confusion_matrix[i,j])
                        )

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



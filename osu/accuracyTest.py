# ---------------------------------------------------------------------
# Implements Model Accuracy Tests
#
# Author: Matthew Drnevich
# ---------------------------------------------------------------------

# Classification algorithm:
#   Number of correct classifications = (((y_hat - y) > -0.5)==((y_hat - y) < 0)).sum()
#   Great algorithm if I do say so myself....


from __future__ import division, print_function
from pylearn2.train_extensions import TrainExtension
import theano.tensor as T
from theano import function
import os

# An instance of this should be provided in the Train class as extensions=[<instance>]
class AccuracyMonitor(TrainExtension):

    def __init__(self, model, train, valid, test, save_path):
        self.args = locals()
        print("Writing to {}".format(os.path.splitext(save_path)[0]+"_classification_accuracy.csv"))
        self.save_file = open(os.path.splitext(save_path)[0]+"_classification_accuracy.csv", 'w')

        X = T.dmatrix('X')
        self.y_hat = function([X], model.fprop(X))
        self.accuracy = lambda x_1, x_2: ((((x_1 - train.y) > -0.5)==((x_1 - train.y) < 0)).sum(),
                                          (((x_2 - test.y) > -0.5)==((x_2 - test.y) < 0)).sum())

        self.train_data_size = train.y.shape[0]
        self.test_data_size = test.y.shape[0]
        self.epoch = 0

    # This is called before the first epoch and at the end of each epoch
    def on_monitor(self, model, dataset, algorithm):
        train_accuracy, test_accuracy = self.accuracy(self.y_hat(self.args['train'].X), self.y_hat(self.args['test'].X))

        train_percent = train_accuracy*100/self.train_data_size
        test_percent = test_accuracy*100/self.test_data_size

        self.save_file.write("{},{},{}\n".format(self.epoch, train_percent, test_percent))
        self.save_file.flush()
        self.epoch += 1

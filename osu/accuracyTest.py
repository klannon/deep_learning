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
from theano import function, shared
import os

# An instance of this should be provided in the Train class as extensions=[<instance>]
class AccuracyMonitor(TrainExtension):

    def __init__(self, model, train, valid, test, save_path):
        self.args = locals()
        print("Writing to {}".format(os.path.splitext(save_path)[0]+"_classification_accuracy.csv"))
        self.save_file = open(os.path.splitext(save_path)[0]+"_classification_accuracy.csv", 'w')

        X_1 = T.fmatrix('X1')
        X_2 = T.fmatrix('X2')
        X_3 = T.fmatrix('X3')
        trainY = shared(train.y)
        validY = shared(valid.y)
        testY = shared(test.y)

        self.accuracy = function([X_1, X_2, X_3], [T.sum(((model.fprop(X_1)-trainY) > -0.5)&((model.fprop(X_1)-trainY) < 0)),
                                                   T.sum(((model.fprop(X_2)-validY) > -0.5)&((model.fprop(X_2)-validY) < 0)),
                                                   T.sum(((model.fprop(X_3)-testY) > -0.5)&((model.fprop(X_3)-testY) < 0))])

        self.train_data_size = train.y.shape[0]
        self.test_data_size = test.y.shape[0]
        self.epoch = 0

    # This is called before the first epoch and at the end of each epoch
    def on_monitor(self, model, dataset, algorithm):
        train_accuracy, valid_accuracy, test_accuracy = self.accuracy(self.args['train'].X, self.args['valid'].X, self.args['test'].X)

        train_percent = train_accuracy/self.train_data_size
        test_percent = test_accuracy/self.test_data_size

        self.save_file.write("{},{},{}\n".format(self.epoch, train_percent, test_percent))
        self.save_file.flush()
        self.epoch += 1

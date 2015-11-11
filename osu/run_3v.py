# ---------------------------------------------------------------------
# Trains a neural network on the run_3v data generated by OSU
#
# Authors: Colin Dablain, Matthew Drnevich, Kevin Lannon
# ---------------------------------------------------------------------

from __future__ import print_function
import sys
import os
import theano
import pylearn2
import physics # in order for this to not give an ImportError, need to
# set PYTHONPATH (see README.md)
""" Because I made major changes to the data structures used in
the csv reader and in physics.py, I made a copy of each in this
directory so other files in the root directory of the repo will continue
to run.  Eventully, I would like to move the versions of physics.py and
csv.py that I edited to the the root directory of the repo and use
them across the project. """
print(physics.__file__)
import pylearn2.training_algorithms.sgd
import pylearn2.models.mlp as mlp
import pylearn2.train
import pylearn2.space
from math import floor


def init_train():
    # Initialize train object.
    idpath = os.path.splitext(os.path.abspath(__file__))[0] # ID for output files.
    save_path = idpath + '.pkl'

    # Dataset
    pathToTrainValidData = os.environ['PYLEARN2_DATA_PATH']+os.sep+'train_all_3v_ttbar_wjet.txt'
    pathToTestData = os.environ['PYLEARN2_DATA_PATH']+os.sep+'test_all_3v_ttbar_wjet.txt'
    
    train_fraction = 0.8 # 1700000 in train file for train and valid
    numLabels = 2 # Number of output nodes...softmax interpretation here
    
    dataset_train, dataset_valid, dataset_test  = physics.PHYSICS(pathToTrainValidData,
                                                                  pathToTestData,
                                                                  train_fraction,
                                                                  numLabels=numLabels)
    # For monitoring updates without having to read in a file again.
    monitor_percent = 0.02*train_fraction
    cutoff = floor(monitor_percent*len(dataset_train.X))
    data_dict = {'data': dataset_train.X[:cutoff, :], 'labels': dataset_train.y[:cutoff], 'size': lambda: (cutoff, dataset_train.X.shape[1])}
    dataset_train_monitor = physics._PHYSICS(data_dict, 'monitor', dataset_train.args['benchmark'])


    nvis = dataset_train.X.shape[1] # number of visible layers

    # Model
    model = pylearn2.models.mlp.MLP(layers=[#mlp.Linear(
                                             #    layer_name='h0',
                                             #    dim=100,
                                             #    istdev=.1),
                                             mlp.RectifiedLinear(
                                                 layer_name='r0',
	                                             dim=100,
                                                 istdev=.1),
                                             #mlp.Linear(
                                             #    layer_name='h1',
                                             #    dim=100,
                                             #    istdev=.05),
                                             mlp.RectifiedLinear(
                                                 layer_name='r1',
	                                             dim=100,
                                                 istdev=.05),
                                             #mlp.Tanh(
                                             #    layer_name='h2',
                                             #    dim=100,
                                             #    istdev=.05),
                                             mlp.RectifiedLinear(
                                                 layer_name='r2',
	                                             dim=100,
                                                 istdev=.05),
                                             #mlp.Linear(
                                             #    layer_name='h3',
                                             #    dim=100,
                                             #    istdev=.05),
                                             mlp.RectifiedLinear(
                                                 layer_name='r3',
	                                             dim=100,
                                                 istdev=.05),
                                             #mlp.Sigmoid(
	                                         #    layer_name='y',
	                                         #    dim=2,
	                                         #    istdev=.001)
                                              mlp.Softmax(
                                                  layer_name='y',
                                                  n_classes=2,
	                                              istdev=.001)
                                            ],
                                     nvis=nvis)
    """model = pylearn2.models.mlp.MLP(layers=[mlp.Tanh(
                                                layer_name='h0',
                                                dim=100,
                                                istdev=.1),
                                            mlp.Tanh(
                                                layer_name='h1',
                                                dim=100,
                                                istdev=.05),
                                            mlp.Tanh(
                                                layer_name='h2',
                                                dim=100,
                                                istdev=.05),
                                            mlp.Tanh(
                                                layer_name='h3',
                                                dim=100,
                                                istdev=.05),
                                            # mlp.Sigmoid(
                                            #     layer_name='y',
                                            #     dim=2,
                                            #     istdev=.001)
                                            mlp.Softmax(
                                                layer_name='y',
                                                n_classes=2,
                                                istdev=.001)
                                           ],
                                    nvis=nvis
                                    )"""


    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
                    batch_size=16,
                    learning_rate=.001,
                    monitoring_dataset = {'train':dataset_train_monitor,
                                          'valid':dataset_valid,
                                          'test':dataset_test
                                          }
                )
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 save_path=save_path,
                                 save_freq=100)
    return train

def train(mytrain):
    # Execute training loop.
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print('Using=%s' % theano.config.device) # Can use gpus.
    print('Writing to %s' % logfile)
    print('Writing to %s' % mytrain.save_path)
    sys.stdout = open(logfile, 'w')
    print("opened log file")
    mytrain.main_loop()

if __name__ == "__main__":
    mytrain = init_train()
    train(mytrain)

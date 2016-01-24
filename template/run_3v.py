# ---------------------------------------------------------------------
# Trains a neural network on the run_3v data generated by OSU
#
# Authors: Colin Dablain, Matthew Drnevich, Kevin Lannon
# ---------------------------------------------------------------------

from __future__ import print_function
import sys
import os
import os.path
import theano
import argparse
from math import floor
import physics # in order for this to not give an ImportError, need to
# set PYTHONPATH (see README.md)
from profiling.terminators import Timeout
print(physics.__file__)

import pylearn2
import pylearn2.training_algorithms.sgd
import pylearn2.models.mlp as mlp
import pylearn2.train
import pylearn2.space
import pylearn2.termination_criteria


def init_train(learningRate, batchSize, numLayers, nodesPerLayer,
               timeout=None, maxEpochs=None):               # EDITED
    hostname = os.getenv("HOST", os.getpid()) # So scripts can be run simultaneously on different machines
    results_dir = "{1}{0}results{0}".format(os.sep, os.getcwd())
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    idpath = "{}{}_layers{}".format(results_dir, hostname, numLayers)
    print(idpath)
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
    network_layers = []
    count = 1
    while(count <= numLayers):
        network_layers.append(mlp.RectifiedLinear(
            layer_name=('r%i' % count),
            dim=nodesPerLayer,
            istdev=.1))
        count += 1
    # add final layer
    network_layers.append(mlp.Softmax(
        layer_name='y',
        n_classes=2,
        istdev=.001))
    print(network_layers)
    model = pylearn2.models.mlp.MLP(layers=network_layers,
                                     nvis=nvis)

    # Configure when the training will terminate
    if timeout:
        terminator = Timeout(timeout*60) # Timeout takes an argument in seconds, so timeout is in minutes
    elif maxEpochs:
        terminator = pylearn2.termination_criteria.EpochCounter(max_epochs=maxEpochs)
    else:
        terminator = None

    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
        batch_size=batchSize,
        learning_rate=learningRate,
        monitoring_dataset = {'train':dataset_train_monitor,
                              'valid':dataset_valid,
                              'test':dataset_test
                          },
        # update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
        #     decay_factor=1.0000003, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
        #     min_lr=.000001
        # ),
        termination_criterion=terminator
    )
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 save_path=save_path,
                                 save_freq=100)
    return train


def train(mytrain, batchSize, timeout, maxEpochs):
    # Execute training loop.
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print('Using=%s' % theano.config.device) # Can use gpus.
    print('Writing to %s' % logfile)
    print('Writing to %s' % mytrain.save_path)
    sys.stdout = open(logfile, 'w')
    #
    # print statements after here are written to the log file
    #
    print("opened log file")
    print("Model:")
    print(mytrain.model)
    print("\n\nAlgorithm:")
    print(mytrain.algorithm)
    print("\n\nAdditional Hyperparameters:")
    print("Batch size: %i" % batchSize)
    print("Maximum Epochs: %i" % maxEpochs)
    print("Maximum runtime: %f minutes" % timeout)
    # All of the other  hyperparameters can be deduced from the log file
    mytrain.main_loop()

def run(timeout=None, maxEpochs=100):
    parser = argparse.ArgumentParser()

    ###################################
    ## SET UP COMMAND LINE ARGUMENTS ##
    ###################################

    parser.add_argument("-r", "--learningRate", help="learning rate")
    parser.add_argument("-b", "--batchSize", help="size of each batch "
                        + "(subset of training set)")
    parser.add_argument("-l", "--numLayers",
                        help="number of hidden layers in the network")
    parser.add_argument("-e", "--maxEpochs",
                        help="number of epochs to run for")
    parser.add_argument("-n", "--nodesPerLayer",
                        help="number of nodes per layer")
    args = parser.parse_args()

    ########################
    ## VALIDATE ARGUMENTS ##
    ########################

    # Catches both the TypeError that gets thrown if the argument/flag
    # isn't supplied and the ValueError that gets thrown if an argument
    # is supplied with a type besides that specified in the 'try' block


    ## Default Values if no argument is supplied
    learningRate = .001
    batchSize = 256
    numLayers = 4
    nodesPerLayer = 50
    # maxEpochs is specified in the call to run()

    ## args.learningRate
    try:
        learningRate = float(args.learningRate)
        print("Learning Rate: %f" % learningRate)
    except:
        print("Learning Rate: %f (Default)" % learningRate)

    ## args.batchSize
    try:
        batchSize = int(args.batchSize)
        print("Batch Size: %i" % batchSize)
    except:
        print("Batch Size: %i (Default)" % batchSize)

    ## args.numLayers
    try:
        numLayers = int(args.numLayers)
        print("Number of Layers: %i" % numLayers)
    except:
        print("Number of Layers: %i (Default)" % numLayers)

    ## args.maxEpochs
    try:
        maxEpochs = int(args.maxEpochs)
        print("Number of Epochs to run for: %i" % maxEpochs)
    except:
        print("Number of Epochs to run for: %i (Default)" % maxEpochs)

    ## args.nodesPerLayer
    try:
        numEpochs = int(args.nodesPerLayer)
        print("Number of nodes per layer: %i" % nodesPerLayer)
    except:
        print("Number of nodes per layer: %i (Default)" %
              nodesPerLayer)


    ##########################################
    ## INITIALIZE TRAINING OBJECT AND TRAIN ##
    ##########################################

    learningRate = .001
    batchSize = 256
    numLayers = 4
    nodesPerLayer = 50
    # maxEpochs is specified in the call to run()


    mytrain = init_train(learningRate, batchSize, numLayers,
                         nodesPerLayer, timeout, maxEpochs)
    train(mytrain, batchSize, timeout, maxEpochs)

if __name__ == "__main__":
    run()

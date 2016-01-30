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
from physics import PHYSICS # in order for this to not give an ImportError, need to
# set PYTHONPATH (see README.md)
from profiling.terminators import Timeout
from time import time
#print(physics.__file__)

import pylearn2
import pylearn2.training_algorithms.sgd
import pylearn2.models.mlp as mlp
import pylearn2.train
import pylearn2.space
import pylearn2.termination_criteria

from monitoring import TrainVeil, make_data_slim


def init_train(learningRate, batchSize, numLayers, nodesPerLayer,
               timeout=None, maxEpochs=None):               # EDITED
    hostname = os.getenv("HOST", os.getpid()) # So scripts can be run simultaneously on different machines
    results_dir = "{1}{0}results{0}".format(os.sep, os.getcwd())
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    idpath = "{}{}_layers{}_batchSize{}_nodes{}_time{}".format(results_dir, hostname, numLayers, batchSize, nodesPerLayer, time())
    print(idpath) # ...now our save file name is foolproof
    save_path = idpath + '.pkl'

    # Dataset
    pylearn_path = os.environ['PYLEARN2_DATA_PATH']+os.sep
    path_to_train_X, path_to_train_Y = pylearn_path+'train_all_3v_ttbar_wjet_X.npy', pylearn_path+'train_all_3v_ttbar_wjet_Y.npy'
    path_to_test_X, path_to_test_Y = pylearn_path+'test_all_3v_ttbar_wjet_X.npy', pylearn_path+'test_all_3v_ttbar_wjet_Y.npy'

    dataset_train, dataset_test = PHYSICS(), PHYSICS()
    dataset_train.load_from_file(path_to_train_X, path_to_train_Y)
    dataset_test.load_from_file(path_to_test_X, path_to_test_Y)

    monitor_train, monitor_test = make_data_slim((dataset_train, dataset_test))

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
        monitoring_dataset={'train': monitor_train,
                              'test': monitor_test},
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

    TrainVeil(train)

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
    print("Maximum runtime: %f minutes" % timeout) if timeout else print("Maximum runtime: None")
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
    parser.add_argument("-t", "--timeout",
                        help="how long it should train in minutes")
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
        nodesPerLayer = int(args.nodesPerLayer)
        print("Number of nodes per layer: %i" % nodesPerLayer)
    except:
        print("Number of nodes per layer: %i (Default)" %
              nodesPerLayer)

    ## args.timeout
    try:
        timeout = int(args.timeout)
        print("Timeout: %f" % timeout)
    except:
        print("Timeout: {} (Default)".format(timeout)
              if timeout else "Timeout: None (Default)")


    ##########################################
    ## INITIALIZE TRAINING OBJECT AND TRAIN ##
    ##########################################

    mytrain = init_train(learningRate, batchSize, numLayers,
                         nodesPerLayer, timeout, maxEpochs)
    train(mytrain, batchSize, timeout, maxEpochs)

if __name__ == "__main__":
    run()

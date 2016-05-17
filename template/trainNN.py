# ---------------------------------------------------------------------
# Trains a neural network
#
# Authors: Colin Dablain, Matthew Drnevich, Kevin Lannon
# ---------------------------------------------------------------------

from __future__ import print_function

import os
import os.path
import sys

import argparse
import theano

from physics import PHYSICS # in order for this to not give an ImportError, need to
# set PYTHONPATH (see README.md)
from template.terminators import Timeout, AdjustmentWatcher, TerminatorManager
from time import time

import cPickle

from transformations import transform

def init_train(training_f, testing_f, *args, **kwargs):

    defaults = dict(batchSize=32,
                    numLayers=4,
                    nodesPerLayer=50,
                    learningRate=0.001,
                    saveDir='.',
                    monitorFraction=(0.02, 0.5),
                    numWeightDev=0.1)

    for key, val in kwargs.items():
        if val: defaults[key] = val

    batchSize = int(defaults.get("batchSize"))
    numLayers = defaults.get("numLayers")
    nodesPerLayer = defaults.get("nodesPerLayer")
    learningRate = defaults.get("learningRate")
    saveDir = defaults.get("saveDir")
    monitorFraction = defaults.get("monitorFraction")
    timeout = defaults.get("timeout")
    maxEpochs = defaults.get("maxEpochs")
    benchmark = defaults.get("benchmark")
    width = defaults.get("width")
    slope = defaults.get("slope")
    sigma_w = defaults.get("numWeightDev")

    hostname = os.getenv("HOST", os.getpid()) # So scripts can be run simultaneously on different machines
    if saveDir == '.':
        results_dir = "{1}{0}results{0}".format(os.sep, os.getcwd())
    else:
        results_dir = saveDir if os.path.split(saveDir) is '' else saveDir+os.sep

    if results_dir.split(os.sep)[0] == '.':
        results_dir = os.getcwd() + os.sep + os.sep.join(results_dir.split(os.sep)[1:])

    pwd = ''
    while results_dir.split(os.sep)[0] == '..':
        results_dir = os.sep.join(results_dir.split(os.sep)[1:])
        if pwd == '':
            pwd = os.sep.join(os.getcwd().split(os.sep)[:-1])
        else:
            pwd = os.sep.join(pwd.split(os.sep)[:-1])
    if pwd:
        results_dir = pwd + os.sep + results_dir

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # If you give idpath but no custom name then the new data will be appended to the old log and overwrite the old pkl.
    # If you give customName then that name relative to saveDir will be used to save the model.
    # If you do not provide either of these then defaults based upon your saveDir and time are used.
    if kwargs.get("idpath") and not kwargs.get("customName"):
        idpath = kwargs.get("idpath")
    elif kwargs.get("customName"):
        idpath = results_dir + kwargs.get("customName")
    else:
        idpath = "{}{}_time{}".format(results_dir, hostname, time())
    save_path = idpath + '.pkl'

    benchmark_tr = benchmark if benchmark else training_f[0]
    benchmark_te = benchmark if benchmark else testing_f[0]

    # Dataset
    pylearn_path = os.environ['PYLEARN2_DATA_PATH']+os.sep
    path_to_train_X, path_to_train_Y = [pylearn_path+f for f in training_f]
    path_to_test_X, path_to_test_Y = [pylearn_path+f for f in testing_f]

    dataset_train = PHYSICS(benchmark=benchmark_tr, which_set='train')
    dataset_test = PHYSICS(benchmark=benchmark_te, which_set='test')
    dataset_train.load_from_file(path_to_train_X, path_to_train_Y)
    dataset_test.load_from_file(path_to_test_X, path_to_test_Y)

    dataset_train, dataset_test = transform(dataset_train, dataset_test)

    monitor_train, monitor_test = make_data_slim((dataset_train, dataset_test), monitorFraction)

    nvis = dataset_train.X.shape[1] # number of visible layers
    print(nvis)
    if not kwargs.get("model"):
        # Model
        network_layers = []
        count = 1
        while(count <= numLayers):
            network_layers.append(mlp.RectifiedLinear(
                layer_name=('r{}'.format(count)),
                dim=nodesPerLayer,
                istdev=sigma_w))
            count += 1
        # add final layer
        network_layers.append(mlp.Softmax(
            layer_name='y',
            n_classes=2,
            istdev=.001))
        model = pylearn2.models.mlp.MLP(layers=network_layers,
                                        nvis=nvis)
    else:
        model = kwargs.get("model")
        del model.monitor  # If you try to use a serialized model's monitor then this code crashes.

    # Configure when the training will terminate
    terminator_t = Timeout(timeout*60) if timeout else None  # Timeout takes an argument in seconds, so timeout is in minutes
    terminator_e = pylearn2.termination_criteria.EpochCounter(max_epochs=maxEpochs) if maxEpochs else None
    watch = AdjustmentWatcher(width, slope) if width and slope else None

    terms = TerminatorManager(*[t for t in (terminator_e, terminator_t, watch) if t])

    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
        batch_size=batchSize,
        learning_rate=learningRate,
        monitoring_batches=1,
        monitoring_dataset={'train': monitor_train,
                              'test': monitor_test},
        # update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
        #     decay_factor=1.0000003, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
        #     min_lr=.000001
        # ),
        termination_criterion=terms
    )

    save_freq = kwargs.get("saveFrequency") if kwargs.get("saveFrequency") else 100

    config = dict(training_f=training_f,
                  testing_f=testing_f,
                  benchmark=benchmark,
                  monitorFraction=monitorFraction,
                  numlayers=numLayers,
                  nodesPerLayer=nodesPerLayer,
                  batchSize=batchSize,
                  learningRate=learningRate,
                  idpath=idpath)

    # Train
    trainer = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 save_path=save_path,
                                 save_freq=save_freq,
                                 extensions=[ObserveWeights(),])

    TrainVeil(trainer)

    trainer.path_to_files = (path_to_train_X, path_to_train_Y, path_to_test_X, path_to_test_Y)

    with open(idpath+'.cfg', 'w') as cfg:
        cPickle.dump(config, cfg, cPickle.HIGHEST_PROTOCOL)

    trainer.config = config

    return trainer

# Note that you could use a .cfg file for a different model than your .pkl if you want.
def continue_training(path_to_cfg ,**kwargs):
    with open(path_to_cfg) as f:
        cfg = cPickle.load(f)
    for key, val in kwargs.items():
        if val: cfg[key] = val
    return init_train(model=serial.load(cfg["idpath"]+'.pkl'), **cfg)


def train(mytrain, batchSize=None, timeout=None, maxEpochs=None, *args, **kwargs):
    # Execute training loop.
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print('Using={}'.format(theano.config.device)) # Can use gpus.
    print('Writing to {}'.format(logfile))
    print('Saving to {}'.format(mytrain.save_path))
    sys.stdout = open(logfile, 'a')
    #
    # print statements after here are written to the log file
    #
    print("Opened log file")
    print("Files:")
    for f in mytrain.path_to_files:
        print(f)
    print("\nModel:")
    print(mytrain.model)
    print("\n\nAlgorithm:")
    print(mytrain.algorithm)
    print("\n\nAdditional Hyperparameters:")
    print("Batch size: {}".format(batchSize))
    print("Maximum Epochs: {}".format(maxEpochs))
    print("Maximum runtime: {} minutes".format(timeout))
    # All of the other  hyperparameters can be deduced from the log file
    mytrain.main_loop()
    return mytrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###################################
    ## SET UP COMMAND LINE ARGUMENTS ##
    ###################################

    parser.add_argument("-r", "--learningRate", help="learning rate",
                        type=float, default=None)
    parser.add_argument("-b", "--batchSize", help="size of each batch "
                        + "(subset of training set)", type=int, default=None)
    parser.add_argument("-l", "--numLayers",
                        help="number of hidden layers in the network",
                        type=int, default=None)
    parser.add_argument("-e", "--maxEpochs",
                        help="number of epochs to run for", type=int,
                        default=None)
    parser.add_argument("-n", "--nodesPerLayer",
                        help="number of nodes per layer", type=int,
                        default=None)
    parser.add_argument("-t", "--timeout",
                        help="how long it should train in minutes",
                        type=float, default=None)
    parser.add_argument("-mf", "--monitorFraction", help="a two-tuple with "
                        + "the  training and testing monitoring percents",
                        default=None, type=tuple)
    parser.add_argument("-wd", "--numWeightDev", help="number of std deviations"
                        +" to distribute the weights over", type=float, default=None)
    parser.add_argument("-sf", "--saveFrequency", help="how often the model "
                        +"should be saved and backed up", default=100, type=int)
    parser.add_argument("-c", "--customName", help="name for the log and "
                        +"pkl files from your model", default=None)
    parser.add_argument("-bm", "--benchmark", help="keyword[s] that "
                        +"represent the type of data", default=None)
    parser.add_argument("-s", "--saveDir", help="parent directory to save the"
                        +" results in", default='.')
    parser.add_argument("-w", "--width", help="the number of datapoints to "
                        +"average the slope over", type=int, default=None)
    parser.add_argument("-m", "--slope", help="the slope that determines a "
                        +"plateau i.e. when to stop training", type=float, default=None)
    parser.add_argument("--adjustments", help="adjustments you want to "
                        +"make after each training finishes. Needs to be"
                        +" the switch full name (without --) followed by"
                        +" the adjustment you wish to make. You can do"
                        +" this for however many parameters you wish.", nargs="*", default=None)
    parser.add_argument("--continue", help="continue training based "
                        +"upon a .cfg file", default=None)
    parser.add_argument("--idpath", help="The path to your .pkl file,"
                        +" but WITHOUT the extension", default=None)
    parser.add_argument("--training_f", nargs=2, metavar='train_file',
                        help="the <train_X>.npy and <train_Y>.npy files "
                        +"relative to PYLEARN2_DATA_PATH")
    parser.add_argument("--testing_f", nargs=2, metavar='test_file',
                        help="the <test_X>.npy and <test_Y>.npy files "
                        +"relative to PYLEARN2_DATA_PATH")
    args = vars(parser.parse_args())

    # Creates the adjustments dictionary
    temp = []
    i = 0
    if args["adjustments"]:
        while i < len(args["adjustments"]):
            temp.append(args["adjustments"][i:i+2])
            temp[-1][-1] = float(temp[-1][-1])
            i += 2
    args["adjustments"] = dict(temp)

    # maxEpochs is specified in the call to run()
    print("PARAMETERS GIVEN")
    print("Learning Rate: {}".format(args['learningRate']))
    print("Batch Size: {}".format(args['batchSize']))
    print("Number of Layers: {}".format(args['numLayers']))
    print("Number of Epochs to run for: {}".format(args['maxEpochs']))
    print("Number of nodes per layer: {}".format(args['nodesPerLayer']))
    print("Timeout: {}".format(args['timeout']))
    print("Save Directory: {}{}".format(args['saveDir'], os.linesep))

    ##########################################
    ## INITIALIZE TRAINING OBJECT AND TRAIN ##
    ##########################################
    mytrain = continue_training(args.get("continue"), **args) if args.get("continue") else init_train(**args)
    mytrain = train(mytrain, **args)
    if args["width"] and args["slope"] and args["adjustments"]:
        lr = args["adjustments"].get("learningRate")
        b = args["adjustments"].get("batchSize")
        args.update(mytrain.config)
        args["batchSize"] += b if b else 0
        args["learningRate"] += lr if lr else 0
        if args["maxEpochs"]:
            args["maxEpochs"] -= mytrain.model.monitor._epochs_seen
        while args["learningRate"] > 0 and (args["maxEpochs"] is None or args["maxEpochs"] > 0):
            mytrain = train(init_train(model=mytrain.model, **args), **args)
            args.update(mytrain.config)
            args["batchSize"] += b if b else 0
            args["learningRate"] += lr if lr else 0
            if args["maxEpochs"]:
                args["maxEpochs"] -= mytrain.model.monitor._epochs_seen

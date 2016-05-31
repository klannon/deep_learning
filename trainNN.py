from __future__ import print_function

import datetime, os, sys, time, json, argparse

from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import SGD

import deep_learning.protobuf as pb
import deep_learning.utils.dataset as ds
from deep_learning.utils import progress, convert_seconds
from deep_learning.utils.configure import set_configurations
from deep_learning.utils.validate import Validator
from math import ceil
from time import clock
import numpy as np
from keras.utils.visualize_util import plot

"""
model.from_json(file)

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
visualization
"""


def build(config=None):

    if config is None:
        config = set_configurations()
    ##
    # Experiment log set-up
    ##

    exp = pb.Experiment()
    exp.start_date_time = str(datetime.datetime.now())
    exp.dataset = pb.Experiment.Dataset.Value(config["dataset"])
    exp.coordinates = config["coords"]
    exp.batch_size = config["batch_size"]
    exp.description = config["save_name"] if config["save_name"] else ''

    ##
    # Construct the network
    ##

    model = Sequential()

    layer = exp.structure.add()
    layer.type = 0
    layer.input_dimension = 15
    layer.output_dimension = config["nodes"]

    model.add(Dense(config["nodes"], input_dim=15))
    model.add(Activation("relu"))

    for l in xrange(config["layers"]):
        layer = exp.structure.add()
        layer.type = 0
        layer.input_dimension = config["nodes"]
        layer.output_dimension = config["nodes"]
        model.add(Dense(config["nodes"]))
        model.add(Activation("relu"))

    layer = exp.structure.add()
    layer.type = 1
    layer.input_dimension = config["nodes"]
    layer.output_dimension = 2
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    ##
    # Generate the optimization method
    ##

    opt = pb.SGD()
    opt.lr = config["learning_rate"]
    exp.sgd.MergeFrom(opt)

    ##
    # Compile the model
    ##

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=opt.lr), metrics=['accuracy'])

    if not config.has_key("terms"):
        config["terms"] = dict(epochs=config["max_epochs"],
                               timeout=config["timeout"],
                               plateau=dict(m=config["slope"],
                                            w=config["width"]))

    return model, exp, config["terms"]



def run(model, exp, terms, save_freq=5):

    exp_dir = ds.get_path_to_dataset(pb.Experiment.Dataset.Name(exp.dataset))

    ##
    # Creates a name based on the host machine name or PID if no custom name is given.
    # Uses hexadecimal representation of time to shorten file name (with x as the decimal point)
    ##

    if exp.description:
        output_file_name = exp.description
    else:
        output_file_name = ("{}_{}".format(str(os.getenv("HOST", os.getpid())).split('.')[0],
                                           'x'.join(map(lambda x: hex(int(x))[2:], str(time.time()).split('.')))))
    save_dir = os.path.join(exp_dir, output_file_name)
    exp_file_name = "{}.exp".format(output_file_name)

    ##
    # Load data from .npz archive created by invoking
    # deep_learning/utils/archive.py
    ##

    x_train, y_train, x_test, y_test = ds.load_dataset(pb.Experiment.Dataset.Name(exp.dataset), exp.coordinates)

    train_length = x_train.shape[0]
    num_batches = int(ceil(train_length / exp.batch_size))

    valid = Validator(exp, terms)

    eTimes = np.array([])
    valid._clock = clock()
    while valid.check():
        t = clock()
        if valid._num_epochs:
            print("Epoch {}/{}".format(valid.epochs+1, valid._num_epochs))
        else:
            print("Epoch {}".format(valid.epochs+1))
        bETA = 0
        bTimes = np.array([])
        for b in xrange(num_batches):
            bt = clock()
            progress(b, num_batches, exp.batch_size, bETA)
            model.train_on_batch(x_train[b*exp.batch_size:b*exp.batch_size+exp.batch_size, :],
                                 y_train[b*exp.batch_size:b*exp.batch_size+exp.batch_size, :])
            bTimes = np.append(bTimes, clock()-bt)
            bETA = np.median(bTimes)*(num_batches-b-1)
        progress(num_batches, num_batches, exp.batch_size, 0, end='\n')
        epoch = exp.results.add()
        epoch.train_loss, epoch.train_accuracy = model.evaluate(x_train, y_train, batch_size=exp.batch_size, verbose=2)
        epoch.test_loss, epoch.test_accuracy = model.evaluate(x_test, y_test, batch_size=exp.batch_size, verbose=2)
        print("\t Train Accuracy: {:.3f}\tTest Accuracy: {:.3f}".format(epoch.train_accuracy, epoch.test_accuracy))
        if len(exp.results) >= terms["plateau"]["w"]:
            print("\t Slope: {:.5f} (test_accuracy / epoch)".format(valid.slope))
        if (len(exp.results) % save_freq) == 0:
            save(model, exp, save_dir, exp_file_name)
            print("\t Saved the model")
        epoch.num_seconds = int(round(clock() - t))
        print("\t Time this epoch: {}s".format(epoch.num_seconds), end='')
        if valid._num_epochs:
            eTimes = np.append(eTimes, epoch.num_seconds)
            print("\tFinal ETA: {}".format(convert_seconds(np.median(eTimes) * (valid._num_epochs - valid.epochs))))
        else:
            print()
        sys.stdout.flush()

    exp.end_date_time = str(datetime.datetime.now())
    exp.total_time = valid.time

    print("\n"+valid.failed)
    print("Total Time: {}".format(convert_seconds(valid.time)))

    save(model, exp, save_dir, exp_file_name)

def save(model, exp, save_dir, exp_file_name):
    ##
    # Save the model configuration, weights, and experiment object
    ##

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    with open(exp_file_name, "wb") as experiment_file:
        experiment_file.write(exp.SerializeToString())
        experiment_file.close()

    config = model.to_json()
    with open("cfg.json", 'w') as fp:
        json.dump(config, fp)

    w = model.get_weights()
    np.save("weights", w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##
    # Set up the command line arguments
    ##

    parser.add_argument("dataset", metavar="Dataset", help="the dataset you wish to use")
    parser.add_argument("coords", metavar="Format", help="the data format you wish to use")
    parser.add_argument("-r", "--learning_rate", help="learning rate",
                        type=float, default=None)
    parser.add_argument("-b", "--batch_size", help="size of each batch "
                                                  + "(subset of training set)", type=int, default=None)
    parser.add_argument("-l", "--layers",
                        help="number of hidden layers in the network",
                        type=int, default=None)
    parser.add_argument("-e", "--max_epochs",
                        help="number of epochs to run for", type=int,
                        default=None)
    parser.add_argument("-n", "--nodes",
                        help="number of nodes per layer", type=int,
                        default=None)
    parser.add_argument("-t", "--timeout",
                        help="how long it should train in minutes",
                        type=float, default=None)
    parser.add_argument("-mf", "--monitor_fraction", help="a two-tuple with "
                                                         + "the  training and testing monitoring percents",
                        default=None, type=tuple)
    parser.add_argument("-sf", "--save_freq", help="how often the model "
                                                       + "should be saved and backed up", default=None, type=int)
    parser.add_argument("-s", "--save_name", help="name for the data directory "
                                                   + "and the .exp file within", default=None)
    parser.add_argument("-w", "--width", help="the number of datapoints to "
                                              + "average the slope over", type=int, default=None)
    parser.add_argument("-m", "--slope", help="the slope that determines a "
                                              + "plateau i.e. when to stop training", type=float, default=None)
    parser.add_argument("--config", help="train based upon a .cfg file", default=None)
    parser.add_argument("--defaults", help="run on defaults", action="store_true")
    args = vars(parser.parse_args())

    defaults = dict(learning_rate=0.01,
                    batch_size=64,
                    layers=5,
                    max_epochs=None,
                    nodes=50,
                    timeout=None,
                    monitor_fraction=0,
                    save_freq=5,
                    save_name=None,
                    width=2,
                    slope=None,
                    config=None,
                    defaults=False)

    if bool(filter(None, args.values())):
        for k,v in args.items():
            if not v:
                args[k] = defaults[k]
        model, exp, terms = build(args)
    else:
        model, exp, terms = build()

    run(model, exp, terms, args["save_freq"])


"""
After training finishes, this is the dump structure:

--> <Custom-name>
-----> cfg.json
-----> structure.png (optional)
-----> <Custom-name>.exp
-----> weights.npy

MAKE THE DATA DIR A NECESSARY ARGUMENT WHEN INSTALLING PACKAGE (like a config file)

NOTE PREPROCESSING STEPS

IMPLEMENT CONTINUE TRAINING

Create UID (Unique Identifier Code) for naming log + experiment files, much like permissions.
Utilize hexadecimal labeling with

RUN UNTIL IT REACHES A CERTAIN ACCURACY?

MONITORING FRACTION?

DIFFERENT NODES PER LAYER
"""
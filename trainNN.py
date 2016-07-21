from __future__ import print_function, division

import datetime, os, sys, time, argparse
# Need Keras >= 1.0.5
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from theano import function, printing
import json
#theano.config.exception_verbosity="high";
theano.config.traceback.limit = 20
import keras.backend as K
from keras.layers import Dense, Dropout, Input, Merge, merge, Lambda
from keras.engine.topology import Layer
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2

import deep_learning.protobuf as pb
import deep_learning.utils.dataset as ds
from deep_learning.utils import progress, convert_seconds
from deep_learning.utils.configure import set_configurations
from deep_learning.utils.validate import Validator
import deep_learning.utils.transformations as tr
import deep_learning.utils.stats as st
import deep_learning.utils.graphs as gr
import deep_learning.utils.archive as ar
import matplotlib.pyplot as plt
from math import ceil
from time import clock
import numpy as np
from deep_learning.utils import E
#from keras.utils.visualize_util import plot

"""
model.from_json(file)

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
visualization
"""

def load_model(exp_name):
    data, name = exp_name.split('/')
    exp_dir = ds.get_path_to_dataset(data) + os.sep + name +os.sep
    with open(exp_dir+"cfg.json") as json:
        model = model_from_json(json.read())
    model.set_weights(np.load(exp_dir+"weights.npy"))
    return model


class Permute(Layer):
    def __init__(self, output_dim, permutations, batch_size, **kwargs):
        self.output_dim = output_dim
        self.permutations = permutations
        self.num_p = len(permutations)
        self.batch_size = batch_size
        #self.labels = T.zeros((1, self.num_p))

        """def _get_labels(x, labels, num_p):
            skip = x.shape[0] // num_p
            srng = RandomStreams()
            self.indices = srng.permutation(n=num_p, size=(1,))[0]
            T.set_subtensor(labels[:], T.zeros((1, num_p)))
            T.set_subtensor(labels[0, 0], 1)
            T.set_subtensor(labels[0, :], labels[0, self.indices])
            identity = T.zeros((self.num_p, self.num_p * skip))
            theano.scan(lambda i, identity, skip: T.set_subtensor(identity[i, i * skip:(i + 1) * skip], [1] * skip),
                        n_steps=self.num_p,
                        sequences=[T.arange(self.num_p)],
                        non_sequences=[identity, skip])
            return T.dot(labels, identity)"""
        #x = T.dmatrix()
        #labels = T.imatrix()
        #self.get_labels = function([self, x, labels], _get_labels)

        super(Permute, self).__init__(**kwargs)

    def build(self, input_shape):
        self.transforms = [E(p) for p in self.permutations]
        self.T = K.variable(self.transforms)
        self.trainable = False

    def call(self, x, mask=None):
        srng = RandomStreams()
        self.indices = srng.permutation(n=self.num_p, size=(1,))[0]

        skip = x.shape[0] // self.num_p

        rval = K.concatenate([K.dot(x, self.T[i].transpose()) for i in xrange(self.num_p)], axis=0)

        #indexes = printing.Print("Indexes")(indexes)
        temp = rval.copy()
        theano.scan(lambda b, x, skip: T.set_subtensor(rval[b::skip], temp[b::skip][self.indices,]),
                             n_steps=skip,
                             sequences=[T.arange(skip)],
                             non_sequences=[x, skip])
        #return T.flatten(out, outdim=2)

        return rval

    def get_output_shape_for(self, input_shape):
        return (self.num_p, self.output_dim)

    def compute_mask(self, input, input_mask=None):
        return None

### OLD MODEL
def build_default(config, exp):
    ##
    # Construct the network
    ##

    model = Sequential()

    layer = exp.structure.add()
    layer.type = 0
    layer.input_dimension = 44
    layer.output_dimension = config["nodes"]

    model.add(Dense(config["nodes"], input_dim=44, activation="relu", W_regularizer=l1(0.001)))
    #model.add(Dropout(0.2))

    for l in xrange(config["layers"]-1):
        layer = exp.structure.add()
        layer.type = 0
        layer.input_dimension = config["nodes"]
        layer.output_dimension = config["nodes"]
        model.add(Dense(config["nodes"], activation="relu", W_regularizer=l1(0.001)))
    #    model.add(Dropout(0.2))

    layer = exp.structure.add()
    layer.type = 1
    layer.input_dimension = config["nodes"]
    layer.output_dimension = 2
    model.add(Dense(output_dim=2, activation="softmax"))

    return model

### SORTING NET
#inputs = [Input(shape=(44,), name="Event Input {}".format(i)) for i in xrange(len(perms))]

"""def clean_sorting_outputs(x):
    skip = x.shape[0] // len(perms)
    out, _ = theano.scan(lambda b, x, skip: K.reshape(x[b*skip:(b+1)*skip], (1, x.shape[1] * len(perms))),
                         n_steps=skip,
                         sequences=[T.arange(skip)],
                         non_sequences=[x, skip])
    return T.flatten(out, outdim=2)

o = Lambda(clean_sorting_outputs,
           output_shape=lambda s: (exp.batch_size, 20 * len(perms)),
           name="Filter")(o)"""

def build_sorting_net(config, exp):
    inputs = Input(shape=(44,), name="Event Input")
    extended_net = Sequential(name="ReLu Network")
    extended_net.add(Dense(50, input_dim=44, activation="relu", W_regularizer=l1(0.001)))
    extended_net.add(Dense(50, activation="relu", W_regularizer=l1(0.001)))
    extended_net.add(Dense(50, activation="relu", W_regularizer=l1(0.001)))
    extended_net.add(Dense(50, activation="relu", W_regularizer=l1(0.001)))
    extended_net.add(Dense(50, activation="relu", W_regularizer=l1(0.001)))
    soft = Dense(2, activation="softmax", name="Classifier (Softmax)")
    o = extended_net(inputs)
    o = soft(o)

    return Model(input=inputs, output=o)

def build_supernet(config, exp):

    ### SUPER NET
    perms = list(ar.gen_permutations(2,7,2))

    def clean_outputs(x):
        skip = x.shape[0] // len(perms)
        out, _ = theano.scan(lambda b, x, skip: K.reshape(x[b::skip], (1, x.shape[1] * len(perms))),
                             n_steps=skip,
                             sequences=[T.arange(skip)],
                             non_sequences=[x, skip])
        return T.flatten(out, outdim=2)

    inputs = Input(shape=(44,), name="Event Input")

    sorted_model = Sequential(name="Sorted Model")
    for ix, layer in enumerate(load_model("ttHLep/S_1to1_small").layers[:-1]):
        layer.trainable = False
        sorted_model.add(layer)
        sorted_model.layers[ix].set_weights(layer.get_weights())

    o = sorted_model(Permute(44, perms, exp.batch_size, name="Permutator")(inputs))

    o = Lambda(clean_outputs,
               output_shape=lambda s: (s[0] // len(perms), 20 * len(perms)),
               name="Filter")(o)

    ### SUPER-NET CLASSIFIER EXTENSION
    extended_net = Sequential(name="ReLu Network")
    extended_net.add(Dense(config["nodes"], input_dim=16800, activation="relu"))
    for l in xrange(config["layers"] - 1):
        layer = exp.structure.add()
        layer.type = 0
        layer.input_dimension = config["nodes"]
        layer.output_dimension = config["nodes"]
        extended_net.add(Dense(config["nodes"], activation="relu"))
    soft = Dense(2, activation="softmax", name="Classifier (Softmax)")

    o = extended_net(o)
    o = soft(o)

    return Model(input=inputs, output=o)

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
    ##
    # Creates a name based on the host machine name or PID if no custom name is given.
    # Uses hexadecimal representation of time to shorten file name (with x as the decimal point)
    ##
    if config["save_name"]:
        exp.description = config["save_name"]
    else:
        exp.description = "{}_{}".format(str(os.getenv("HOST",
                                                       os.getpid())).split('.')[0],
                                         'x'.join(map(lambda x: hex(int(x))[2:], str(time.time()).split('.'))))


    model = build_supernet(config, exp)

    ##
    # Generate the optimization method
    ##

    opt = pb.Adam()
    opt.lr = config["learning_rate"]
    exp.adam.MergeFrom(opt)

    ##
    # Compile the model
    ##

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=opt.lr), metrics=['accuracy'])

    if not config.has_key("terms"):
        config["terms"] = dict(epochs=config["max_epochs"],
                               timeout=config["timeout"],
                               plateau=dict(y=config["rise"],
                                            x=config["run"]))

    return model, exp, config["terms"]



def run(model, exp, terms, save_freq=5, data=None):

    exp_dir = ds.get_path_to_dataset(pb.Experiment.Dataset.Name(exp.dataset))
    save_dir = os.path.join(exp_dir, exp.description)

    ##
    # Load data from .npz archive created by invoking
    # deep_learning/utils/archive.py
    ##

    if data:
        x_train, y_train, x_test, y_test = data
        x_train, x_test = tr.transform(x_train, x_test)
    else:
        h_file, (x_train, y_train, x_test, y_test) = ds.load_dataset(pb.Experiment.Dataset.Name(exp.dataset), exp.coordinates+'/transformed')
        data = x_train, y_train, x_test, y_test

    exp_file_name = exp.description + '.exp'

    # Start training

    train_length = x_train.shape[0]
    num_batches = int(ceil(train_length / exp.batch_size))

    valid = Validator(exp, terms)

    eTimes = np.array([])
    valid._clock = clock()
    model.summary()
    while valid.check():
        t = clock()
        if valid._num_epochs:
            print("Epoch {}/{}".format(valid.epochs+1, valid._num_epochs))
        else:
            print("Epoch {}".format(valid.epochs+1))
        bETA = 0
        bTimes = np.array([])
        #print("\t Training: ")
        for b in xrange(num_batches):
            bt = clock()
            # Update progress bar
            progress(b, num_batches, exp.batch_size, bETA)
            # Train on a batch
            x_batch = x_train[b*exp.batch_size:b*exp.batch_size+exp.batch_size, :]
            y_batch = y_train[b*exp.batch_size:b*exp.batch_size+exp.batch_size, :]
            model.train_on_batch(x_batch, y_batch)
            bTimes = np.append(bTimes, clock()-bt)
            bETA = np.median(bTimes)*(num_batches-b-1)
        # Finish progress bar
        progress(num_batches, num_batches, exp.batch_size, 0, end='\n', time=clock()-t)
        # Calculate stats and add the epoch results to the experiment object
        epoch = exp.results.add()
        timer = clock()
        print("Evaluating Train")
        epoch.train_loss, epoch.train_accuracy = model.evaluate_generator(((x_train[i*exp.batch_size:(i+1)*exp.batch_size],
                                                                           y_train[i*exp.batch_size:(i+1)*exp.batch_size]) for i in xrange(num_batches)),
                                                                          num_batches, max_q_size=min((num_batches//2, 10)))
        print("Finished {:.2f}".format(clock()-timer))
        timer = clock()
        print("Evaluating Test")
        epoch.test_loss, epoch.test_accuracy = model.evaluate_generator(((x_test[i*exp.batch_size:(i+1)*exp.batch_size],
                                                                           y_test[i*exp.batch_size:(i+1)*exp.batch_size]) for i in xrange(int(ceil(x_test.shape[0]/exp.batch_size)))),
                                                                        int(ceil(x_test.shape[0] / exp.batch_size)), max_q_size=min((int(ceil(x_test.shape[0] / exp.batch_size))//2, 10)))
        print("Finished {:.2f}".format(clock() - timer))
        timer = clock()
        print("Calculating Sig")
        epoch.s_b = st.significance(model, data)
        print("Finished {:.2f}".format(clock() - timer))
        timer = clock()
        #print("Calculating AUC {:.2f}".format(clock()))
        #epoch.auc = st.AUC(model, data, experiment_epoch=epoch)
        #print("Finished {:.2f}".format(clock() - timer))
        timer = clock()
        for r in st.num_of_each_cell(model, data):
            epoch.matrix.add().columns.extend(r)
        print("Making CFM")
        matrix = st.confusion_matrix(model, data, offset='\t ')
        print("Finished {:.2f}".format(clock() - timer))
        epoch.num_seconds = clock() - t
        timer=clock()
        print("Getting output")
        output = st.get_output_distro(model, data)
        epoch.output.background.extend(output["background"])
        epoch.output.signal.extend(output["signal"])
        print("Finished {:.2f}".format(clock() - timer))
        # Print statistics
        print("\t Train Accuracy: {:.3f}\tTest Accuracy: {:.3f}".format(epoch.train_accuracy, epoch.test_accuracy))
        if valid.update_w():
            print("\t Slope: {:.5f} (test_accuracy / second)".format(valid.slope))
        print("\t Time this epoch: {:.2f}s".format(epoch.num_seconds), end='')
        if valid._num_epochs:
            eTimes = np.append(eTimes, epoch.num_seconds)
            print("\tFinal ETA: {}".format(convert_seconds(np.median(eTimes) * (valid._num_epochs - valid.epochs))))
        else:
            print()
        print("\t Significance (S/sqrt(B)): {:.2f}".format(epoch.s_b))
        print("\t Area Under the Curve (efficiency): {:.3f}".format(epoch.auc))
        print(matrix)

        if (len(exp.results) % save_freq) == 0:
            save(model, exp, save_dir, exp_file_name)
            print("\t ", end='')
        sys.stdout.flush()

    exp.end_date_time = str(datetime.datetime.now())
    exp.total_time = valid.time

    print("\n"+valid.failed)
    print("Total Time: {}".format(convert_seconds(valid.time)))

    save(model, exp, save_dir, exp_file_name, graph=True)
    print("\t ", end='')
    h_file.close()

def save(model, exp, save_dir, exp_file_name, graph=False):
    ##
    # Save the model configuration, weights, and experiment object
    ##

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    try:
        with open(exp_file_name, "wb") as experiment_file:
            experiment_file.write(exp.SerializeToString())
    except Exception as e:
        print("Failed to save experiment object: {}".format(e))

    config = model.get_config()
    try:
        with open("cfg.json", 'w') as fp:
            fp.write(model.to_json())
    except Exception as e:
        print("Failed to save json configuration: {}\n{}".format(e, config))

    try:
        w = model.get_weights()
        np.save("weights", w)
    except Exception as e:
        print("Failed to save model weights (numpy): {}".format(e))

    """try:
        model.save_weights("weights.h5", overwrite=True)
    except Exception as e:
        print("Failed to save model weights (hdf5): {}".format(e))"""

    if graph:
        model1 = load_model("ttHLep/U_1to1")
        h_file, data = ds.load_dataset("ttHLep", "Unsorted_Large/transformed")
        gr.output_distro(model1, data, subplot=221, labels=["Default Bkg", "Default Sig"])
        gr.output_distro(model, data, subplot=222, labels=["Supernet Bkg", "Supernet Sig"])
        gr.overlay_distro(model1, model, data, category="background", labels=["Default", "Supernet"], subplot=223)
        gr.overlay_distro(model1, model, data, labels=["Default", "Supernet"], subplot=224)
        """gr.s_b(exp)
        gr.auc(exp)
        gr.correct(exp)
        gr.accuracy(exp)"""
        plt.tight_layout()
        plt.savefig("{}{}{}.png".format(save_dir, os.sep, exp.description), format="png")
        plt.clf()

    print("Saved the model data{}".format(" and graph" if graph else '') + " to {}\n".format(save_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##
    # Set up the command line arguments
    ##

    parser.add_argument("dataset", metavar="Dataset/Format", help="The dataset and format you want separated by a slash")
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
    parser.add_argument("-m", "--monitor_fraction", help="a two-tuple with "
                                                         + "the  training and testing monitoring percents",
                        default=None, type=tuple)
    parser.add_argument("-f", "--save_freq", help="how often the model "
                                                       + "should be saved and backed up", default=None, type=int)
    parser.add_argument("-s", "--save_name", help="name for the data directory "
                                                   + "and the .exp file within", default=None)
    parser.add_argument("-x", "--run", help="the interval of time to average the slope over",
                        type=float, default=None)
    parser.add_argument("-y", "--rise", help="the percentile increase in accuracy that you expect over this interval",
                        type=lambda y: float(y) if y.isdigit() else float(y[:-1])/100, default=None)
    parser.add_argument("--config", help="train based upon a .cfg file", default=None)
    parser.add_argument("--defaults", help="run on defaults", action="store_true")
    args = vars(parser.parse_args())
    _temp = args["dataset"].split('/')
    args["dataset"] = _temp[0]
    args["coords"] = '/'.join(_temp[1:])

    defaults = dict(learning_rate=0.001,
                    batch_size=64,
                    layers=2,
                    max_epochs=None,
                    nodes=20,
                    timeout=None,
                    monitor_fraction=0,
                    save_freq=5,
                    save_name=None,
                    run=None,
                    rise=None,
                    config=None,
                    split=False,
                    defaults=False)

    for k,v in args.items():
        if not v:
            args[k] = defaults[k]

    model, exp, terms = build(args)
    run(model, exp, terms, args["save_freq"])


"""

Look at new data on full and also on data amount equivalent to old

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

COULD DO A LITTLE MORE WORK ON PREPROCESSING

SHUFFLE THE DATA WHEN IT'S LOADED?

Fix shapes: scan over lr and batch
then try another shape and repeat

(I+1)H + H(H+1)(L-1) + O(H+1)

Shapes:
    50x5
    35x10
    28x15
    24x20

regularizations

save models

shuffling and normalizing each batch

Try regularizations

    Try to reduce information given to network (e.g. take out a few jets, or leave only the "top" few, or both leptons,
                                                could look through the file and divide them into n-non-null jets and
                                                see if it was just counting)
    Plot BDT vs Network output curve to see if they think similarly

PREPROCESSING STEPS:
---> Update .json file
---> Organize the files into one file or train + test files
---> Call utils.archive.create_archive with the appropriate arguments
---> Should be good to go!

"""
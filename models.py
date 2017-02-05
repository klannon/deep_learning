from __future__ import print_function, division

import datetime, os, sys, time, argparse
# Need Keras >= 1.0.5
import theano.tensor as T
import theano
theano.config.traceback.limit = 20 # Sets the error traceback length
from theano.tensor.shared_randomstreams import RandomStreams
from keras.layers import Dense, Dropout, Input, Merge, merge, Lambda
from keras.engine.topology import Layer
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras.backend as K

import deep_learning.utils.dataset as ds
from deep_learning.utils import progress, convert_seconds
from deep_learning.utils.validate import Validator
import deep_learning.utils.archive as ar
import numpy as np
from deep_learning.utils import E

def load_model(exp_name):
    """
    Loads a model from an experiment and returns the model.

    Parameters
    ----------
    exp_name <string> : a forward-slash separated string for the experiment name i.e. <dataset>/<experiment>

    Returns
    -------
    model <Keras.engine.topology.Model> : a Keras Model instance as defined in the cfg.json of
                                         the experiment that is being loaded.

    """
    data, name = exp_name.split('/')
    exp_dir = ds.get_path_to_dataset(data) + os.sep + name +os.sep
    with open(exp_dir+"cfg.json") as json:
        model = model_from_json(json.read())
    model.set_weights(np.load(exp_dir+"weights.npy"))
    return model

class Supernet(Model):
    """
    This is Supernet. Supernet uses a permutation generator linked to a pre-trained model on correctly assigned
    data and then a flattening layer and finally a regular feed forward network.
    """
    def __init__(self, config, exp):
        """

        Parameters
        ----------
        config : A configuration diction is needed. These can be generated from ...
        exp : A protobuf experiment object as generated in ...
        """
        inputs = [Input(shape=(44,), name="Event Permutation {}".format(i)) for i in xrange(840)]

        sorted_model = Sequential(name="Sorted Model")
        for ix, layer in enumerate(load_model("ttHLep/CAoptimized").layers[:-1]):
            layer.trainable = False
            sorted_model.add(layer)
            sorted_model.layers[ix].set_weights(layer.get_weights())

        o = merge([sorted_model(x) for x in inputs], mode="concat")

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

        self.experiment = exp
        self.experiment_config = config
        self.metrics_function = None
        super(Supernet, self).__init__(inputs, o, "Supernet")

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        metrics = ["accuracy"] if not metrics else metrics
        loss = "categorical_crossentropy" if not loss else loss
        super(Supernet, self).compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights,
                                      sample_weight_mode=sample_weight_mode, **kwargs)
        self._make_metrics()
        return None

    def _make_metrics(self):
        """ Assumes metrics=['cross_entropy', 'accuracy'] """
        inputs = self.inputs+self.targets+self.sample_weights
        outputs = [self.total_loss] + self.metrics
        self.metrics_function = K.function(inputs, outputs, **self._function_kwargs)

    def get_metrics(self, x_):
        """ Returns the metrics calculated in batches """
        if not hasattr(self, 'metrics_function'):
            raise Exception('You must compile your model before using it.')

        batch_size = self.experiment_config["batch_size"]

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

def build_default(config, exp):
    """
    This will build a basic feed forward neural network with equally sized hidden layers, rectified linear activation
    functions, and then a sigmoid output.

    Parameters
    ----------
    config <dict> : A dictionary of configurable parameters as defined in the --help switch.
    exp <deep_learning.protobuf.Experiment> : A custom Protobuf object to store the experiment data within.

    Returns
    -------
    model <Keras.models.Sequential> : A fully constructed Keras neural network (Feed Forward) that is ready to train.
    """

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

def build_supernet(config, exp):
    """
    This will build a special neural network that we have dubbed "Supernet". Supernet consists of a permutation layer
    that will produce all feasible permutations of the input data. Then we load an optimized model trained on data that
    has been properly sorted and apply that model to each of the permuted inputs. Finally, we train a new neural
    network on the outputs of the optimized (and frozen) network.

    Parameters
    ----------
    config <dict> : A dictionary of configurable parameters as defined in the --help switch.
    exp <deep_learning.protobuf.Experiment> : A custom Protobuf object to store the experiment data within.

    Returns
    -------
    model <Keras.models.Sequential> : A fully constructed Keras neural network (Feed Forward) that is ready to train.
    """

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
    for ix, layer in enumerate(load_model("ttHLep/CAoptimized").layers[:-1]):
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

networks = {"default": build_default,
            "supernet": build_supernet}
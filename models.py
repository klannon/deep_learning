from __future__ import print_function, division

import datetime, os, sys, time, argparse
# Need Keras >= 1.0.5
import theano.tensor as T
import theano
import json
theano.config.traceback.limit = 20
from keras.layers import Dense, Dropout, Input, Merge, merge, Flatten
from keras.engine.topology import Layer
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras.backend as K

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

def load_model(exp_name):
    data, name = exp_name.split('/')
    exp_dir = ds.get_path_to_dataset(data) + os.sep + name +os.sep
    with open(exp_dir+"cfg.json") as json:
        model = model_from_json(json.read())
    model.set_weights(np.load(exp_dir+"weights.npy"))
    return model

class Supernet(Model):
    def __init__(self, config, exp):
        inputs = [Input(shape=(44,), name="Event Permutation {}".format(i)) for i in xrange(840)]

        sorted_model = Sequential(name="Sorted Model")
        for ix, layer in enumerate(load_model("ttHLep/S_1to1_small").layers[:-1]):
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


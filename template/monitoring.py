from __future__ import print_function
from random import sample
from physics import PHYSICS
from functools import wraps
from pylearn2.train import Train
from types import MethodType

__author__ = 'Matt'

def make_data_slim(datasets, data_percent=0.02):

    rval = []
    for data in datasets:
        X = data.X
        Y = data.y
        total = X.shape[0]

        indices = sample(xrange(total), int(data_percent*total))
        rval.append(PHYSICS(X[indices], Y[indices]))

    return rval


def monitor_refiner(monitor, channels='basic'):
    """
    This is an in-place editor of your pylearn2 model's monitor.
    You may choose the names of which channels you would like
    to monitor and all of the other channels will be forgotten.
    **Important: You should call this method before you start
    your training loop for maximum resource gains.

    Parameters
    ----------
    monitor : pylearn2.monitor.Monitor
        The pylearn2 model object that needs a refined monitor.
    channels : list
        The channels that you would like to KEEP in the monitor.
        All other channels will be removed.
        ** Note: This does not have to be the full name, it can
        be any string indicative of the channel that you want.
    Returns
    -------
    None
    """

    if channels == 'basic':
        channels = ('learning_rate',
                    'objective',
                    'y_nll',
                    'y_misclass',
                    'total_seconds_last_epoch',
                    'training_seconds_this_epoch')

    for key in monitor.channels:
        for chan in channels:
            if chan in key: break
        else:
            del monitor.channels[key]
    return None

def TrainVeil(train_instance):
    train_instance._old_setup = train_instance.setup
    train_instance.setup = MethodType(fake_setup, train_instance)

@wraps(Train.setup)
def fake_setup(self):
    self._old_setup()
    monitor_refiner(self.model.monitor)
    return None

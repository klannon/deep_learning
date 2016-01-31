from __future__ import print_function
from random import sample
from physics import PHYSICS
from functools import wraps
from pylearn2.train import Train
from types import MethodType

__author__ = 'Matt'

def make_data_slim(datasets, data_percent=0.02):
    """
    This function takes one or more PHYSICS instances and takes a random
    sample of each, based on the percent that is provided, and returns
    new PHYSICS instances of the new, smaller dataset.

    Parameters
    ----------
    datasets : list of PHYSICS instances
        This needs to be a list of data that you want to slim down.
        The data will be returned in the same order that they are submitted.
    data_percent : float
        This is what percent of the data you would like to have returned.

    Returns
    -------
    rval : list of PHYSICS instances
        This is a list of new PHYSICS instances that have a slimmed amount of data.
    """
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
                    'recall',
                    'precision',
                    'total_seconds_last_epoch',
                    'training_seconds_this_epoch')

    for key in monitor.channels:
        for chan in channels:
            if chan in key: break
        else:
            del monitor.channels[key]
    return None

def TrainVeil(train_instance):
    """
    This decorator (if you could decorate an instance) redefines the <setup>
    method for this instance. The instance now thinks that <fake_setup> is
    actually the bound method <setup>. Therefore, when <setup> is called
    upon the first call to monitoring, <fake_setup> will be run in its
    place (look at <fake_setup>'s documentation if you wish to know the
    effects of implementing it). This "veiling" of the <setup> method is
    achieved in-place, but the instance is returned as well for those who
    need it.

    Parameters
    ----------
    train_instance : pylearn2.train.Train
        This needs to be an instance of the <Train> class that you wish
        to have its <setup> method altered.

    Returns
    -------
    train_instance : pylearn2.train.Train
        This is the <Train> instance now properly veiled.
    """
    train_instance._old_setup = train_instance.setup
    train_instance.setup = MethodType(fake_setup, train_instance)
    return train_instance

@wraps(Train.setup)
def fake_setup(self):
    """
    This function (or method) is particularly used as a stand-in
    for the <Train.setup> method in order to allow monitoring refinement.
    First the original <setup> method is called in order to build the
    necessary channels. Then the <monitor_refiner> is called in order
    to reduce the channels that end up being compiled and used for
    monitoring. This function also wraps <Train.setup> so that it still
    appears as the <setup> function for all other purposes.

    Parameters
    ----------
    self : pylearn2.train.Train
        This is the instance of <Train> that called the method.

    Returns
    -------
    None
    """
    self._old_setup()
    monitor_refiner(self.model.monitor)
    return None
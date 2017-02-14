from __future__ import division
import numpy as np
from math import sqrt
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr
import deep_learning.utils as ut

#import matplotlib.pyplot as plt
from numpy import trapz
from math import ceil

LUMINOSITY = 30  # fb^-1
TTHIGGS_X_SECTION = 212  # fb
TTHIGGS_GENERATED = 3991615
TTBAR_X_SECTION = 365399  # fb
TTBAR_GENERATED = 353216236
BACKGROUND_SAMPLE = 25811
SIGNAL_SAMPLE = 42374
SAMPLES = (BACKGROUND_SAMPLE, SIGNAL_SAMPLE)

CFM = """
{0}Sum over rows: Actual Class
{0}Sum over columns: Predicted Class
{0}Confusion Matrix:
{0}{1}
"""

class MATRIX(object):
    def __init__(self, matrix, offset=''):
        self.matrix = matrix
        self.offset = offset

    def __str__(self):
        return CFM.format(self.offset, ('\n'+self.offset).join([str(row) for row in self.matrix]))


def count_filter(model, criteria, (x_test, y_test), batch_size=256, index=False, **kwargs):
    """

    Parameters
    ----------
    model
    criteria
    batch_size
    kwargs

    Returns
    -------
    <tuple> : A tuple with the number of events that passed the filter for each class.

    """
    rval = []
    for i in xrange(int(ceil(y_test.shape[0]/batch_size))):
        predictions = model.predict([x_test[i*batch_size:(i+1)*batch_size]], **kwargs)
        if type(index) is np.ndarray:
            if index[i*batch_size:(i+1)*batch_size].any():
                predictions = predictions[index[i*batch_size:(i+1)*batch_size]]
                y_temp = y_test[i*batch_size:(i+1)*batch_size][index[i*batch_size:(i+1)*batch_size]]
            else:
                continue
        else:
            y_temp = y_test[i*batch_size:(i+1)*batch_size]
        bArray = criteria(predictions, y_temp)
        rval.append(ut.sum_cols(bArray, batch_size))
    return tuple([c.sum() for c in np.array(rval).T])

def num_of_each_cell(model, data):
    x_train, y_train, x_test, y_test = data
    matrix = []
    for i in xrange(y_test.shape[1]):
        index = (y_test[:,i] > 0).flatten()
        num_predicted = count_filter(model, lambda p, y: np.vstack(map(lambda x: x==max(x), p)), (x_test, y_test),
                                     index=index)
        matrix.append(num_predicted)
    return np.array(matrix)

# Need to generalize for more categories
def efficiencies(model, data, over_rows=True, **kwargs):
    matrix = num_of_each_cell(model, data)
    rval = np.array(map(lambda row: [x / sum(row) for x in row], matrix if over_rows else matrix.T))
    return rval if over_rows else rval.T

# Need to generalize for more categories
def significance(model, data, override=None):
    if override:
        z = zip(override, SAMPLES,
                [LUMINOSITY * TTBAR_X_SECTION / TTBAR_GENERATED, LUMINOSITY * TTHIGGS_X_SECTION / TTHIGGS_GENERATED])
        predictions = [p * t * c for p, t, c in z]  # Percent, total, Constant
    else:
        efficiency = efficiencies(model, data)[:,1]  # bs, ss
        z = zip(efficiency, SAMPLES,
                [LUMINOSITY * TTBAR_X_SECTION / TTBAR_GENERATED, LUMINOSITY * TTHIGGS_X_SECTION / TTHIGGS_GENERATED])
        predictions = [p * t * c for p, t, c in z]  # Percent, total, Constant
    return predictions[1] / sqrt(predictions[0])

# ""
def AUC(model, data, datapoints=20, save='', experiment_epoch=None):
    datapoints += 1
    e_b = np.zeros(datapoints)
    e_s = np.zeros(datapoints)
    for i in xrange(datapoints):
        cutoff = 1-i*(1/datapoints)
        e_b[i], e_s[i] = efficiencies(model, data)[:,1]
        if experiment_epoch:
            point = experiment_epoch.curve.add()
            point.signal = e_s[i]
            point.background = e_b[i]
            point.cutoff = cutoff
    """if save:
        plt.plot(e_b, e_s)
        plt.title("Efficiency Curve")
        plt.ylabel("Signal Efficiency")
        plt.xlabel("Background Inefficiency")
        plt.savefig(save, format="png")"""
    return trapz(e_s,e_b)

# ""
def confusion_matrix(model, data, offset='', **kwargs):
    #eff = efficiencies(model, data, **kwargs)
    #return MATRIX.format(offset, *(eff*100).flatten())
    return MATRIX(num_of_each_cell(model, data), offset=offset)

def get_output_distro(model, data, batch_size=64, nbins=50):
    x_train, y_train, x_test, y_test = data
    output = np.zeros(y_test.shape)
    for i in xrange(int(ceil(x_test.shape[0] / batch_size))):
        output[i * batch_size:(i + 1) * batch_size] = model.predict(x_test[i * batch_size:(i + 1) * batch_size])
    background = output[(y_test[:] == 1)[:, 0]][:, 1]
    signal = output[(y_test[:] == 1)[:, 1]][:, 1]

    bins = [i / nbins for i in xrange(nbins + 1)]
    frequencies = {"background": [], "signal": []}

    for i in xrange(nbins):
        frequencies["background"].append(((bins[i] <= background) & (background <= bins[i + 1])).sum())
        frequencies["signal"].append(((bins[i] <= signal) & (signal <= bins[i + 1])).sum())

    return frequencies
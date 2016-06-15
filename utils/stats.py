from __future__ import division
import numpy as np
from math import sqrt
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr

import matplotlib.pyplot as plt
from numpy import trapz

LUMINOSITY = 30  # fb^-1
TTHIGGS_X_SECTION = 212  # fb
TTHIGGS_GENERATED = 3991615
TTBAR_X_SECTION = 365399  # fb
TTBAR_GENERATED = 353216236

MATRIX = """
{0}      Predicted
{0}   ==================
{0}R  || {3:.1f} || {1:.1f} || TTBar
{0}e  ==================
{0}a  || {4:.1f} || {2:.1f} || TTHiggs
{0}l  ==================
{0}     TTBar   TTHiggs
"""

def count_filter(model, criteria, (x_test, y_test), **kwargs):
    predictions = model.predict(x_test, **kwargs)
    bArray = criteria(predictions, y_test)
    return tuple([c.sum() for c in bArray.T])

# Need to generalize for more categories
def efficiencies(model, data, cutoff=0.5):
    x_train, y_train, x_test, y_test = data
    num_correct = count_filter(model, lambda p, y: (y > 0) == (p - y > cutoff-1), (x_test, y_test))
    num_predicted = count_filter(model, lambda p, y: p > cutoff, (x_test, y_test))
    s = num_correct[1]
    b = num_predicted[1] - s
    test_total = [c.sum() for c in y_test.T]
    return [x / t for x, t in zip([b, s], test_total)]

# Need to generalize for more categories
def significance(model, data):
    x_train, y_train, x_test, y_test = data
    efficiency = efficiencies(model, data)
    total = [c.sum() + d.sum() for c, d in zip(y_test.T, y_train.T)]
    z = zip(efficiency, total,
            [LUMINOSITY * TTBAR_X_SECTION / TTBAR_GENERATED, LUMINOSITY * TTHIGGS_X_SECTION / TTHIGGS_GENERATED])
    predictions = [p * t * c for p, t, c in z]  # Percent, total, Constant
    return predictions[1] / sqrt(predictions[0])

# ""
def AUC(model, data, datapoints=20, save=False):
    datapoints += 1
    e_b = np.zeros(datapoints)
    e_s = np.zeros(datapoints)
    for i in xrange(datapoints):
        e_b[i], e_s[i] = efficiencies(model, data, 1-i*(1/datapoints))
    plt.plot(e_b, e_s)
    if save:
        plt.title("Efficiency Curve")
        plt.ylabel("Signal Efficiency")
        plt.xlabel("Background Efficiency")
        plt.xlabel("Background Efficiency")
        plt.savefig("AUC.png", format="png")
    return trapz(e_s,e_b)

# ""
def confusion_matrix(model, data, offset=''):
    eff = efficiencies(model, data)
    return MATRIX.format(offset, *[n*100 for n in (eff+[1-x for x in eff])])


if __name__ == "__main__":
    from deep_learning.trainNN import load_model
    model = load_model("ttHLep/U_Optimal")
    x_train, y_train, x_test, y_test = ds.load_dataset("ttHLep", "Unsorted")
    x_train, x_test = tr.transform(x_train, x_test)
    data = (x_train, y_train, x_test, y_test)
    print significance(model, data)
    print AUC(model, data, save=True)
    print confusion_matrix(model, data)
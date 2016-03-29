# ---------------------------------------------------------------------
# Implements An Extension for Model Profiling
#
# Author: Matthew Drnevich
# ---------------------------------------------------------------------

from pylearn2.termination_criteria import TerminationCriterion
from time import time
from pylearn2.monitor import read_channel

class Timeout(TerminationCriterion):
    """
    Keep learning until a given timeout is reached.

    Parameters
    ----------
    criteria : iterable
        An integer in seconds determining how long the model should train for
        in clock time.
    """

    def __init__(self, timeout):
        self.timeout = timeout
        self.start = time()

    def continue_learning(self, model):
        if time() - self.start > self.timeout:
            return False
        else:
            return True

class AdjustmentWatcher(TerminationCriterion):
    """
    Returns True until DynamicAdjustment says to stop training.
    """

    def __init__(self, width, slope):
        self.w = width
        self.m = slope
        self.accuracy = [n/width for n in xrange(0, width)]

    def continue_learning(self, model):
        acc = read_channel(model, "test_y_misclass")
        self.accuracy = self.accuracy[1:]+[acc]
        slope = sum(self.accuracy)/self.w
        if slope <= self.m:
            return False
        else:
            return True

class TerminatorManager(object):

    def __init__(self, *terminators):
        self.terminators = terminators

    def continue_learning(self, model):
        return all([t.continue_learning(model) for t in self.terminators])
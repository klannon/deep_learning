# ---------------------------------------------------------------------
# Implements An Extension for Model Profiling
#
# Author: Matthew Drnevich
# ---------------------------------------------------------------------

from pylearn2.termination_criteria import TerminationCriterion
from time import time

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
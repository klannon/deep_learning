from __future__ import division
from time import clock

class Validator(object):

    def __init__(self, experiment, terms):
        self.terms = terms
        self.exp = experiment
        self._num_epochs = terms["epochs"]
        self._timeout = terms["timeout"]
        self._m = terms["plateau"]["m"]
        self._w = terms["plateau"]["w"]
        self._clock = clock()
        self.failed = ''

    def check(self):
        if self._num_epochs and self.epochs >= self._num_epochs:
            self.failed = 'Reached max epochs'
            return False
        if self._timeout and self.time >= self._timeout:
            self.failed = 'Reached max time'
            return False
        if (self._m and self._w) and (len(self.exp.results) >= self._w) and (self.slope <= self._m):
            self.failed = 'Reached plateau'
            return False
        return True

    @property
    def epochs(self):
        return len(self.exp.results)

    @property
    def slope(self):
        return (self.exp.results[-1].test_accuracy - self.exp.results[-self._w].test_accuracy) / (self._w - 1)

    @property
    def time(self):
        return clock() - self._clock
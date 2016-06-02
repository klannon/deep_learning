from __future__ import division
from time import clock

class Validator(object):

    def __init__(self, experiment, terms):
        self.terms = terms
        self.exp = experiment
        self._num_epochs = terms["epochs"]
        self._timeout = terms["timeout"]
        self._x = terms["plateau"]["x"]
        self._y = terms["plateau"]["y"]
        self._m = self._y / self._x
        self._w = 1000
        self._clock = clock()
        self.failed = ''

    def check(self):
        if self._num_epochs and self.epochs >= self._num_epochs:
            self.failed = 'Reached max epochs'
            return False
        if self._timeout and self.time >= self._timeout:
            self.failed = 'Reached max time'
            return False
        if (self._x and self._y) and self.update_w() and (self.slope <= self._m):
            self.failed = 'Reached plateau'
            return False
        return True

    @property
    def epochs(self):
        return len(self.exp.results)

    @property
    def slope(self):
        return (self.exp.results[-1].test_accuracy - self.exp.results[-self._w].test_accuracy) / \
               sum([x.num_seconds for x in self.exp.results[1-self._w:]])

    @property
    def time(self):
        return clock() - self._clock

    def update_w(self):
        counts = [x.num_seconds for x in self.exp.results]
        for i in reversed(xrange(len(counts))):
            if sum(counts[i:]) >= self._x:
                self._w = len(counts) - i + 1
                return True if len(counts) >= self._w else False
        return False
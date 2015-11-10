# Pylearn2 dataset for physics data.
#__authors__ = "Peter Sadowski"
# May 2014

###########
# CHANGES
#  Changed PHYSICS so that it has a managing function which takes a data file
#  and returns the training, validation, and testing sets at once. There also
#  was no need for most of the variables used in the PHYSICS class so they were
#  removed or merged into csvData.
#
#  Fixed the shape of the labels array
###########

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
import os
import numpy as np
import pickle as pkl
import csvData
import os

def PHYSICS(dataPath,
            trainPerc,
            validPerc,
            *args,
            **kwargs):

    benchmark = dataPath.split(os.sep)[-1].split('.')[0] # Returns the name of the file without its extension

    train, valid, test = csvData.getData(dataPath, trainPerc, validPerc, benchmark=benchmark, **kwargs)

    return (_PHYSICS(train, 'train', benchmark),
            _PHYSICS(valid, 'valid', benchmark),
            _PHYSICS(test, 'test', benchmark))

class _PHYSICS(dense_design_matrix.DenseDesignMatrix):
    def __init__(self,
                 data,
                 which_set='?',
                 benchmark=''):

        self.args = locals()
        
        # Need to allocate two arrays X (inputs) and y (targets)

        print 'Data loaded: benchmark {} ({})'.format(benchmark, which_set)

        # Initialize the superclass. DenseDesignMatrix
        super(_PHYSICS, self).__init__(X=data['data'], y=data['labels'])
        
    def standardize(self, X):
        """
        Standardize each feature:
        1) If data contains negative values, we assume its either normally or uniformly distributed, center, and standardize.
        2) elseif data has large values, we set mean to 1.
        """
        
        for j in range(X.shape[1]):
            vec = X[:, j]
            if np.min(vec) < 0:
                # Assume data is Gaussian or uniform -- center and standardize.
                vec = vec - np.mean(vec)
                vec = vec / np.std(vec)
            elif np.max(vec) > 1.0:
                # Assume data is exponential -- just set mean to 1.
                vec = vec / np.mean(vec)
            X[:,j] = vec
        return X
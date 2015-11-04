# Pylearn2 dataset for physics data.
#__authors__ = "Peter Sadowski"
# May 2014

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
import os
import numpy as np
import pickle as pkl
import csv

class PHYSICS(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, 
                 which_set,
                 benchmark,
                 derived_feat=True,
                 version='',
                 seed=None, # Randomize data order if seed is not None
                 start=0, 
                 stop=np.inf):

        self.args = locals()
        path = os.environ['PYLEARN2_DATA_PATH']

        if derived_feat == 'False':
            derived_feat = False
        elif derived_feat == 'True':
            derived_feat = True

        if benchmark == 1:
            inputfile = '%s/HIGGS.csv' % path
        elif benchmark == 2:
            inputfile = '%s/SUSY.csv' % path
        
        # Need to allocate two arrays X (inputs) and y (targets)
        # We know the size so let's allocate them from the outset instead

        # Define set of training, testing, and validation sets
        if benchmark == 1:
            # HIGGS
            ntrain = 10000000 
            nvalid = 500000 
            ntest  = 500000
        elif benchmark == 2:
            # SUSY
            ntrain = 4000000 
            nvalid = 500000 
            ntest  = 500000


        if which_set == 'train':
            offset = 0
            nrows = ntrain
        elif which_set == 'valid':
            offset = ntrain
            nrows = nvalid
        elif which_set == 'test':
            offset = ntrain+nvalid
            nrows = ntest

        # Define the feature lists and relevant columns
        if benchmark == 1:
            if derived_feat == 'only':
                xcolmin = 22
                xcolmax = 29
                ycolmin = 0
                ycolmax = 1
            elif not derived_feat:
                xcolmin = 1
                xcolmax = 22
                ycolmin = 0
                ycolmax = 1
            else:
                xcolmin = 1
                xcolmax = 29
                ycolmin = 0
                ycolmax = 1
        elif benchmark == 2:
            if derived_feat == 'only':
                xcolmin = 9
                xcolmax = 19
                ycolmin = 0
                ycolmax = 1
            if not derived_feat:
                xcolmin = 1
                xcolmax = 9
                ycolmin = 0
                ycolmax = 1
            else:
                xcolmin = 1
                xcolmax = 19
                ycolmin = 0
                ycolmax = 1


        # Limit number of samples
        stop = min(stop,nrows)
        X = np.empty([stop,xcolmax-xcolmin], dtype='float32')
        y = np.empty([stop,ycolmax-ycolmin], dtype='float32')

        # Randomize data order.
        indices = np.arange(X.shape[0])
        if seed:
            rng = np.random.RandomState(42)  # reproducible results with a fixed seed
            rng.shuffle(indices)

        # Finally, let's read in the data
        reader = csv.reader(open(inputfile))
        nskipped = 0
        nread = 0
        for row in reader:

            # Possibly skip some events at the start
            if nskipped < offset:
                nskipped += 1
                continue

            # Stop when we get to the number we want
            if nread >= stop:
                break

            # If we're here, it means we want to save this result
            X[indices[nread]] = row[xcolmin:xcolmax]
            y[indices[nread]] = row[ycolmin:ycolmax]
            nread += 1

        print 'Data loaded: benchmark {} ({})'.format(benchmark,which_set)

        # Initialize the superclass. DenseDesignMatrix
        super(PHYSICS,self).__init__(X=X, y=y)
        
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





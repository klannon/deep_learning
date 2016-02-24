# ---------------------------------------------------------------------
# Loads data into a dense_design_matrix object
#
# Authors: Matthew Drnevich
#
# ---------------------------------------------------------------------

from pylearn2.datasets import dense_design_matrix
import numpy as np

class PHYSICS(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, data_X=None, data_Y=None, benchmark='', which_set='?'):
        self.args = locals()
        self.benchmark = benchmark
        self.which_set = which_set

        # Initialize the superclass. DenseDesignMatrix
        if isinstance(data_X, np.ndarray):
            super(PHYSICS, self).__init__(X=data_X, y=data_Y)
            print 'Data loaded: {} ({})'.format(benchmark, which_set)

    def load_data(self, data_X, data_Y):

        # Initialize the superclass. DenseDesignMatrix
        self.__init__(data_X, data_Y, self.benchmark, self.which_set)

        #print 'Data loaded: {} ({})'.format(benchmark, which_set)

    def load_from_file(self, path_to_X, path_to_Y):

        X = np.load(path_to_X)
        Y = np.load(path_to_Y)

        self.__init__(X, Y, self.benchmark, self.which_set)
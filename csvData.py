from __future__ import print_function
import re
import numpy as np
from math import floor

def split(string, sep):
    temp = re.split("[{0}]+".format(sep), string) # splits the string around sep, i.e. isolates instances of sep
    temp.pop()
    return temp

def readFile(path, sep):
    data = np.array([split(line, sep) for line in file(path)], dtype=np.float32) # reads the data into a numpy array efficiently and converts the strings to 32-bit floats
    return data, data.shape[0], data.shape[1] # returns array, rows, columns

def getData(pathTrain, pathTest):
    testD, testROWS, testCOLS = readFile(pathTest, ',')
    trainD, trainROWS, trainCOLS = readFile(pathTrain, ',')

    # Print some examples
    print("test rows {0}; cols {1}".format(testROWS, testCOLS))
    print("train rows {0}; cols {1}".format(trainROWS, trainCOLS))
    for i in xrange(10):
        print ('Label: {0} \t Train: {1}'.format(trainD[i, 0], trainD[i, 1:]))

    trainPercent = 0.75
    cutoff = floor(trainPercent*trainROWS)

    # Should we shuffle a copy or the actual data?

    np.random.shuffle(trainD)

    trainData = {'data': trainD[:cutoff, 1:], 'labels': trainD[:cutoff, 0], 'size': lambda: (cutoff, trainD.shape[1])}
    valData = {'data': trainD[cutoff:, 1:], 'labels': trainD[cutoff:, 0], 'size': lambda: (trainROWS-cutoff, trainD.shape[1])}

    return trainData, valData, testD
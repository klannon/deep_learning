from __future__ import print_function
import re
import numpy as np
from math import floor

def split(string, sep):
    temp = re.split("[{0}]+".format(sep), string) # splits the string around sep, i.e. isolates instances of sep
    temp.pop() # Remove the empty string at the end
    return temp

def readFile(path, sep):
    data = np.array([split(line, sep) for line in file(path)], dtype=np.float32) # reads the data into a numpy array efficiently and converts the strings to 32-bit floats
    return data, data.shape[0], data.shape[1] # returns array, rows, columns

def getData(pathData, trainPercent, validPercent):
    trainD, trainROWS, trainCOLS = readFile(pathData, ',')

    # Print some examples
    print("train rows {0}; cols {1}".format(trainROWS, trainCOLS))
    for i in xrange(10):
        print ('Label: {0} \t Train: {1}'.format(trainD[i, 0], trainD[i, 1:]))

    trCutoff = floor(trainPercent*trainROWS)
    vaCutoff = trCutoff + floor(validPercent*trainROWS)

    # Test data will be the remainder

    # Should we shuffle a copy or the actual data?

    np.random.shuffle(trainD)

    trainData = {'data': trainD[:trCutoff, 1:], 'labels': trainD[:trCutoff, 0], 'size': lambda: (trCutoff, trainD.shape[1])}
    valData = {'data': trainD[trCutoff:vaCutoff, 1:], 'labels': trainD[trCutoff:vaCutoff, 0], 'size': lambda: (vaCutoff-trCutoff, trainD.shape[1])}
    testData = {'data': trainD[trCutoff+vaCutoff:, 1:], 'labels': trainD[trCutoff+vaCutoff:, 0], 'size': lambda: (trainROWS-vaCutoff, trainD.shape[1])}

    return trainData, valData, testData
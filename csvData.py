from __future__ import print_function
import numpy as np
from math import floor
import csv

# Used the input method from previous physics.py
def readFile(path, benchmark, derived_feat=True):

    if derived_feat == 'False':
        derived_feat = False
    elif derived_feat == 'True':
        derived_feat = True

    if benchmark == 'HIGGS':
        nrows = 11000000
    elif benchmark == 'SUSY':
        nrows = 5000000
    else:
        raise Exception('Not a valid file, needs to be HIGGS or SUSY')

    # Define the feature lists and relevant columns
    if benchmark == 'HIGGS':
        if derived_feat == 'only':
            xcolmin = 22
            xcolmax = 29
        elif not derived_feat:
            xcolmin = 1
            xcolmax = 22
        else:
            xcolmin = 1
            xcolmax = 29
    elif benchmark == 'SUSY':
        if derived_feat == 'only':
            xcolmin = 9
            xcolmax = 19
        if not derived_feat:
            xcolmin = 1
            xcolmax = 9
        else:
            xcolmin = 1
            xcolmax = 19

    data = np.empty([nrows, xcolmax-xcolmin+1], dtype='float32')

    reader = csv.reader(open(path))
    nread = 0
    for row in reader:
        temp = row[xcolmin:xcolmax]
        temp.insert(0, row[0])
        data[nread] = temp
        nread += 1

    return data, data.shape[0], data.shape[1]

def getData(pathData, trainPercent, validPercent, benchmark, derived_feat=True):
    trainD, trainROWS, trainCOLS = readFile(pathData, benchmark, derived_feat)

    # Print some examples
    print("train rows {0}; cols {1}".format(trainROWS, trainCOLS))
    #for i in xrange(10):
    #    print ('Label: {0} \t Train: {1}'.format(trainD[i, 0], trainD[i, 1:]))

    trCutoff = floor(trainPercent*trainROWS)
    vaCutoff = trCutoff + floor(validPercent*trainROWS)

    # Test data will be the remainder

    # Should we shuffle a copy or the actual data?

    np.random.shuffle(trainD)

    trainData = {'data': trainD[:trCutoff, 1:], 'labels': trainD[:trCutoff, 0], 'size': lambda: (trCutoff, trainD.shape[1])}
    valData = {'data': trainD[trCutoff:vaCutoff, 1:], 'labels': trainD[trCutoff:vaCutoff, 0], 'size': lambda: (vaCutoff-trCutoff, trainD.shape[1])}
    testData = {'data': trainD[trCutoff+vaCutoff:, 1:], 'labels': trainD[trCutoff+vaCutoff:, 0], 'size': lambda: (trainROWS-vaCutoff, trainD.shape[1])}

    return trainData, valData, testData
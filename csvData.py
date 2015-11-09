from __future__ import print_function
import numpy as np
from math import floor
import re

# Used the input method from previous physics.py
def readFile(path, benchmark, derived_feat=True, sep='[^(?:\-?\d+\.?\d*e?\d*)]', nrows=None, xcolmin=1, xcolmax=None):

    if derived_feat == 'False':
        derived_feat = False
    elif derived_feat == 'True':
        derived_feat = True

    if nrows is not int:
        if benchmark == 'HIGGS':
            nrows = 11000000
        elif benchmark == 'SUSY':
            nrows = 5000000
        elif 'test' in benchmark:
            nrows = 90000
        elif 'train' in benchmark:
            nrows = 1700000
        else:
            count = 0
            with open(path) as f:
                for line in f:
                    count +=1
            nrows = count

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
        elif not derived_feat:
            xcolmin = 1
            xcolmax = 9
        else:
            xcolmin = 1
            xcolmax = 19

    nread = 0
    reader = (re.split(sep, line) for line in open(path))

    if not xcolmax:
        firstLine = next(reader)
        rrow = filter(None, firstLine)
        xcolmax = len(rrow)
        data = np.empty([nrows, xcolmax-xcolmin+1], dtype='float32')
        temp = rrow[xcolmin:xcolmax]
        temp.insert(0, rrow[0])
        data[nread] = temp
        nread += 1
    else:
        data = np.empty([nrows, xcolmax-xcolmin+1], dtype='float32')

    for row in reader:
        print(row)
        rrow = filter(None, row)
        temp = rrow[xcolmin:xcolmax]
        temp.insert(0, rrow[0])
        data[nread] = temp
        nread += 1
        if nread > nrows:
            break

    return data, data.shape[0], data.shape[1]

def getData(pathData, trainPercent, validPercent, benchmark, derived_feat=True, sep='[^(?:\-?\d+\.?\d*e?\d*)]', nrows=None, xcolmin=1, xcolmax=None):
    trainD, trainROWS, trainCOLS = readFile(pathData, benchmark, derived_feat, sep, nrows, xcolmin, xcolmax)

    # Print some examples
    #print("train rows {0}; cols {1}".format(trainROWS, trainCOLS))
    #for i in xrange(10):
    #    print ('Label: {0} \t Train: {1}'.format(trainD[i, 0], trainD[i, 1:]))

    trCutoff = floor(trainPercent*trainROWS)
    vaCutoff = trCutoff + floor(validPercent*trainROWS)

    # Test data will be the remainder

    np.random.shuffle(trainD)

    trainData = {'data': trainD[:trCutoff, 1:], 'labels': trainD[:trCutoff, 0], 'size': lambda: (trCutoff, trainD.shape[1])}
    valData = {'data': trainD[trCutoff:vaCutoff, 1:], 'labels': trainD[trCutoff:vaCutoff, 0], 'size': lambda: (vaCutoff-trCutoff, trainD.shape[1])}
    testData = {'data': trainD[trCutoff+vaCutoff:, 1:], 'labels': trainD[trCutoff+vaCutoff:, 0], 'size': lambda: (trainROWS-vaCutoff, trainD.shape[1])}

    return trainData, valData, testData

if __name__ == '__main__':
    readFile('OSUtorch/test_all_3v_ttbar_wjet.txt', 'test')
    #readFile('SUSY.csv', 'SUSY')
from __future__ import print_function
import numpy as np
from math import floor
import re

# Used the input method from previous physics.py
def readFile(path,
             benchmark='',
             derived_feat=True,
             sep='[^(?:\-?\d+\.?\d*e?\d*)]',
             nrows=None,
             xcolmin=1,
             xcolmax=None,
             numLabels=1):

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
                    count += 1
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
    labelRange = map(lambda x: str(x+1), range(numLabels))
    reader = (re.split(sep, line) for line in open(path))

    if not xcolmax:
        firstLine = next(reader)
        rrow = filter(None, firstLine)
        xcolmax = len(rrow)
        data = np.empty([nrows, xcolmax-xcolmin+numLabels], dtype='float32')
        temp = rrow[xcolmin:xcolmax]
        if numLabels == 1:
            label = [rrow[0]]
        else:
            try:
                ix = labelRange.index(rrow[0])
            except ValueError:
                raise Exception('Label in your data is not in the range of numLabels provided')
            label = [1 if num == ix else 0 for num in xrange(numLabels)]
        temp = label + temp
        data[nread] = temp
        nread += 1
    else:
        data = np.empty([nrows, xcolmax-xcolmin+numLabels], dtype='float32')

    for row in reader:
        rrow = filter(None, row)
        temp = rrow[xcolmin:xcolmax]
        if numLabels == 1:
            label = [rrow[0]]
        else:
            try:
                ix = labelRange.index(rrow[0])
            except ValueError:
                raise Exception('Label in your data is not in the range of numLabels provided')
            label = [1 if num == ix else 0 for num in xrange(numLabels)]
        temp = label + temp
        data[nread] = temp
        nread += 1
        if nread > nrows:
            break

    return data, data.shape[0], data.shape[1]

def getData(pathData,
            trainPercent,
            validPercent,
            benchmark='',
            derived_feat=True,
            sep='[^(?:\-?\d+\.?\d*e?\d*)]',
            nrows=None,
            xcolmin=1,
            xcolmax=None,
            numLabels=1):
    trainD, trainROWS, trainCOLS = readFile(pathData, benchmark, derived_feat, sep, nrows, xcolmin, xcolmax, numLabels)

    trCutoff = floor(trainPercent*trainROWS)
    vaCutoff = trCutoff + floor(validPercent*trainROWS)

    # Test data will be the remainder

    np.random.shuffle(trainD)

    trainData = {'data': trainD[:trCutoff, numLabels:], 'size': lambda: (trCutoff, trainD.shape[1])}
    valData = {'data': trainD[trCutoff:vaCutoff, numLabels:], 'size': lambda: (vaCutoff-trCutoff, trainD.shape[1])}
    testData = {'data': trainD[trCutoff+vaCutoff:, numLabels:], 'size': lambda: (trainROWS-vaCutoff, trainD.shape[1])}

    if numLabels == 1:
        trainData['labels'] = trainD[:trCutoff, 0:numLabels].reshape(trCutoff, 1)
        valData['labels'] = trainD[trCutoff:vaCutoff, 0:numLabels].reshape(vaCutoff-trCutoff, 1)
        testData['labels'] = trainD[trCutoff+vaCutoff:, 0:numLabels].reshape(trainROWS-vaCutoff, 1)
    else:
        trainData['labels'] = trainD[:trCutoff, 0:numLabels]
        valData['labels'] = trainD[trCutoff:vaCutoff, 0:numLabels]
        testData['labels'] = trainD[trCutoff+vaCutoff:, 0:numLabels]

    return trainData, valData, testData
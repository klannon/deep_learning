from __future__ import print_function
import numpy as np
from math import floor
import re

# Used the input method from previous physics.py
def readFile(pathToData,
             benchmark='',
             nrows=None,
             xcolmin=1,
             xcolmax=None,
             numLabels=1,
             sep='[^(?:\-?\d+\.?\d*e?\d*)]'):
    """
    pathToData : location of file to be loaded
        This is pretty self-explanatory: where is the file you want
        to load located?

    nrows : number of rows of data to be read in
        This variable seems to be of dubious value because readFile()
        seems like it could be implemented without it, but I won't
        play with it for now.

    xcolmin : furthest left column that will be read in
        First column from the raw data that will be read in and thus
        the first column that will appear in the list created from the
        read-in data

    xcolmax : furthest right column that will be read in
        Last column from the raw data that will be read in and thus the
        last column that will appear in the list created from the read-
        in data

    sep : delimiter between entries in data
        The character that separates columns in the data file. Though
        the script (being named csv.py) implicitly assumes that the
        delimiter will be a comma, the default value allows a lot more
        flexibility for possible delimiters
    """

    if nrows is not int:
        if 'test' in benchmark:
            nrows = 90000
        elif 'train' in benchmark:
            nrows = 1700000
        else:
            count = 0
            with open(pathToData) as f:
                for line in f:
                    count += 1
            nrows = count

    print("Loading %s, which has %i rows" %(pathToData, nrows))
    nread = 0
    labelRange = map(lambda x: str(x+1), range(numLabels))
    reader = (re.split(sep, line) for line in open(pathToData))

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
        if (nread % 10000 == 0):
            print(nread)
        if nread > nrows:
            break

    return data, data.shape[0], data.shape[1]

# Changing default value of trainFraction because skipping the randomization of data should not be encouraged
def getData(pathToData,
            trainFraction=1,
            benchmark='',
            nrows=None,
            xcolmin=1,
            xcolmax=None,
            numLabels=1,
            sep='[^(?:\-?\d+\.?\d*e?\d*)]'):
    """
    loads data from a file

    This funciton loads data (the assumption is csv, but any separator
    that does not violate the regex sep will be accepted too) from file
    located at pathToData and splits it into training and validation
    sets if necessary.
    @getData is sensitive to whether the data needs to be split up into
    subsets (the assumption is splitting a file that contains training
    and validation data into subsets.  Testing data is assumed to be
    provided in its own file).

    Parameters
    ----------
    pathToData : location of file to be loaded
        This is pretty self-explanatory: where is the file you want
        to load located?

    trainFraction : fraction of data for training
        Number less than 1 indicating how much of the data in the file
        is for training.

    sep : delimiter between entries in data
        The character that separates columns in the data file. Though
        the script (being named csv.py) implicitly assumes that the
        delimiter will be a comma, the default value allows a lot more
        flexibility for possible delimiters
    """
    data, dataROWS, dataCOLS = readFile(pathToData, benchmark, nrows, xcolmin, xcolmax, numLabels, sep)

    # Don't do this if you want data for training, it needs to be shuffled....only for testing
    if not trainFraction:
        testData = {'data': data[:, numLabels:], 'size': lambda: (dataROWS, data.shape[1])}
        if numLabels == 1:
            testData['labels'] = data[:, 0:numLabels].reshape(dataROWS, 1)
        else:
            testData['labels'] = data[:, 0:numLabels]
        return testData

    trCutoff = floor(trainFraction*dataROWS) # last row that training
    # data appears on

    # Test data will be the remainder

    np.random.shuffle(data)

    trainData = {'data': data[:trCutoff, numLabels:], 'size': lambda: (trCutoff, data.shape[1])}
    valData = {'data': data[trCutoff:, numLabels:], 'size': lambda: (dataROWS-trCutoff, data.shape[1])}

    if numLabels == 1:
        trainData['labels'] = data[:trCutoff, 0:numLabels].reshape(trCutoff, 1)
        valData['labels'] = data[trCutoff:, 0:numLabels].reshape(dataROWS-trCutoff, 1)
    else:
        trainData['labels'] = data[:trCutoff, 0:numLabels]
        valData['labels'] = data[trCutoff:, 0:numLabels]

    return trainData, valData
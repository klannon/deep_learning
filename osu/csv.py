from __future__ import print_function
import numpy as np
from math import floor
import re

__author__ = 'Matthew Drnevich'


# Used the input method from previous physics.py
def readFile(pathToData, *args, **kwargs):
    """
    pathToData : location of file to be loaded
        This is pretty self-explanatory: where is the file you want
        to load located?

    kwargs :
        benchmark : a string that is associated with certain file sizes.
            It is recommended to use the file's name for this argument
            which will then set nrows accordingly.

        nrows : number of rows of data to be read in
            If provided this will speed up the program by not requiring
            the program to cycle through the file and count the number
            of lines, or if you just don't want to read the entire file.

        xcolmin : furthest left column that will be read in for input
            First column from the raw data that will be read in for inputs
            and thus the first column that will appear in the inputs array
            created from the read-in data.

        xcolmax : furthest right column that will be read in
            Last column from the raw data that will be read in and thus the
            last column that will appear in the list created from the read-
            in data.

        numLabels : the number of labels your network will output
            Your data file should only have one column of numbers
            describing the expected label for a training set. If
            numLabels=1 (default) this column should contain 0s and 1s.
            If numLabels>1 then this column should contain a number
            in [1,numLabels] such that the number indicates which node
            produced a value of 1 (assumes a softmax output layer).

        sep : A regular expression for what you are trying to isolate
            in the data file. The default will isolate any number with
            or without a decimal XXX.XXe+/-XX value at the
            end.
    """

    # Define default values and then override with user-defined values
    d = dict(benchmark='',
                  nrows=None,
                  xcolmin=1,
                  xcolmax=None,
                  numLabels=1,
                  sep='(\-?\d+\.?\d*e?(?:\+|\-)?\d*)')
    d.update(kwargs)

    # Define variables
    nrows = d['nrows']
    benchmark = d['benchmark']
    xcolmin = d['xcolmin']
    xcolmax = d['xcolmax']
    numLabels = d['numLabels']
    sep = d['sep']

    # If the number of rows isn't given the derive it from benchmark or count it.
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
    # Define the range that the labels can be within
    labelRange = map(lambda x: str(x+1), range(numLabels))
    # Create a generator that splits each line of the file
    reader = (re.findall(sep, line) for line in open(pathToData))

    # If the maximum number of columns for the input data isn't provided then read the first line and count the columns.
    # In both cases a numpy array of the proper sihape is created with all 0s.
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

    # This reads through each line of the file generator and assigns the values to the numpy array.
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

    # Returns the data, number of rows, and number of columns.
    return data, data.shape[0], data.shape[1]

# Changing default value of trainFraction because skipping the randomization of data should not be encouraged
def getData(pathToData,
            trainFraction=1.0,
            numLabels=1,
            *args,
            **kwargs):
    """
    loads data from a file

    This function loads data from the file located at pathToData and
    splits it into training and validation sets if necessary.
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

    **Note: validFraction = 1 - trainFraction

    numLabels : the number of labels your network will output
            Your data file should only have one column of numbers
            describing the expected label for a training set. If
            numLabels=1 (default) this column should contain 0s and 1s.
            If numLabels>1 then this column should contain a number
            in [1,numLabels] such that the number indicates which node
            produced a value of 1 (assumes a softmax output layer).

    kwargs : These are arguments to be pass into readFile
    """
    data, dataROWS, dataCOLS = readFile(pathToData, *args, numLabels=numLabels, **kwargs)

    # Don't do this if you want data for training, it needs to be shuffled....only for testing, not training
    if not trainFraction:
        testData = {'data': data[:, numLabels:], 'size': lambda: (dataROWS, data.shape[1])}
        if numLabels == 1:
            testData['labels'] = data[:, 0:numLabels].reshape(dataROWS, 1)
        else:
            testData['labels'] = data[:, 0:numLabels]
        return testData

    # This defines the last row that will be used for training data
    # The rest of the data will be used for validation data.
    trCutoff = floor(trainFraction*dataROWS)

    # Randomize the data
    np.random.shuffle(data)

    # Create dictionaries of the data with keys: 'data', 'labels', and 'size'. Note that size is a function.
    trainData = {'data': data[:trCutoff, numLabels:], 'size': lambda: (trCutoff, data.shape[1])}
    valData = {'data': data[trCutoff:, numLabels:], 'size': lambda: (dataROWS-trCutoff, data.shape[1])}

    # Reshape the 'labels' array if necessary because numpy can be dumb.
    if numLabels == 1:
        trainData['labels'] = data[:trCutoff, 0:numLabels].reshape(trCutoff, 1)
        valData['labels'] = data[trCutoff:, 0:numLabels].reshape(dataROWS-trCutoff, 1)
    else:
        trainData['labels'] = data[:trCutoff, 0:numLabels]
        valData['labels'] = data[trCutoff:, 0:numLabels]

    return trainData, valData
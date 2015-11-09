from __future__ import print_function
import numpy as np
from math import floor
import re

# Used the input method from previous physics.py
def readFile(pathToData, nrows=None, xcolmin=1, xcolmax=None, sep='[^-?\d+\.?\d*?e\d*]'):
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
        count = 0
        with open(pathToData) as f:
            for line in f:
                count +=1
        nrows = count
    print("Loading %s, which has %i rows" %(pathToData, nrows))

    nread = 0
    reader = (re.split(sep, line) for line in open(pathToData))

    firstLine = next(reader)
    rrow = filter(None, firstLine)
    xcolmax = len(rrow)
    data = np.empty([nrows, xcolmax-xcolmin+2], dtype='float32')
    temp = rrow[xcolmin:xcolmax]
    temp.insert(0, rrow[0])
    temp.insert(0, 1)
    data[nread] = temp
    nread += 1

    for row in reader:
        rrow = filter(None, row) # makes sure there are no null values
        temp = rrow[xcolmin:xcolmax]
        temp.insert(0, rrow[0])
        temp.insert(0, 1)
        data[nread] = temp
        nread += 1
        if (nread % 10000 == 0):
            print(nread)
        if nread > nrows:
            break

    return data, data.shape[0], data.shape[1]

def getData(pathToData, trainFraction=None, sep='[^-?\d+\.?\d*?e\d*]'):
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
    data, dataROWS, dataCOLS = readFile(pathToData, nrows=None, xcolmin=2, xcolmax=None, sep='[^-?\d+\.?\d*?e\d*]')

    if not trainFraction:
        testData = {'data': data[:, 2:], 'labels': data[:, 0:2]}
        return testData

    trCutoff = floor(trainFraction*dataROWS) # last row that training
    # data appears on

    # Should we shuffle a copy or the actual data?
    
    trainData = {'data': data[:trCutoff, 2:], 'labels': data[:trCutoff, 0:2]}
    valData = {'data': data[trCutoff:, 2:], 'labels': data[trCutoff:, 0:2]}

    return trainData, valData

if __name__ == '__main__':
    readFile('../OSUtorch/test_all_3v_ttbar_wjet.txt')
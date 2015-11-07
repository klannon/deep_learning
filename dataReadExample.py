import csvData
from os import sep

__doc__ = """
This is an example implementation of the csvData.getData function.
It is worth noting that this function can be extended to any dataset
and you can narrow the data selection by supplying certain parameters
to the function, or by default it will read every row and column of
the file.
"""

dataPath = 'OSUtorch/train_all_3v_ttbar_wjet.txt'
testPath = 'OSUtorch/test_all_3v_ttbar_wjet.txt'

benchmarkD = dataPath.split(sep)[-1].split('.')[0]  # Name of file w/o the extension, though it works with just
benchmarkT = testPath.split(sep)[-1].split('.')[0]  # the word "train" or "test"

trainData, valData = csvData.getData(dataPath, 0.8, 0.2, benchmarkD)[:2]
testData = csvData.getData(testPath, 1, 0, benchmarkT)[0]
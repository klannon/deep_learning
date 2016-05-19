import sys
import os

source_file_name = sys.argv[1]
print ("Loading data from %s" % source_file_name)
x = open(source_file_name, "r")
source_experiment = os.path.splitext(source_file_name)[0]
print source_experiment
maximum = 0
maximum_epoch = 0
epochs = 0
for line in x:
	value = float(line)
	if value > maximum:
		maximum = value
		maximum_epoch = epochs
	epochs += 1

print("Maximum: %f" % maximum)
print("At epoch %i" % maximum_epoch)

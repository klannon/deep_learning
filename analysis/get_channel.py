import sys
import os

source_file_name = sys.argv[1]
print ("Loading data from %s" % source_file_name)
x = open(source_file_name, "r")
source_experiment = os.path.splitext(source_file_name)[0]
print source_experiment
channel_to_extract = ""
try:
	channel_to_extract = sys.argv[2]
except IndexError:
	channel_to_extract = "test_y_misclass"
destination_file_name = (channel_to_extract + "_" + source_experiment
                         + ".log")
y = open(destination_file_name, "w")
for line in x:
	found = line.find(channel_to_extract)
	channel_str = None
	if found != -1:
		try:
			channel_str = line.split(":")[1]
		except:
			print line
	if channel_str != None:
		channel_str = channel_str.strip()
		if (channel_to_extract == "test_y_misclass"):
			misclass = float(channel_str)
			accuracy = 1 - misclass
			accuracy_str = str(accuracy)
			channel_str = accuracy_str
		print channel_str
		y.write(channel_str + "\n")

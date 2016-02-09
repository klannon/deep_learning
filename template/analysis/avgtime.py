import sys
import os

source_file_name = sys.argv[1]
print ("Loading data from %s" % source_file_name)
x = open(source_file_name, "r")
source_experiment = os.path.splitext(source_file_name)[0]
print source_experiment
channel_to_extract = "total_seconds_last_epoch"
times = []
for line in x:
	time = 0
	found = line.find(channel_to_extract)
	channel_str = None
	if found != -1:
		try:
			channel_str = line.split(":")[1]
		except:
			print line
	if channel_str != None:
		channel_str = channel_str.strip()
		time = float(channel_str)
		if time == (0.0):
			print(False)
		if time != (0.0):
			print(True)
			minutes = time / 60
			times.append(minutes)

print ("Average time per epoch (min):%f" %reduce(lambda x, y: x + y, times) / len(times))

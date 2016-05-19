import sys
import os
import matplotlib.pyplot as plt

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
		minutes = time / 60
		times.append(minutes)

############################
# Name the plot and axises #
############################

plt.plot(times) # adds the list of data read in to the plot

plot_title = raw_input("Enter the plot name or press enter for " +
                       "defaults (minutes_last_epoch): ")
plot_x_label = ""
plot_y_label = ""
if (plot_title == ""): # if you want to plot accuracy vs num epochs
	print "Using defaults:"
	plot_title = "Minutes per epoch vs Epochs"
	print "Plot title: %s" % plot_title
	plot_x_label = "Epochs"
	print "Plot x label: %s" % plot_x_label
	plot_y_label = "Minutes to train"
	print "Plot y label: %s" % plot_y_label

##############################################
# Assigns the plot labels to the plot object #
##############################################

plt.xlabel(plot_x_label) # sets x label on the graph
plt.ylabel(plot_y_label) # sets y label on the graph
plt.title(plot_title) # sets title on the graph

#############################################
# Writes the plot to a file and displays it #
#############################################

source_file_name = os.path.splitext(sys.argv[1])[0] # gets file name without extension
destination_file_name = "plot_minutes_last_epoch" + source_file_name + ".png"
plt.savefig(destination_file_name, format='png', dpi=500)
plt.show()


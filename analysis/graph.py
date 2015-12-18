####################################################
# plot a file in the form output by get_channel.py #
# Author: Colin Dabalin                            #
####################################################

import matplotlib.pyplot as plt
import sys
import os

#####################################################
# Opens the data file and adds the data to the plot #
#####################################################

f = open(sys.argv[1]) # >> python graph.py "file name"
y = []

for line in f: # the file is assumed to contain 1 item of data per line
	y.append(float(line))

plt.plot(y) # adds the list of data read in to the plot

############################
# Name the plot and axises #
############################

plot_title = raw_input("Enter the plot name or press enter for " +
                       "defaults (accuracy): ")
plot_x_label = ""
plot_y_label = ""
if (plot_title == ""): # if you want to plot accuracy vs num epochs
	print "Using defaults:"
	plot_title = "Test Accuracy vs Epochs"
	print "Plot title: %s" % plot_title
	plot_x_label = "Epochs"
	print "Plot x label: %s" % plot_x_label
	plot_y_label = "test_y_accuracy"
	print "Plot y label: %s" % plot_y_label
else: # if you want to customize the plot labels
	plot_x_label = raw_input("Enter the x-axis label: ")
	plot_y_label = raw_input("Enter the y-axis label: ")

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
destination_file_name = "plot_" + source_file_name + ".png"
plt.savefig(destination_file_name, format='png', dpi=500)
plt.show()

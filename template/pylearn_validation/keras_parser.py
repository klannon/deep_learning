from __future__ import print_function
import sys

if (len(sys.argv) != 2): # if no log file is provided as an argument
    print("Usage: " + sys.argv[0] + " LOG_FILE_NAME")
    exit(-1)

f = None
    
try:
    f = open(sys.argv[1])
except IOError:
    print(sys.argv[1] + ": not a valid file")

accuracies = []
times = []

while(True):
    line = f.readline()
    if(line == ""):
        break
    if("Epoch" in line):
        data = f.readline().strip().split(" - ")
        # print(data)
        time = data[0].split("s")[0]
        times.append(time)
        print(time)
        loss = data[1].split("loss: ")[1]
        # print(loss)
        accuracy = data[2].split("acc: ")[1]
        accuracies.append(accuracy)

f.close()

input_name = sys.argv[1].split(".")[0]
output = open(("%s_time.log" % input_name), "w")
for time in times:
    output.write(time)
    output.write("\n")

output.close()

from __future__ import print_function

__author__ = 'Matt'

import re
import os
import os.path

def by_time(x,y, recent=True):
    val = cmp(os.path.getmtime(x), os.path.getmtime(y))
    return val if not recent else -val

def by_epochs(x,y, largest=True):
    x, y = os.stat(x), os.stat(y)
    val = cmp(x.st_size, y.st_size)
    return val if not largest else -val

results = os.getcwd()+os.sep+'results'+os.sep

# Extract .log files and add their prefix
log_files = [results+x for x in filter(lambda x: x.endswith('.log'), next(os.walk(results))[2])]

sorted_logs = sorted(log_files, by_epochs)

prefix = os.path.split(sorted_logs[0])[1].split('_')[0]
efile = open(sorted_logs[0])

eseen = 0
for line in efile:
    line = re.split('\s+', line)
    if 'Epochs' in line:
        i = line.index('Epochs') + 2
        eseen = line[i]
monitors = int(eseen) + 1

stats_files = [results+x for x in filter(lambda x: x.startswith(prefix) and x.endswith(".txt"), next(os.walk(results))[2])]

for filename in stats_files:
    sfile = open(filename)
    head, tail = os.path.split(filename)
    newfile = open(head+os.sep+"EXTRACTED_"+tail, 'w')
    for line in sfile:
        if re.match('.*\s+{}\s+.*'.format(monitors), line) is not None:
            newfile.write(line)
    newfile.close()
    sfile.close()

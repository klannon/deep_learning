from __future__ import division, print_function
from os import fstat
import tempfile, csv
from math import pi


def which(myDict):
    rval = []
    for k, v in myDict.items():
        if v is True:
            rval.append(k)
    return rval

def progress(batch, total, batch_size, eta, end=''):
    bars = batch*100//total//4
    rval = "\t {}/{} [".format(batch*batch_size, total*batch_size)+">"*bars+"."*(25-bars)+"]"
    rval += " ETA: {:.2f}s".format(eta)
    if fstat(0) == fstat(1):
        print("\r"+rval, end=end)
    else:
        if batch == total:
            print(rval)

def convert_seconds(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    d, h, m, s = [int(t) for t in [d, h, m, s]]
    u = ['d', 'h', 'm', 's']
    rval = ''
    for i,t in enumerate([d, h, m, s]):
        if t == 0 and u[i] != 's':
            continue
        rval += "{}{} ".format(t, u[i])
    return rval.rstrip()

def cleanCharlieFile(file_path):
    with tempfile.TemporaryFile() as temp:
        with open(file_path, 'rb') as data_file:
            reader = csv.reader(data_file)
            reader.next()
            for line in reader:
                temp.write(','.join(filter(None, [x.strip() for x in [line[0],] + line[2:]])) + "\n")
        temp.seek(0)
        with open(file_path, 'w') as data_file:
            for line in temp:
                data_file.write(line)

def verify_angle(angle):
    if angle > pi:
        while angle > pi:
            angle -= 2*pi
    elif angle < -pi:
        while angle < -pi:
            angle += 2*pi
    return angle
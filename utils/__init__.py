from __future__ import division, print_function
from os import fstat


def which(myDict):
    rval = ""
    for k, v in myDict.items():
        if v is True:
            rval += k + ', '
    return rval.rstrip(', ') if rval.rstrip(', ') else None

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

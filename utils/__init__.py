from __future__ import division, print_function
from os import fstat
import tempfile, csv
import numpy as np
from math import pi, ceil


def which(myDict):
    rval = []
    for k, v in myDict.items():
        if v is True:
            rval.append(k)
    return rval

def progress(batch, total, batch_size, eta, end='', time=0):
    bars = batch*100//total//4
    rval = "\t {}/{} [".format(batch*batch_size, total*batch_size)+">"*bars+"."*(25-bars)+"]"
    rval += " ETA: {:.2f}s".format(eta) if eta else " - {:.2f}s".format(time)
    if fstat(0) == fstat(1):
        print("\r"+rval+" "*(80-len(rval)), end=end)
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

def E(indices, step=4):
    dim = len(indices)*step
    I = np.zeros((dim,dim))
    for row, col in zip(xrange(len(indices)), indices):
        mask = I[step*row:step*(row+1), step*col:step*(col+1)]
        np.fill_diagonal(mask, 1)
    return I

def gen_permutations(num_b_jets, num_jets, num_leptons):
    _leps = set(range(num_leptons))
    _bjets = set(range(num_b_jets))
    _jets = list(range(num_jets))
    for l_1 in _leps:
        l_2 = tuple(_leps - {l_1,})[0]
        for b_1 in _bjets:
            b_2 = tuple(_bjets - {b_1,})[0]
            for j1_1 in _jets:
                for j1_2 in _jets[_jets.index(j1_1)+1:]:
                    remains_5 = list(set(_jets) - {j1_1, j1_2})
                    for j2_1 in remains_5:
                        for j2_2 in remains_5[remains_5.index(j2_1)+1:]:
                            remains_3 = sorted(list(set(remains_5) - {j2_1, j2_2}))
                            yield [b_1,
                                   b_2,
                                   j1_1+2,
                                   j1_2+2,
                                   j2_1+2,
                                   j2_2+2]+map(lambda j: j+2, remains_3)+[l_1+9,
                                   l_2+9]

def permute_event(event):
    perms = list(gen_permutations(2, 7, 2))
    num_perms = len(perms)
    transforms = np.zeros((44, 44 * num_perms))
    for i, p in enumerate(perms):
        transforms[:, i * 44:(i + 1) * 44] = E(p)

    labels = np.concatenate((np.ones((num_perms,)).reshape((num_perms, 1)),
                             np.zeros((num_perms,)).reshape((num_perms, 1))), axis=1)

    labels[0] = [0, 1]
    permuted = np.dot(event, transforms).reshape((num_perms, event.size))
    arange = np.arange(num_perms)
    np.random.shuffle(arange)

    return permuted[arange], labels[arange]

def get_file_len_and_shape(fname, delim=','):
    s = set()
    with open(fname) as f:
        for i, l in enumerate(csv.reader(f, delimiter=delim)):
            if i == 0:
                cols = len(l)-1
            s.add(int(l[0]))
    return i + 1, [cols, len(s)]

def sum_cols(array, buffer=64):
    return tuple([sum(
        [array[j * buffer:(j + 1) * buffer, i].sum() for j in xrange(int(ceil(array.shape[0] / buffer)))])
           for i in xrange(array.shape[1])])
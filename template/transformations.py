from sklearn import preprocessing as pp
from numpy import concatenate

__author__ = 'Matthew Drnevich'

def standardize(datasets):   return map(pp.scale, datasets)

def get_transform(dataset):   return pp.StandardScaler().fit(dataset)

def transform(train_set, test_set):

    scale = pp.StandardScaler().fit(train_set)
    print scale
    train_set, test_set = scale.transform(train_set), scale.transform(test_set)
    return train_set, test_set

# This was being tested but determined to be detrimental. Recommended not to use, at least for the time being
def group_transform(train_set, test_set):
    pt = concatenate([train_set.X[:, i*3:i*3+1] for i in xrange(5)], axis=1)
    etaPhi = concatenate([train_set.X[:, i*3:i*3+2] for i in xrange(5)], axis=1)
    test_pt = concatenate([test_set.X[:, i*3:i*3+1] for i in xrange(5)], axis=1)
    test_etaPhi = concatenate([test_set.X[:, i*3:i*3+2] for i in xrange(5)], axis=1)

    pt_scale = get_transform(pt.reshape((pt.size, 1)))
    etaPhi_scale = get_transform(etaPhi.reshape((etaPhi.size, 1)))

    pt = pt_scale.transform(pt)
    etaPhi = etaPhi_scale.transform(etaPhi)
    test_pt = pt_scale.transform(test_pt)
    test_etaPhi = etaPhi_scale.transform(test_etaPhi)

    for i in xrange(15):
        pI = 0
        eI = 0
        if i == 0:
            train_set_X = pt[:, 0:1]
            test_set_X = test_pt[:, 0:1]
            pI += 1
        elif i > 0 and i % 3 == 0:
            train_set_X = concatenate((train_set_X, pt[:, pI:pI+1]), axis=1)
            test_set_X = concatenate((test_set_X, test_pt[:, pI:pI+1]), axis=1)
            pI += 1
        else:
            train_set_X = concatenate((train_set_X, etaPhi[:, eI:eI+1]), axis=1)
            test_set_X = concatenate((test_set_X, test_etaPhi[:, eI:eI+1]), axis=1)
            eI += 1

    train_set.X = train_set_X
    test_set.X = test_set_X
    return train_set, test_set

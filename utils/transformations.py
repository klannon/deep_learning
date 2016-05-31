from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp
import numpy as np


def standardize(datasets):   return map(pp.scale, datasets)

def get_transform(dataset):   return StandardScaler().fit(dataset)

def transform(train_set, test_set):
    scale = StandardScaler().fit(train_set)
    train_set, test_set = scale.transform(train_set), scale.transform(test_set)
    return train_set, test_set

def shuffle_in_unison(a, b):
    """ Shuffle two numpy arrays (data and labels) simultaneously.
    Code curtosy of (http://stackoverflow.com/questions/4601373/
    better-way-to-shuffle-two-numpy-arrays-in-unison).

    Parameters
    ----------
    a : numpy array
    b : (another) numpy array

    Returns
    -------
    a : shuffled version of a
    b: shuffled (the same way as a) version of b
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return (a, b)
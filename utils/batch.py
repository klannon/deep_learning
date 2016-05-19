"""
batch.py contains some utilities that will be used by trainNN.py
to construct the main network training loop
"""

import numpy as np

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

"""
This module contains functions for loading data from and writing data
to .npz files
"""
import json, math, os
import numpy as np
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr
from deep_learning.utils import verify_angle
import deep_learning.utils.misc as misc
from deep_learning.utils import gen_permutations, E

def read_config_file(dataset_name, format):
    """ Reads a json file containing the locations of train/test data
    Each dataset should contain a file (DATASET_NAME.json).  This file
    should contain a description of the different formats
    that the data is available in.  Each of these coordinate
    systems will most likely be split into two files: one file with the
    training data and another file with the testing data.  The names of
    both of these files should be included in the json file.
    
    Parameters
    ----------
    dataset_name : name of valid dataset in deep_learning/data

    format: name of the format to get the paths for

    Returns
    -------
    train_path : path to the training file listed in the json file

    test_path : path to the testing file listed in the json file
    """
    dataset_path = ds.get_path_to_dataset(dataset_name)
    json_path = os.path.join(dataset_path, ("%s.json" % dataset_name))
    json_file = open(json_path, "r")

    json_data = json.load(json_file)

    file_dict = json_data[format]

    if all([x in file_dict for x in ["train_file", "test_file"]]):
        train_file_name = file_dict["train_file"]
        test_file_name = file_dict["test_file"]
        train_path = os.path.join(dataset_path, train_file_name)
        test_path = os.path.join(dataset_path, test_file_name)
        return dict(train_path=train_path,
                    test_path=test_path)

    elif all([x in file_dict for x in ["background", "signal"]]):
        background = file_dict["background"]
        signal = file_dict["signal"]
        background_path = os.path.join(dataset_path, background)
        signal_path = os.path.join(dataset_path, signal)
        return dict(background_path=background_path,
                    signal_path=signal_path)

    elif "both" in file_dict:
        both_path = os.path.join(dataset_path, file_dict["both"])
        return dict(both=both_path)

def make_one_hot(labels):
    """ makes a one hot encoding of labels 

    Parameters
    ----------
    labels : list of dataset labels to be encoded

    Returns
    -------
    labels_one_hot : one hot encoding of labels
    """
    y_max = math.ceil(np.amax(labels))
    y_min = math.floor(np.amin(labels))
    num_classes = int((y_max - y_min) + 1)
    labels_one_hot = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        class_index = labels[i] - y_min # where in the sequence of
        # classes (from y_min to y_max) labels[i] is
        labels_one_hot[i, class_index] = 1

    return labels_one_hot

def augment(dataset, format, shift_size):
    shift_size *= (math.pi/180.0)
    num_shifts = int(2*math.pi / shift_size)
    x_train, y_train, x_test, y_test = ds.load_dataset(dataset, format)
    augmented_x = np.zeros((x_train.shape[0]*num_shifts, x_train.shape[1]))
    augmented_y = np.zeros((y_train.shape[0]*num_shifts, y_train.shape[1]))
    for ix, line in enumerate(x_train):
        if (ix+1)%1000 == 0: print ix+1
        for s in xrange(num_shifts):
            shift = s * shift_size
            augmented_x[ix*num_shifts+s] = [verify_angle(val+shift) if index%4==2 else val for index,val in enumerate(line)]
            augmented_y[ix*num_shifts+s] = y_train[ix]
    tr.shuffle_in_unison(augmented_x, augmented_y)

    output_path = os.path.join(ds.get_path_to_dataset(dataset), "augmented_{}.npz".format(format))
    np.savez(output_path, x_train=augmented_x, x_test=x_test, y_train=augmented_y, y_test=y_test)

def create_archive(dataset_name, format):
    """ converts a series of text files into a single .npz archive
    create_archive takes the name of a dataset and the
    format that the data is in, loads the config file
    from the dataset's directory, loads the right text files, and saves
    the training and testing data as numpy arrays in a single .npz
    file. The data is normalized and its variance is set to unity
    before saving, but the data will be randomized when it is loaded
    because otherwise we would be using the same "random" ordering of
    the data each time we train a network on the dataset

    Parameters
    ----------
    dataset_name : name of the dataset (directory) to look for a
    configuration file in

    format : name of the format that you want to
    build a .npz archive for

    Notes
    -----
    The locations of the text files to load should be described in
    the config file.  See the example in the OSUTTBAR dataset
    directory.
    """
    path_dict = read_config_file(dataset_name, format)
    if "train_path" in path_dict:
        train_raw = np.genfromtxt(path_dict["train_path"], delimiter=',', dtype="float32")
        test_raw = np.genfromtxt(path_dict["test_path"], delimiter=',', dtype="float32")
    elif "background_path" in path_dict:
        background = np.genfromtxt(path_dict["background_path"], delimiter=',', dtype="float32")
        signal = np.genfromtxt(path_dict["signal_path"], delimiter=',', dtype="float32")
        total = np.concatenate((background, signal), axis=0)
        np.random.shuffle(total)
        cutoff = int(total.shape[0]*0.8) # 80% training 20% testing
        train_raw = total[:cutoff, :]
        test_raw = total[cutoff:, :]
    elif "both" in path_dict:
        both = np.genfromtxt(path_dict["both"], delimiter=',', dtype="float32")
        np.random.shuffle(both)
        cutoff = int(both.shape[0] * 0.8)  # 80% training 20% testing
        train_raw = both[:cutoff, :]
        test_raw = both[cutoff:, :]


    y_train = make_one_hot(train_raw[:, 0])
    y_test = make_one_hot(test_raw[:, 0])

    x_train = train_raw[:, 1:]
    del train_raw
    x_test = test_raw[:, 1:]
    del test_raw

    output_path = os.path.join(ds.get_path_to_dataset(dataset_name), ("%s.npz" % format))
    np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

def equalize(dataset):
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)
    total_background = int(y_train[:,0].sum()+y_test[:,0].sum())
    total_signal = int(y_train[:,1].sum()+y_test[:,1].sum())

    ix = np.random.choice(total_signal, total_signal-total_background, replace=False)

    all_x_signal = np.concatenate((x_train[y_train[:,1]==1], x_test[y_test[:,1]==1]))
    all_y_signal = np.concatenate((y_train[y_train[:,1]==1], y_test[y_test[:,1]==1]))
    all_x_background = np.concatenate((x_train[y_train[:,0]==1], x_test[y_test[:,0]==1]))
    all_y_background = np.concatenate((y_train[y_train[:,0]==1], y_test[y_test[:,0]==1]))

    small_x_signal = np.delete(all_x_signal, ix, axis=0)
    small_y_signal = np.delete(all_y_signal, ix, axis=0)

    all_x = np.concatenate((all_x_background, small_x_signal))
    all_y = np.concatenate((all_y_background, small_y_signal))

    tr.shuffle_in_unison(all_x, all_y)

    cutoff = int(all_x.shape[0] * 0.8)  # 80% training 20% testing
    train_x = all_x[:cutoff]
    train_y = all_y[:cutoff]
    test_x = all_x[cutoff:]
    test_y = all_y[cutoff:]

    output_path = os.path.join(ds.get_path_to_dataset(data), "equalized_{}.npz".format(format))
    np.savez(output_path, x_train=train_x, x_test=test_x, y_train=train_y, y_test=test_y)

def save_ratios(dataset, nums=None):
    nums = [2,1,1] if not nums else nums
    data, format = dataset.split('/')
    datasets = misc.splitter(dataset, nums)
    for i, (x_train, y_train, x_test, y_test) in enumerate(datasets):
        output_path = os.path.join(ds.get_path_to_dataset(data), "{}_{}to{}.npz".format(format, nums[i], nums[-i-1]))
        np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

def save_by_jet_num(dataset, num_jets):
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)
    if num_jets.endswith("+"):
        val = lambda x: x >= int(num_jets[:-1])
    elif num_jets.endswith("-"):
        val = lambda x: x <= int(num_jets[:-1])
    else:
        val = lambda x: x == int(num_jets)
    all_x = np.concatenate((x_train, x_test), axis=0)
    all_y = np.concatenate((y_train, y_test), axis=0)
    nulls = np.zeros((all_x.shape[0], all_x.shape[1]/4), dtype=np.bool)
    for y in xrange(all_x.shape[1]/4):
        for ix, row in enumerate((x_train > 0)[:, y*4:(y+1)*4]):
            nulls[ix, y] = all(row == 0)
    events_with_x_jets = val(np.array([row[:-2].sum() for row in ~nulls]))

    all_x, all_y = all_x[events_with_x_jets], all_y[events_with_x_jets]
    tr.shuffle_in_unison(all_x, all_y)

    cutoff = int(all_x.shape[0] * 0.8)  # 80% training 20% testing
    train_x = all_x[:cutoff]
    train_y = all_y[:cutoff]
    test_x = all_x[cutoff:]
    test_y = all_y[cutoff:]

    output_path = os.path.join(ds.get_path_to_dataset(data), "{}jets_{}.npz".format(num_jets, format))
    np.savez(output_path, x_train=train_x, x_test=test_x, y_train=train_y, y_test=test_y)

def permutate_sorted(dataset):
    """ Only use this for sorted data! Also, this takes up a significant amount of RAM """
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)

    # Generate permutations, transforms, and alter the dataset
    perms = list(gen_permutations(2, 7, 2))
    num_perms = len(perms)
    transforms = np.zeros((44, 44 * num_perms))
    for i, p in enumerate(perms):
        transforms[:, i * 44:(i + 1) * 44] = E(p)

    # For the training data
    sorted_train_x = np.zeros((x_train.shape[0] * num_perms, x_train.shape[1]))
    sorted_train_y = np.zeros((sorted_train_x.shape[0], 2))

    for i, batch in enumerate(x_train):
        labels = np.concatenate((np.ones((num_perms,)).reshape((num_perms, 1)),
                                 np.zeros((num_perms,)).reshape((num_perms, 1))), axis=1)
        labels[0] = [0, 1]
        event = np.dot(batch, transforms).reshape((num_perms, x_train.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_train_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_train_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    # For the testing data
    sorted_test_x = np.zeros((x_test.shape[0] * num_perms, x_test.shape[1]))
    sorted_test_y = np.zeros((sorted_test_x.shape[0], 2))

    for i, batch in enumerate(x_test):
        labels = np.concatenate((np.ones((num_perms,)).reshape((num_perms, 1)),
                                 np.zeros((num_perms,)).reshape((num_perms, 1))), axis=1)
        labels[0] = [0, 1]
        event = np.dot(batch, transforms).reshape((num_perms, x_test.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_test_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_test_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    output_path = os.path.join(ds.get_path_to_dataset(data), "{}_{}.npz".format(format, "Permutated"))
    np.savez(output_path, x_train=sorted_train_x, x_test=sorted_test_x, y_train=sorted_train_y, y_test=sorted_test_y)

if __name__ == "__main__":
    #save_by_jet_num('ttHLep/Unsorted', "5-")
    #save_ratios("ttHLep/5-jets_Unsorted", [1,])
    pass
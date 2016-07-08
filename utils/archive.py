"""
This module contains functions for loading data from and writing data
to .npz files
"""
from __future__ import division
import json, math, os
import numpy as np
import tables
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr
from deep_learning.utils import verify_angle, get_file_len_and_shape
import deep_learning.utils.misc as misc

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

def make_one_hot(labels, y_min=None, y_max=None):
    """ makes a one hot encoding of labels 

    Parameters
    ----------
    labels : list of dataset labels to be encoded

    Returns
    -------
    labels_one_hot : one hot encoding of labels
    """
    if type(labels) is np.float32:
        nrows = 1
        labels = [labels]
    else:
        nrows = labels.shape[0]
    y_max = y_max if type(y_max) is int else math.ceil(np.amax(labels))
    y_min = y_min if type(y_min) is int else math.floor(np.amin(labels))
    num_classes = int((y_max - y_min) + 1)
    labels_one_hot = np.zeros((nrows, num_classes))
    for i in xrange(nrows):
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

def create_archive(dataset_name, format, buffer=1000, train_fraction=0.8):
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
    output_path = os.path.join(ds.get_path_to_dataset(dataset_name), "{}.hdf5".format(dataset_name))

    if "train_path" in path_dict:
        train_len, train_cols = get_file_len_and_shape(path_dict["train_path"])
        test_len, test_cols = get_file_len_and_shape(path_dict["test_path"])
        assert train_cols == test_cols  # Train and test files should have the same data shape
        h_file, h_data = add_group_hdf5(output_path, format, zip([train_len]*2+[test_len]*2, train_cols+test_cols))
        with open(path_dict["train_path"]) as train_f:
            for i, l in enumerate(train_f):
                event = np.fromstring(l, sep=',', dtype="float32")
                h_data[0][i] = event[1:]
                h_data[1][i] = make_one_hot(event[0], 0, train_cols[1]-1)[0]
        with open(path_dict["test_path"]) as test_f:
            for i, l in enumerate(test_f):
                event = np.fromstring(l, sep=',', dtype="float32")
                h_data[2][i] = event[1:]
                h_data[3][i] = make_one_hot(event[0], 0, test_cols[1]-1)[0]
        #train_raw = np.genfromtxt(path_dict["train_path"], delimiter=',', dtype="float32")
        #test_raw = np.genfromtxt(path_dict["test_path"], delimiter=',', dtype="float32")
    elif "background_path" in path_dict:
        bkg_len, bkg_cols = get_file_len_and_shape(path_dict["background_path"])
        sig_len, sig_cols = get_file_len_and_shape(path_dict["signal_path"])
        n_labels = 2

        total_len = bkg_len+sig_len
        bkg_read_amt = int(bkg_len*buffer/total_len)
        sig_read_amt = int(sig_len*buffer/total_len)
        assert bkg_cols == sig_cols # Bkg and sig files should have the same data shape
        h_file, h_data = add_group_hdf5(output_path,
                                        format,
                                        zip([round(total_len*train_fraction)] * 2 + [round(total_len*(1-train_fraction))] * 2,
                                            [bkg_cols[0], n_labels]*2))
        # Read in a buffer of 1000 lines at a time (allows fraction accuracy to 0.x5)
        with open(path_dict["background_path"]) as bkg_f:
            with open(path_dict["signal_path"]) as sig_f:
                i = 0
                while i < total_len - 1:
                    x_buffer_array = np.zeros((buffer, bkg_cols[0]))
                    y_buffer_array = np.zeros((buffer, n_labels))
                    ix = 0
                    for j in xrange(bkg_read_amt):
                        line = bkg_f.readline()
                        if line:
                            event = np.fromstring(line, sep=',', dtype="float32")
                            x_buffer_array[ix] = event[1:]
                            y_buffer_array[ix] = make_one_hot(event[0], 0, n_labels - 1)[0]
                            ix += 1
                        else:
                            break
                    for k in xrange(sig_read_amt):
                        line = sig_f.readline()
                        if line:
                            event = np.fromstring(line, sep=',', dtype="float32")
                            x_buffer_array[ix] = event[1:]
                            y_buffer_array[ix] = make_one_hot(event[0], 0, n_labels - 1)[0]
                            ix += 1
                        else:
                            break
                    indices = np.any(~(x_buffer_array==0), axis=1)
                    x_buffer_array = x_buffer_array[indices]
                    y_buffer_array = y_buffer_array[indices]
                    tr.shuffle_in_unison(x_buffer_array, y_buffer_array)
                    cutoff = int(x_buffer_array.shape[0]*train_fraction)
                    for r in x_buffer_array[:cutoff]:
                        h_data[0].append(r[None])
                    for r in y_buffer_array[:cutoff]:
                        h_data[1].append(r[None])
                    for r in x_buffer_array[cutoff:]:
                        h_data[2].append(r[None])
                    for r in y_buffer_array[cutoff:]:
                        h_data[3].append(r[None])
                    i += ix
    elif "both" in path_dict:
        both = np.genfromtxt(path_dict["both"], delimiter=',', dtype="float32")
        np.random.shuffle(both)
        cutoff = int(both.shape[0] * 0.8)  # 80% training 20% testing
        train_raw = both[:cutoff, :]
        test_raw = both[cutoff:, :]
    h_file.flush()
    h_file.close()

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

def add_group_hdf5(save_path, group, expected_shapes):
    hdf5_file = tables.open_file(save_path, mode='a')
    h_comp = tables.Filters(complevel=5, complib='blosc')
    h_group = hdf5_file.create_group("/", group, group)
    h_data = []
    for k, shape in zip(["x_train", "y_train", "x_test", "y_test"], expected_shapes):
        h_data.append(hdf5_file.create_earray(h_group, k,
                                              tables.Float32Atom(),
                                              shape=(0, shape[1]),
                                              filters=h_comp,
                                              expectedrows=shape[0]))
    return hdf5_file, h_data


if __name__ == "__main__":
    #save_by_jet_num('ttHLep/Unsorted', "5-")
    #save_ratios("ttHLep/5-jets_Unsorted", [1,])
    pass
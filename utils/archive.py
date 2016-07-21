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
from deep_learning.utils import verify_angle, get_file_len_and_shape, sum_cols
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

# HDF5
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

# HDF5
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
            for l in train_f:
                event = np.fromstring(l, sep=',', dtype="float32")
                h_data[0].append(event[1:][None])
                h_data[1].append(make_one_hot(event[0], 0, train_cols[1]-1)[0])
        with open(path_dict["test_path"]) as test_f:
            for l in test_f:
                event = np.fromstring(l, sep=',', dtype="float32")
                h_data[2].append(event[1:][None])
                h_data[3].append(make_one_hot(event[0], 0, test_cols[1]-1)[0])

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
        # Read in a buffer of 1000 lines at a time (allows fraction accuracy to 0.xxx)
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

        total_len, total_cols = get_file_len_and_shape(path_dict["both"])
        train_len = round(train_fraction*total_len)
        test_len = round((1-train_fraction)*total_len)
        h_file, h_data = add_group_hdf5(output_path, format,
                                        zip([train_len] * 2 + [test_len] * 2, total_cols*2))
        with open(path_dict["both"]) as data_f:
            x_buffer_array = np.zeros((buffer, total_cols[0]))
            y_buffer_array = np.zeros((buffer, total_cols[1]))
            for i, l in enumerate(data_f):
                event = np.fromstring(l, sep=',', dtype="float32")
                x_buffer_array[i%buffer] = event[1:]
                y_buffer_array[i%buffer] = make_one_hot(event[0], 0, total_cols[1] - 1)[0]
                if i%buffer == buffer - 1:
                    indices = np.any(~(x_buffer_array == 0), axis=1)
                    x_buffer_array = x_buffer_array[indices]
                    y_buffer_array = y_buffer_array[indices]
                    tr.shuffle_in_unison(x_buffer_array, y_buffer_array)
                    cutoff = int(x_buffer_array.shape[0] * train_fraction)
                    for r in x_buffer_array[:cutoff]:
                        h_data[0].append(r[None])
                    for r in y_buffer_array[:cutoff]:
                        h_data[1].append(r[None])
                    for r in x_buffer_array[cutoff:]:
                        h_data[2].append(r[None])
                    for r in y_buffer_array[cutoff:]:
                        h_data[3].append(r[None])
                    x_buffer_array = np.zeros((buffer, total_cols[0]))
                    y_buffer_array = np.zeros((buffer, total_cols[1]))

    h_file.flush()
    h_file.close()

def save_ratios(dataset, ratios, buffer=1000):

    ratios = [ratios] if type(ratios) is str else ratios
    ratios = map(lambda x: map(float, x.split(':')), ratios)
    data = dataset.split('/')[0]
    format = '/'.join(dataset.split('/')[1:])
    main_file, (x_train, y_train, x_test, y_test) = ds.load_dataset(data, format, mode='a')

    bkg_test, sig_test = sum_cols(y_test)
    bkg_train, sig_train = sum_cols(y_train)

    TEST_UPPER_LIMIT = int(1.5 * bkg_test) if bkg_test < sig_test else int(1.5 * sig_test)
    TRAIN_UPPER_LIMIT = int(1.5 * bkg_train) if bkg_train < sig_train else int(1.5 * sig_train)

    temp_h_file, temp_h_data = add_group_hdf5(".deep_learning.temp.hdf5", "Temp",
                                    [(bkg_train, x_train.shape[1]),
                                     (bkg_train, y_train.shape[1]),
                                     (sig_train, x_train.shape[1]),
                                     (sig_train, y_train.shape[1]),
                                     (bkg_test, x_test.shape[1]),
                                     (bkg_test, y_test.shape[1]),
                                     (sig_test, x_test.shape[1]),
                                     (sig_test, y_test.shape[1])],
                                    names=["train_bkg_x",
                                           "train_bkg_y",
                                           "train_sig_x",
                                           "train_sig_y",
                                           "test_bkg_x",
                                           "test_bkg_y",
                                           "test_sig_x",
                                           "test_sig_y"])

    print "Generating temporary files..."
    for i in xrange(int(math.ceil(x_train.shape[0] / buffer))):
        # index should be same shape and need to reshape the result :/
        train_bkg_index = np.array([[False]*x_train.shape[1]]*x_train.shape[0])
        train_sig_index = np.array([[False]*x_train.shape[1]]*x_train.shape[0])
        test_bkg_index = np.array([[False]*x_test.shape[1]]*x_test.shape[0])
        test_sig_index = np.array([[False]*x_test.shape[1]]*x_test.shape[0])

        for j in xrange(x_train.shape[1]):
            train_bkg_index[i * buffer:(i + 1) * buffer, j] = y_train[i * buffer:(i + 1) * buffer, 0] == 1
            train_sig_index[i * buffer:(i + 1) * buffer, j] = y_train[i * buffer:(i + 1) * buffer, 1] == 1
        for j in xrange(x_test.shape[1]):
            test_bkg_index[i * buffer:(i + 1) * buffer, j] = y_test[i * buffer:(i + 1) * buffer, 0] == 1
            test_sig_index[i * buffer:(i + 1) * buffer, j] = y_test[i * buffer:(i + 1) * buffer, 1] == 1

        selection = x_train[train_bkg_index]
        temp_h_data[0].append(selection.reshape((selection.size/x_train.shape[1], x_train.shape[1])))

        selection = y_train[train_bkg_index[:, :y_train.shape[1]]]
        temp_h_data[1].append(selection.reshape((selection.size/y_train.shape[1], y_train.shape[1])))

        selection = x_train[train_sig_index]
        temp_h_data[2].append(selection.reshape((selection.size/x_train.shape[1], x_train.shape[1])))

        selection = y_train[train_sig_index[:, :y_train.shape[1]]]
        temp_h_data[3].append(selection.reshape((selection.size/y_train.shape[1], y_train.shape[1])))

        selection = x_test[test_bkg_index]
        temp_h_data[4].append(selection.reshape((selection.size/x_test.shape[1], x_test.shape[1])))

        selection = y_test[test_bkg_index[:, :y_test.shape[1]]]
        temp_h_data[5].append(selection.reshape((selection.size/y_test.shape[1], y_test.shape[1])))

        selection = x_test[test_sig_index]
        temp_h_data[6].append(selection.reshape((selection.size/x_test.shape[1], x_test.shape[1])))

        selection = y_test[test_sig_index[:, :y_test.shape[1]]]
        temp_h_data[7].append(selection.reshape((selection.size/y_test.shape[1], y_test.shape[1])))

    # Perform all of this in archive so that you write to file every iteration
    buffer_reset = buffer
    for rat in ratios:

        print "Creating ratio {:d}/{:d} ...".format(*map(int, rat))

        h_file, h_data = add_group_hdf5(ds.get_path_to_dataset(data)+os.sep+data+".hdf5",
                                        "{}to{}".format(*map(int, rat)),
                                        [(TRAIN_UPPER_LIMIT, x_train.shape[1]),
                                         (TRAIN_UPPER_LIMIT, y_train.shape[1]),
                                         (TEST_UPPER_LIMIT, x_test.shape[1]),
                                         (TEST_UPPER_LIMIT, y_test.shape[1])],
                                        where='/{}'.format(format))

        test_bkg_indices = np.arange(bkg_test)
        test_sig_indices = np.arange(sig_test)
        train_bkg_indices = np.arange(bkg_train)
        train_sig_indices = np.arange(sig_train)

        train_count = 0
        buffer = buffer_reset
        while train_count < TRAIN_UPPER_LIMIT:
            if TRAIN_UPPER_LIMIT - train_count < buffer:
                buffer = TRAIN_UPPER_LIMIT - train_count

            # Indices to NOT include
            train_bkg_ix = np.random.choice(train_bkg_indices,
                                            train_bkg_indices.size - (rat[0] * buffer / sum(rat)),
                                            replace=False)
            train_sig_ix = np.random.choice(train_sig_indices,
                                            train_sig_indices.size - (rat[1] * buffer / sum(rat)),
                                            replace=False)

            # Indices to keep
            k_train_bkg = np.setdiff1d(train_bkg_indices, train_bkg_ix)
            k_train_sig = np.setdiff1d(train_sig_indices, train_sig_ix)

            train_small_x_sig = temp_h_data[2][k_train_sig]
            train_small_y_sig = temp_h_data[3][k_train_sig]
            train_small_x_bkg = temp_h_data[0][k_train_bkg]
            train_small_y_bkg = temp_h_data[1][k_train_bkg]

            train_x = np.concatenate((train_small_x_bkg, train_small_x_sig))
            train_y = np.concatenate((train_small_y_bkg, train_small_y_sig))

            tr.shuffle_in_unison(train_x, train_y)

            h_data[0].append(train_x)
            h_data[1].append(train_y)

            train_count += k_train_bkg.size + k_train_sig.size

            train_bkg_indices = train_bkg_ix
            train_sig_indices = train_sig_ix

        test_count = 0
        buffer = buffer_reset
        while test_count < TEST_UPPER_LIMIT:
            if TEST_UPPER_LIMIT - test_count < buffer:
                buffer = TEST_UPPER_LIMIT - test_count

            # Indices to NOT include
            test_bkg_ix = np.random.choice(test_bkg_indices, test_bkg_indices.size - (rat[0] * buffer / sum(rat)),
                                           replace=False)
            test_sig_ix = np.random.choice(test_sig_indices, test_sig_indices.size - (rat[1] * buffer / sum(rat)),
                                           replace=False)

            # Indices to keep
            k_test_bkg = np.setdiff1d(test_bkg_indices, test_bkg_ix)
            k_test_sig = np.setdiff1d(test_sig_indices, test_sig_ix)

            test_small_x_sig = temp_h_data[6][k_test_sig]
            test_small_y_sig = temp_h_data[7][k_test_sig]
            test_small_x_bkg = temp_h_data[4][k_test_bkg]
            test_small_y_bkg = temp_h_data[5][k_test_bkg]

            test_x = np.concatenate((test_small_x_bkg, test_small_x_sig))
            test_y = np.concatenate((test_small_y_bkg, test_small_y_sig))

            tr.shuffle_in_unison(test_x, test_y)

            h_data[2].append(test_x)
            h_data[3].append(test_y)

            test_count += k_test_bkg.size + k_test_sig.size

            test_bkg_indices = test_bkg_ix
            test_sig_indices = test_sig_ix

        print "Created Group: {}/{}to{}".format(format, *map(int, rat))

        h_file.flush()
        h_file.close()

    main_file.close()
    temp_h_file.close()
    os.remove(".deep_learning.temp.hdf5")

# DEPRECATED, WON'T WORK
def _save_by_jet_num(dataset, num_jets):
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

# NO HDF5
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

    labels = np.concatenate((np.ones((num_perms,)).reshape((num_perms, 1)),
                             np.zeros((num_perms,)).reshape((num_perms, 1))), axis=1)

    labels[0] = [0, 1]

    for i, batch in enumerate(x_train):
        event = np.dot(batch, transforms).reshape((num_perms, x_train.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_train_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_train_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    # For the testing data
    sorted_test_x = np.zeros((x_test.shape[0] * num_perms, x_test.shape[1]))
    sorted_test_y = np.zeros((sorted_test_x.shape[0], 2))

    for i, batch in enumerate(x_test):
        event = np.dot(batch, transforms).reshape((num_perms, x_test.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_test_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_test_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    output_path = os.path.join(ds.get_path_to_dataset(data), "{}_{}.npz".format(format, "Permuted"))
    np.savez(output_path, x_train=sorted_train_x, x_test=sorted_test_x, y_train=sorted_train_y, y_test=sorted_test_y)

# NOT FINALIZED - NO HDF5
def permutate_individual_sorted(dataset):
    """ Only use this for sorted data! Also, this takes up a significant amount of RAM """
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)

    # Generate permutations, transforms, and alter the dataset
    perms = list(gen_permutations(2, 7, 2))
    num_perms = len(perms)

    aperms = np.array(perms)
    labels = np.zeros(aperms.shape)
    r = np.arange(11)
    for i,p in enumerate(aperms):
        labels[i] = (p == r).astype('int32')

    transforms = np.zeros((44, 44 * num_perms))
    for i, p in enumerate(perms):
        transforms[:, i * 44:(i + 1) * 44] = E(p)

    # For the training data
    sorted_train_x = np.zeros((x_train.shape[0] * num_perms, x_train.shape[1]))
    sorted_train_y = np.zeros((sorted_train_x.shape[0], 2))

    for i, batch in enumerate(x_train):
        event = np.dot(batch, transforms).reshape((num_perms, x_train.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_train_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_train_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    # For the testing data
    sorted_test_x = np.zeros((x_test.shape[0] * num_perms, x_test.shape[1]))
    sorted_test_y = np.zeros((sorted_test_x.shape[0], 2))

    for i, batch in enumerate(x_test):
        event = np.dot(batch, transforms).reshape((num_perms, x_test.shape[1]))
        arange = np.arange(num_perms)
        np.random.shuffle(arange)
        sorted_test_x[i * num_perms:(i + 1) * num_perms] = event[arange]
        sorted_test_y[i * num_perms:(i + 1) * num_perms] = labels[arange]

    output_path = os.path.join(ds.get_path_to_dataset(data), "{}_{}.npz".format(format, "Permuted"))
    np.savez(output_path, x_train=sorted_train_x, x_test=sorted_test_x, y_train=sorted_train_y, y_test=sorted_test_y)

def add_group_hdf5(save_path, group, expected_shapes, where='/', names=None):
    names = names if names else ["x_train", "y_train", "x_test", "y_test"]
    hdf5_file = tables.open_file(save_path, mode='a')
    h_comp = tables.Filters(complevel=5, complib='blosc')
    h_group = hdf5_file.create_group(where, group, group)
    h_data = []
    for k, shape in zip(names, expected_shapes):
        h_data.append(hdf5_file.create_earray(h_group, k,
                                              tables.Float32Atom(),
                                              shape=(0, shape[1]),
                                              filters=h_comp,
                                              expectedrows=shape[0]))
    return hdf5_file, h_data

def remove_group_hdf5(save_path, group, where='/', recursive=True):
    hdf5_file = tables.open_file(save_path, mode='a')
    hdf5_file.remove_node(where+group, recursive=recursive)

def add_transformed(save_path, group, buffer=1000, where='/'):
    hdf5_file = tables.open_file(save_path, mode='a')
    parent = hdf5_file.get_node(where+group)
    data = (parent.x_train, parent.y_train, parent.x_test, parent.y_test)
    shapes = map(lambda x: x.shape, data)
    h_comp = tables.Filters(complevel=5, complib='blosc')
    h_group = hdf5_file.create_group(where+group, 'transformed', 'Data scaled to Gaussian distribution')
    h_data = []
    for k, shape in zip(["x_train", "y_train", "x_test", "y_test"], shapes):
        h_data.append(hdf5_file.create_carray(h_group, k,
                                              tables.Float32Atom(),
                                              shape=shape,
                                              filters=h_comp))
    scale = tr.get_transform(data[0])
    for i, d in enumerate(data):
        for j in xrange(int(math.ceil(d.shape[0] / buffer))):
            if i%2 == 1:
                h_data[i][j*buffer:(j+1)*buffer] = d[j*buffer:(j+1)*buffer]
            else:
                h_data[i][j * buffer:(j + 1) * buffer] = scale.transform(d[j * buffer:(j + 1) * buffer])
    hdf5_file.flush()
    hdf5_file.close()
"""
This module contains functions for loading data from and writing data
to .npz files
"""
import json, math, os, csv, tempfile
import numpy as np
import deep_learning.utils.transformations as tr
import deep_learning.utils.dataset as ds

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
    num_classes = (y_max - y_min) + 1
    labels_one_hot = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        class_index = labels[i] - y_min # where in the sequence of
        # classes (from y_min to y_max) labels[i] is
        labels_one_hot[i, class_index] = 1

    return labels_one_hot

def take_out_event_num(file_path):
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
        train_raw = np.genfromtxt(path_dict["train_path"], delimiter=',')
        test_raw = np.genfromtxt(path_dict["test_path"], delimiter=',')
    elif "background_path" in path_dict:
        take_out_event_num(path_dict["background_path"])
        take_out_event_num(path_dict["signal_path"])
        background = np.genfromtxt(path_dict["background_path"], delimiter=',')
        signal = np.genfromtxt(path_dict["signal_path"], delimiter=',')
        total = np.concatenate((background, signal), axis=0)
        np.random.shuffle(total)
        cutoff = int(total.shape[0]*0.8) # 80% training 20% testing
        train_raw = total[:cutoff, :]
        test_raw = total[cutoff:, :]


    y_train = make_one_hot(train_raw[:, 0])
    y_test = make_one_hot(test_raw[:, 0])

    x_train = train_raw[:, 1:]
    del train_raw
    x_test = test_raw[:, 1:]
    del test_raw
    
    # transform all rows, excluding the labels
    (x_train, x_test) = tr.transform(x_train, x_test)
    output_path = os.path.join(ds.get_path_to_dataset(dataset_name), ("%s.npz" % format))
    np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


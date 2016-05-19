"""
This module contains functions for loading data from and writing data
to .npz files


"""
import argparse
import json
import math
import os
import sys
import unittest

import numpy as np

from deep_learning.utils.transformations import transform

##############
# Unit Tests #
##############

class LoaderTestCase(unittest.TestCase):
    def test_get_available_datasets(self):
        """ Test the get_available_datasets function
        Because the datasets in the folder will change over time, this
        test only prints out the results of get_available_datasets().
        The onus is on the user to check that all the datasets in
        deep_learning/data are contained in the list printed by this
        test
        """
        print get_available_datasets()

    def test_verify_dataset(self):
        self.assertEqual(verify_dataset("OSU_TTBAR"), None)
        with self.assertRaises(IOError):
            verify_dataset("there's no way this one exists")
            
    def test_load_dataset(self):
        with self.assertRaises(IOError):
            load_dataset("this one definitely doesn't exist either",
                         "")

    def test_read_config_file(self):
        # nominal case
        read_config_file("OSU_TTBAR", "PtEtaPhi")
        
        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            read_config_file("no way does this exist", "PtEtaPhi")

        # if the coordinate system doesn't exist in the given file
        with self.assertRaises(KeyError):
            read_config_file("OSU_TTBAR", "does not exist")

    def test_write_data_to_archive(self):
        write_data_to_archive("OSU_TTBAR", "PtEtaPhi")

####################
# Module functions #
####################

def get_available_datasets():
    """ Gets a list of the directories in deep_learning/data
    Each one of these directories is assumed to represent a dataset,
    containing a .npz file with training and testing data

    Returns
    -------
    datasets : list of the folders in deep_learning/data/
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Go up one directory and into the "data" directory
    data_dir = os.path.join(current_dir, "..", "data")
    data_dir_contents = os.listdir(data_dir)
    
    # filter out any files that might be in deep_learning/data/
    # beacuse a dataset is assumed to be any folder in this directory
    datasets = []
    for item in data_dir_contents:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)

    return datasets

def verify_dataset(dataset_name):
    """ checks if dataset_name exists. Raises an IOError if it doesn't

    Parameters
    ----------
    dataset_name : name of the dataset to check

    Raises
    ------
    IOError : if the dataset dataset_name doesn't exist

    Returns
    -------
    None : if the dataset dataset_name does exist
    """
    if dataset_name not in get_available_datasets():
        raise IOError("The dataset \"%s\" does not exist.  You can "
                      "create it by visiting deep_learning/data and "
                      "creating the directory \"%s\" there." %
                      (dataset_name, dataset_name))

def get_path_to_dataset(dataset_name, coordinate_system=None):
    """ returns the absolute location of the directory dataset_name
    If the dataset (folder) does not exist, raises an IOError, and if
    coordinate_system.npz does not exist in the directory, an IOError
    is also raised.  If no coordinate system is specified, returns the
    path to the dataset.  If a coordinate system is specified, returns
    the path to the coordinate system within the dataset

    Parameters
    ----------
    dataset_name : dataset to find the path to
    If the dataset name is not valid, an IOError will be raised

    coordinate_system : (default None) If a coordinate system is
    specified, get_path_to_dataset will return the path to the .npz
    file for this coordinate system.  A NameError will be raised if the
    requested coordinate system does not exist.

    Returns
    -------
    if coordinate_system != None : the path to the coordinate system's
    .npz file

    else : the path to the dataset
    
    """
    # check if the dataset exists
    verify_dataset(dataset_name)
    
    # because this file is in the utils directory
    util_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Go up one directory, into the "data" directory, and into the
    # directory dataset_name
    data_dir = os.path.join(util_dir, "..", "data")
    dataset_dir = os.path.join(data_dir, dataset_name)

    # if no coordinate system is specified, return the path to the
    # dataset directory
    if coordinate_system == None:
        return dataset_dir
    # figure out whether the files for a given coordinate system exist
    else:
        dataset_path = os.path.join(dataset_dir,
                                    (coordinate_system + ".npz"))

    if not os.path.isfile(dataset_path):
        raise IOError("The file \"%s\" does not exist" %
                      dataset_path)
    
    else: # if a file exists for the specified coordinate system
        return dataset_path

def load_file(dataset_name, ):
    # reads data 
    dataset_path = get_path_to_dataset(dataset_name)


    ##########################


    ##### TODO: CHANGE THIS AND GET_PATH_TO_DATASET SO IT READS BINARY .NPZ FILES ##########################


    ##########################


    
    x_train = train[:,1:]
    y_train = train[:, 0]

    # makes a one hot encoding of y_train
    y_max = math.ceil(np.amax(y_train))
    y_min = math.floor(np.amin(y_train))
    num_classes = (y_max - y_min) + 1
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    for i in range(y_train.shape[0]):
        class_index = y_train[i] - y_min # where in the sequence of
        # classes (from y_min to y_max) y_train[i] is
        y_train_one_hot[i, class_index] = 1

    return(x_train, y_train_one_hot)

# def load_dataset(dataset_name, coordinate_system):
#     x_test = test[:,1:]
#     print x_test[0:2]
#     y_test_raw = test[:, 0]
#     y_test_list = []
#     # [ 1.  2.]
#     for row in y_test_raw:
#         if row == 1:
#             y_test_list.append([0., 1.])
            
#         if row == 2:
#             y_test_list.append([1., 0.])
#             y_test = np.array(y_test_list)
            
#     print y_test[0:2]

#     return(x_test, y_test)

def load_dataset(dataset_name, coordinate_system):
    dataset_path = get_path_to_dataset(dataset_name, coordinate_system)
    data = np.load(dataset_path)
    return(data['x_train'], data['y_train'], data['x_test'], data['y_test'])

def read_config_file(dataset_name, coordinate_system):
    """ Reads a json file containing the locations of train/test data
    Each dataset should contain a file (DATASET_NAME.json).  This file
    should contain a description of the different coordinate systems
    that the data is available in.  Each of these coordinate
    systems will most likely be split into two files: one file with the
    training data and another file with the testing data.  The names of
    both of these files should be included in the json file.
    
    Parameters
    ----------
    dataset_name : name of valid dataset in deep_learning/data

    coordinate_system: name of the coordinate system to get the paths for

    Returns
    -------
    train_path : path to the training file listed in the json file

    test_path : path to the testing file listed in the json file
    """
    dataset_path = get_path_to_dataset(dataset_name)
    json_path = os.path.join(dataset_path, ("%s.json" % dataset_name))
    json_file = open(json_path, "r")

    json_data = json.load(json_file)

    # print json_data[coordinate_system]
    
    train_file_name = json_data[coordinate_system]["train_file"]
    test_file_name = json_data[coordinate_system]["test_file"]

    train_path = os.path.join(dataset_path, train_file_name)
    test_path = os.path.join(dataset_path, test_file_name)
    
    return (train_path, test_path)

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

def write_data_to_archive(dataset_name, coordinate_system):
    """ converts a series of text files into a single .npz archive
    write_data_to_archive takes the name of a dataset and the
    coordinate system that the data is in, loads the config file
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

    coordinate_system : name of the coordinate system that you want to
    build a .npz archive for

    Notes
    -----
    The locations of the text files to load should be described in
    the config file.  See the example in the OSUTTBAR dataset
    directory.
    """
    (train_path, test_path) = read_config_file(dataset_name,
                                               coordinate_system)
    train_raw = np.genfromtxt(train_path, delimiter=',')
    test_raw = np.genfromtxt(test_path, delimiter=',')
    # print train_raw
    
    y_train = make_one_hot(train_raw[:, 0])
    # print y_train
    y_test = make_one_hot(test_raw[:, 0])

    x_train = train_raw[:, 1:]
    del train_raw
    x_test = test_raw[:, 1:]
    del test_raw
    
    # transform all rows, excluding the labels
    (x_train, x_test) = transform(x_train, x_test)
    output_path = os.path.join(get_path_to_dataset(dataset_name),("%s.npz" % coordinate_system))
    np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    # uncomment the next 2 lines to run the tests
    # suite = unittest.TestLoader().loadTestsFromTestCase(LoaderTestCase)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) != 3:
        print "Usage: python read_write.py DATASET COORDINATE_SYSTEM"
        exit(1)

    write_data_to_archive(sys.argv[1], sys.argv[2])

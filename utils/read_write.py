"""
This module contains functions for loading data from and writing data
to .npz files
"""
import argparse
import json
import math
import os
import shutil
import sys
import unittest

import numpy as np

import deep_learning.utils.transformations as tr
import deep_learning.utils.dataset as ds



################
## Unit Tests ##
################

class ReadWriteTestCase(unittest.TestCase):
    def setUp(self):
        # make a test dataset
        # print os.getcwd()
        data_dir_path = ds.get_data_dir_path()
        # print os.getcwd()
        os.chdir(r"%s" % data_dir_path)
        try:
            os.mkdir("TEST")
        except OSError:
            shutil.rmtree("TEST")
            os.mkdir("TEST")
            
        os.chdir("TEST")

        # make some files in the test dataset
        f = open("TEST.json", "w")
        f.write("{\"PtEtaPhi\":{\"train_file\":\"train_all_ptEtaPhi"
                ".txt\",\"test_file\":\"test_all_ptEtaPhi.txt\"},\""
                "Pt\":{\"train_file\":\"train_all_Pt.txt\",\"test_f"
                "ile\":\"test_all_Pt.txt\"},\"3v\":{\"train_file\":"
                "\"train_all_3v.txt\",\"test_file\":\"test_all_3v.t"
                "xt\"}}")
        f.close()
        os.system("touch train_all_ptEtaPhi.txt")
        os.system("touch test_all_ptEtaPhi.txt")
        os.system("touch train_all_Pt.txt")
        os.system("touch test_all_Pt.txt")
        os.system("touch train_all_3v.txt")
        os.system("touch test_all_3v.txt")

        os.system("touch PtEtaPhi.npz")
        
        os.chdir(data_dir_path)     

    
    def test_load_dataset(self):
        """ Tests the load_data function """

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            load_dataset("this one definitely doesn't exist either",
                         "")

    def test_read_config_file(self):
        """ Tests the read_config_file function """
        # nominal case
        train, test = read_config_file("TEST", "PtEtaPhi")
        test_dataset_path = ds.get_path_to_dataset("TEST")
        self.assertEqual(train, os.path.join(test_dataset_path,
                                             "train_all_ptEtaPhi.txt"))
        self.assertEqual(test, os.path.join(test_dataset_path,
                                            "test_all_ptEtaPhi.txt"))
        
        # if the dataset doesn't exist but the coordinate system does
        with self.assertRaises(IOError):
            read_config_file("no way does this dataset exist",
                             "PtEtaPhi")

        # if the coordinate system doesn't exist in a valid dataset
        with self.assertRaises(KeyError):
            read_config_file("TEST", "Spherical")

    def tearDown(self):
        data_dir_path = ds.get_data_dir_path()
        shutil.rmtree("TEST")


######################
## Module functions ##
######################

def load_dataset(dataset_name, coordinate_system):
    dataset_path = ds.get_path_to_dataset(dataset_name, coordinate_system)
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
    dataset_path = ds.get_path_to_dataset(dataset_name)
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
    (x_train, x_test) = tr.transform(x_train, x_test)
    output_path = os.path.join(ds.get_path_to_dataset(dataset_name),("%s.npz" % coordinate_system))
    np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)




    
if __name__ == "__main__":
    # uncomment the next 2 lines to run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(ReadWriteTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # if len(sys.argv) != 3:
    #     print "Usage: python read_write.py DATASET COORDINATE_SYSTEM"
    #     exit(1)

    # write_data_to_archive(sys.argv[1], sys.argv[2])

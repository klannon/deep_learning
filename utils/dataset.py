import os
import tables

module_dir = os.path.dirname(os.path.realpath(__file__))

def get_data_dir_path():
    """ Gets the absolute path to the data directory

    Returns
    -------
    data_dir_path <string> : absolute path to the data directory
    """
    data_dir_path = os.path.join(module_dir, "..", "data")
    data_dir_path = os.path.realpath(data_dir_path)
    return data_dir_path


def get_available_datasets():
    """ Gets a list of the directories in deep_learning/data
    Each one of these directories is assumed to represent a dataset,
    containing a .json data configuration file.

    Returns
    -------
    datasets <list> : list of the folders in deep_learning/data/
    """

    # list contends of data directory
    data_dir = get_data_dir_path()
    data_dir_contents = os.listdir(data_dir)

    # filter out any files that might be in deep_learning/data/
    # beacuse a dataset is assumed to be any folder in this directory
    datasets = []
    for item in data_dir_contents:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)

    return datasets

def get_nodes(dataset):
    """
    Lists all nodes in the .h5 file for the given dataset.

    Parameters
    ----------
    dataset <string> : the name of the dataset that you want to inspect.

    Returns
    -------
    [<string>] : a list of all the strings of node names in the dataset.

    """
    hdf5_file = tables.open_file(get_path_to_dataset(dataset)+os.sep+dataset+".h5", mode='r')
    rval = [g._v_pathname for g in hdf5_file.walk_groups('/')]
    hdf5_file.close()
    return rval

def get_data_size(dataset, format=None):
    """

    Parameters
    ----------
    dataset
    format

    Returns
    -------

    """
    hdf5_file = tables.open_file(get_path_to_dataset(dataset) + os.sep + dataset + ".h5", mode='r')


def verify_dataset(dataset_name):
    """ checks if a certain dataset directory exists. Raises an IOError if it doesn't

    Parameters
    ----------
    dataset_name : name of the dataset to check

    Raises
    ------
    IOError : if the dataset dataset_name doesn't exist

    Returns
    -------
    None : if the dataset dataset exists
    """
    if dataset_name not in get_available_datasets():
        raise IOError("The dataset \"%s\" does not exist.  You can "
                      "create it by visiting deep_learning/data and "
                      "creating the directory \"%s\" there." %
                      (dataset_name, dataset_name))

    return None # python does automatically, but makes code pretty


def get_path_to_dataset(dataset_name):
    """ returns the absolute location of the directory for that dataset
    If the dataset (folder) does not exist, raises an IOError, and if
    format.npz does not exist in the directory, an IOError
    is also raised.  Returns the path to the dataset.

    Parameters
    ----------
    dataset_name <string> : dataset to find the directory of

    Returns
    -------
    <string> : The path to the dataset directory
    """
    # check if the dataset exists
    verify_dataset(dataset_name)
    
    # Go up one directory, into the "data" directory, and into the
    # directory dataset_name
    data_dir = get_data_dir_path()
    dataset_dir = os.path.join(data_dir, dataset_name)

    return dataset_dir

def get_experiments_from_dataset(dataset_name):
    """
    Returns a list of experiments for a certain dataset.
    
    Parameters
    ----------
    dataset_name <string> : name of the dataset to find experiments for

    Returns
    -------
    experiments [<string>]: list of the names of experiments for a given dataset
    """
    return next(os.walk(get_path_to_dataset(dataset_name)))[1]

def load_dataset(dataset_name, format, mode='r'):
    """
    Loads a given dataset's group and returns both the file and the data.

    Parameters
    ----------
    dataset_name : The name of the dataset to use
    format : The group of data within that dataset that you wish to select. Hierarchical data should be '/'-separated
    mode ('r', 'w', 'a') <string> : Whether to want to read, write, or append to the data, respectively.

    Returns
    -------
    (<hdf5_file>, (<hdf5.group.x_train>, <hdf5.group.y_train>, <hdf5.group.x_test>, <hdf5.group.y_test>))
    This is a tuple with two elements. The first element is the open hdf5 file object and the second is a
    4-tuple containing the selected group's training data, training labels, testing data, and testing labels
    respectively.

    """
    dataset_path = get_path_to_dataset(dataset_name) + os.sep + dataset_name + ".h5"
    hdf5_file = tables.open_file(dataset_path, mode=mode)
    group = hdf5_file.get_node("/{}".format(format))
    return hdf5_file, (group.x_train, group.y_train, group.x_test, group.y_test)


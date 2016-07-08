import os
import tables

module_dir = os.path.dirname(os.path.realpath(__file__))

def get_data_dir_path():
    """ Gets the absolute path to the data directory

    Returns
    -------
    data_dir_path : absolute path to the data directory
    """
    data_dir_path = os.path.join(module_dir, "..", "data")
    data_dir_path = os.path.realpath(data_dir_path)
    return data_dir_path


def get_available_datasets():
    """ Gets a list of the directories in deep_learning/data
    Each one of these directories is assumed to represent a dataset,
    containing a .npz file with training and testing data

    Returns
    -------
    datasets : list of the folders in deep_learning/data/
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

def get_formats_for_dataset(dataset):
    hdf5_file = tables.open_file(get_path_to_dataset(dataset)+os.sep+dataset+".hdf5", mode='r')
    rval = [g._v_name for g in hdf5_file.walk_groups('/')]
    hdf5_file.close()
    return rval

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

    return None # python does automatically, but makes code pretty


def get_path_to_dataset(dataset_name):
    """ returns the absolute location of the directory dataset_name
    If the dataset (folder) does not exist, raises an IOError, and if
    format.npz does not exist in the directory, an IOError
    is also raised.  Returns the path to the dataset.

    Parameters
    ----------
    dataset_name : dataset to find the path to
    If the dataset name is not valid, an IOError will be raised

    Returns
    -------
    The path to the dataset
    """
    # check if the dataset exists
    verify_dataset(dataset_name)
    
    # Go up one directory, into the "data" directory, and into the
    # directory dataset_name
    data_dir = get_data_dir_path()
    dataset_dir = os.path.join(data_dir, dataset_name)

    return dataset_dir

def get_experiments_from_dataset(dataset_name):
    """ gets a list of the .experiment files from a given dataset
    
    Parameters
    ----------
    dataset_name : name of the dataset to search for .experiment files
    in

    Returns
    -------
    experiments : list of the names of the .experiment files in
    dataset_name
    """
    dataset_path = get_path_to_dataset(dataset_name)

    dataset_files = os.listdir(dataset_path)

    experiment_names = []
    for file_name in dataset_files:
        components = file_name.split(".")
        if components[1] == "exp":
            experiment_names.append(components[0])

    return experiment_names

def load_dataset(dataset_name, format):
    dataset_path = get_path_to_dataset(dataset_name) + os.sep + dataset_name + ".hdf5"
    hdf5_file = tables.open_file(dataset_path, mode='r')
    group = hdf5_file.get_node("/{}".format(format))
    return hdf5_file, (group.x_train, group.y_train, group.x_test, group.y_test)


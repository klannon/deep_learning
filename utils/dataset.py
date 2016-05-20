import os, shutil
import unittest

module_dir = os.path.dirname(os.path.realpath(__file__))

################
## Unit Tests ##
################

class DatasetUtilitiesTestCase(unittest.TestCase):
    def setUp(self):
        # make a test dataset
        data_dir_path = get_data_dir_path()
        os.chdir(data_dir_path)
        os.mkdir("TEST")
        os.chdir("TEST")

        # make some files in the test dataset
        os.system("touch test.experiment")
        os.system("touch is_experiment.experiment")
        os.system("touch is_not_experiment.test")
        os.system("touch PtEtaPhi.npz")
        
        os.chdir(data_dir_path)

        
    def test_get_available_datasets(self):
        """ Test the get_available_datasets function """
        available_datasets = get_available_datasets()
        
        # nominal case
        self.assertTrue("TEST" in available_datasets)

        # if the dataset doesn't exist
        self.assertFalse("probably does not exist" in
                         available_datasets)
    
    def test_verify_dataset(self):
        """ Test the verify_dataset function """
        # nominal case
        self.assertEqual(verify_dataset("TEST"), None)

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            verify_dataset("there's no way this one exists")


    def test_get_path_to_dataset(self):
        """ Test the get_path_to_dataset function """
        # nominal case : just dataset_name
        data_dir_path = get_data_dir_path()
        test_dataset_path = os.path.join(data_dir_path, "TEST")
        self.assertEqual(get_path_to_dataset("TEST"), test_dataset_path)
        
        # nominal case: dataset_name and coordinate_system
        test_coordinate_system_path = os.path.join(test_dataset_path,
                                                   "PtEtaPhi.npz")
        self.assertEqual(get_path_to_dataset("TEST", "PtEtaPhi"),
                         test_coordinate_system_path)

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            get_path_to_dataset("some random folder")

        # if the coordinate system doesn't exist in a valid dataset
        with self.assertRaises(IOError):
            get_path_to_dataset("TEST", "Spherical")

            
    def test_get_experiments_from_dataset(self):
        """ Test the get_experiments_from_dataset function """
        # nominal case
        exp = get_experiments_from_dataset("TEST")
        self.assertTrue(set(["test", "is_experiment"]).issubset(exp))

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            get_experiments_from_dataset("something that doesn't exist")

        # if the file exists in the dataset but isn't .experiment
        self.assertFalse("is_not_experiment" in exp)

        # if the file does not exist in the dataset
        self.assertFalse("file_not_in_dataset" in exp)


    def tearDown(self):
        data_dir_path = get_data_dir_path()
        shutil.rmtree("TEST")




        
######################
## Module functions ##
######################

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
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
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
    data_dir = get_data_dir_path()
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
        if components[1] == "experiment":
            experiment_names.append(components[0])

    return experiment_names

if __name__ == "__main__":
    # uncomment the next 2 lines to run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(DatasetUtilitiesTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)


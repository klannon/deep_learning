import os, shutil
import unittest
import deep_learning.utils.dataset as ds
import deep_learning.utils.archive as ar

module_dir = os.path.dirname(os.path.realpath(__file__))


##################################
## Unit Test For Dataset Module ##
##################################

class DatasetUtilitiesTestCase(unittest.TestCase):
    def setUp(self):
        # make a test dataset
        data_dir_path = ds.get_data_dir_path()
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
        available_datasets = ds.get_available_datasets()

        # nominal case
        self.assertTrue("TEST" in available_datasets)

        # if the dataset doesn't exist
        self.assertFalse("probably does not exist" in
                         available_datasets)

    def test_verify_dataset(self):
        """ Test the verify_dataset function """
        # nominal case
        self.assertEqual(ds.verify_dataset("TEST"), None)

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            ds.verify_dataset("there's no way this one exists")

    def test_get_path_to_dataset(self):
        """ Test the get_path_to_dataset function """
        # nominal case : just dataset_name
        data_dir_path = ds.get_data_dir_path()
        test_dataset_path = os.path.join(data_dir_path, "TEST")
        self.assertEqual(ds.get_path_to_dataset("TEST"), test_dataset_path)

        # nominal case: dataset_name and coordinate_system
        test_coordinate_system_path = os.path.join(test_dataset_path,
                                                   "PtEtaPhi.npz")
        self.assertEqual(ds.get_path_to_dataset("TEST", "PtEtaPhi"),
                         test_coordinate_system_path)

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            ds.get_path_to_dataset("some random folder")

        # if the coordinate system doesn't exist in a valid dataset
        with self.assertRaises(IOError):
            ds.get_path_to_dataset("TEST", "Spherical")

    def test_get_experiments_from_dataset(self):
        """ Test the get_experiments_from_dataset function """
        # nominal case
        exp = ds.get_experiments_from_dataset("TEST")
        self.assertTrue(set(["test", "is_experiment"]).issubset(exp))

        # if the dataset doesn't exist
        with self.assertRaises(IOError):
            ds.get_experiments_from_dataset("something that doesn't exist")

        # if the file exists in the dataset but isn't .experiment
        self.assertFalse("is_not_experiment" in exp)

        # if the file does not exist in the dataset
        self.assertFalse("file_not_in_dataset" in exp)

    def tearDown(self):
        data_dir_path = ds.get_data_dir_path()
        shutil.rmtree("TEST")


###################################
## Unit Tests For Archive Module ##
###################################

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
            ds.load_dataset("this one definitely doesn't exist either",
                         "")

    def test_read_config_file(self):
        """ Tests the read_config_file function """
        # nominal case
        train, test = ar.read_config_file("TEST", "PtEtaPhi")
        test_dataset_path = ds.get_path_to_dataset("TEST")
        self.assertEqual(train, os.path.join(test_dataset_path,
                                             "train_all_ptEtaPhi.txt"))
        self.assertEqual(test, os.path.join(test_dataset_path,
                                            "test_all_ptEtaPhi.txt"))

        # if the dataset doesn't exist but the coordinate system does
        with self.assertRaises(IOError):
            ar.read_config_file("no way does this dataset exist",
                             "PtEtaPhi")

        # if the coordinate system doesn't exist in a valid dataset
        with self.assertRaises(KeyError):
            ar.read_config_file("TEST", "Spherical")

    def tearDown(self):
        data_dir_path = ds.get_data_dir_path()
        shutil.rmtree("TEST")
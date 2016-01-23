import os.path
import pylearn2

# used to save the dataset
from pylearn2.utils import serial

# raw dataset operations (loading csv files, shuffling, similar stuff)
import template.physics as physics
print(physics.__file__)

# Dataset
pathToTrainValidData = (os.environ['PYLEARN2_DATA_PATH'] +
                        os.sep+'train_all_3v_ttbar_wjet.txt')
pathToTestData = (os.environ['PYLEARN2_DATA_PATH'] +
                  os.sep+'test_all_3v_ttbar_wjet.txt')
    
train_fraction = 0.8 # 1700000 in train file for train and valid
numLabels = 2 # Number of output nodes...softmax interpretation here
    
train, valid, test = physics.PHYSICS(pathToTrainValidData,
                                     pathToTestData, train_fraction,
                                     numLabels=numLabels)

save_path = os.getcwd()

# training dataset
train.use_design_loc(os.path.join(save_path, "train.npy"))
train_pkl_path = os.path.join(save_path, "train.pkl")
serial.save(train_pkl_path, train)

# testing dataset
test.use_design_loc(os.path.join(save_path, "test.npy"))
test_pkl_path = os.path.join(save_path, "test.pkl")
serial.save(test_pkl_path, test)

# validation/monitoring dataset
valid.use_design_loc(os.path.join(save_path, "valid.npy"))
valid_pkl_path = os.path.join(save_path, "valid.pkl")
serial.save(valid_pkl_path, valid)


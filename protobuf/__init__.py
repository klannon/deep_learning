from experiment_pb2 import *
from optimizers_pb2 import *
from deep_learning.utils.dataset import get_path_to_dataset
from os import sep

def load_experiment(experiment):
    dataset, exp = experiment.split('/')
    exp_path = get_path_to_dataset(dataset)+sep+exp+sep+exp+'.exp'
    with open(exp_path, 'rb') as f:
        exp = Experiment()
        exp.ParseFromString(f.read())
    return exp
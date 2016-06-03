from experiment_pb2 import *
from optimizers_pb2 import *

def load_experiment(experiment_file_path):
    f = open(experiment_file_path)
    exp = Experiment()
    exp.ParseFromString(f.read())
    f.close()
    return exp
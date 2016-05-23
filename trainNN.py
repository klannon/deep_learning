from __future__ import print_function

import datetime
import os
import sys
import time

from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import SGD

import deep_learning.protobuf.experiment_pb2 as pb
import deep_learning.utils.read_write as rw
import deep_learning.utils.dataset as ds

##
# Experiment log set-up
##
exp = pb.Experiment() # container for all info about this experiment
exp.start_date_time = str(datetime.datetime.now())
save_dir = ds.get_path_to_dataset(pb.Experiment.Dataset.Name(exp.Dataset))

##
# Constants
##
exp.dataset = pb.Experiment.Dataset.Value("OSU_TTBAR")
exp.coordinate_system = "PtEtaPhi"

# file naming scheme so simultaneous experiments can be run
output_file_name = ("%s_%s" % (os.getenv("HOST", os.getpid()), time.time()))
experiment_file_name = os.path.join(save_dir, ("%s.experiment" % output_file_name))

##
# Log file set-up (temporary, future will save all info to exp)
##
log_file_path = os.path.join(save_dir, ("%s.log" % output_file_name))
sys.stdout = open(log_file_path, 'w')

##
# Load data from .npz archive created by invoking
# deep_learning/utils/read_write.py
##
x_train, y_train, x_test, y_test = ds.load_dataset(pb.Experiment.Dataset.Name(exp.Dataset),exp.coordinate_system)
print("loaded data")


##
# Construct the network
# TODO: save structure to exp
##
model = Sequential()

layer = exp.structure.add()
layer.Type = "DENSE"
layer.input_dimension = 15
layer.output_dimension = 50
model.add(Dense(50, input_dim=15))
model.add(Activation("relu"))

model.add(Dense(50))
model.add(Activation("relu"))

model.add(Dense(50))
model.add(Activation("relu"))

model.add(Dense(50))
model.add(Activation("relu"))

model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

print("added all the layers")


opt = exp.optimizer
opt = pb.SGD()
opt.lr=0.1
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=opt.lr), metrics=['accuracy'])

print("compiled the model")

##
# Train the model.  Need to finish implementing the batch-by-batch
# training. Experimenting with moving whole dataset to gpu at once
##

# train_length = x_train.shape[0]
# indices = np.arrange(train_length)

# ceil b/c train_length % batch_size isn't necessarily equal to 0
# num_batches = math.ceil(train_length / batch_size)


model.fit(x_train, y_train,
          nb_epoch=1,
          batch_size=64,
          verbose=2)

print("trained the model")

print(model.evaluate(x_test, y_test, batch_size=64, verbose=0))

print("evaluated the model")

exp.end_date_time = str(datetime.datetime.now())


##
# Write the Experiment object to file
# TODO: write to file every 5-10 epochs (when writing the model
# to a .json file) in case job gets killed
##
experiment_file = open(experiment_file_name, "wb")
experiment_file.write(exp.SerializeToString())
experiment_file.close()

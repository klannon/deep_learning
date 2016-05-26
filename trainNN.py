from __future__ import print_function

import datetime, os, sys, time

from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import SGD

import deep_learning.protobuf as pb
import deep_learning.utils.dataset as ds
from deep_learning.utils import Angler
from math import ceil
from time import clock

##
# Experiment log set-up
##
exp = pb.Experiment() # container for all info about this experiment
exp.start_date_time = str(datetime.datetime.now())


##
# Constants
##
exp.dataset = pb.Experiment.Dataset.Value("OSU_TTBAR")
exp.coordinates = "PtEtaPhi"
save_dir = ds.get_path_to_dataset(pb.Experiment.Dataset.Name(exp.dataset))

# file naming scheme so simultaneous experiments can be run
output_file_name = ("%s_%s" % (str(os.getenv("HOST", os.getpid())).split('.')[0], time.time()))
experiment_file_name = os.path.join(save_dir, ("%s.experiment" % output_file_name))

#sys.stdout = Angler(exp)

##
# Load data from .npz archive created by invoking
# deep_learning/utils/archive.py
##
x_train, y_train, x_test, y_test = ds.load_dataset(pb.Experiment.Dataset.Name(exp.dataset), exp.coordinates)
print("loaded data")


##
# Construct the network
# TODO: save structure to exp
##
model = Sequential()

layer = exp.structure.add()
layer.type = 0
layer.input_dimension = 15
layer.output_dimension = 50

model.add(Dense(50, input_dim=15))
model.add(Activation("relu"))

for i in xrange(3):
    layer = exp.structure.add()
    layer.type = 0
    layer.input_dimension = 50
    layer.output_dimension = 50
    model.add(Dense(50))
    model.add(Activation("relu"))

layer = exp.structure.add()
layer.type = 1
layer.input_dimension = 50
layer.output_dimension = 2
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

print("added all the layers")


opt = pb.SGD()
opt.lr = 0.1
exp.sgd.MergeFrom(opt)

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=opt.lr), metrics=['accuracy'])

print("compiled the model")

##
# Train the model.  Need to finish implementing the batch-by-batch
# training. Experimenting with moving whole dataset to gpu at once
##

num_epochs = 20
batch_size = 64
train_length = x_train.shape[0]
num_batches = int(ceil(train_length/batch_size))

# indices = np.arrange(train_length)

# ceil b/c train_length % batch_size isn't necessarily equal to 0
# num_batches = math.ceil(train_length / batch_size)

exp.batch_size = batch_size

clock()

for i in xrange(num_epochs):
    t = clock()
    for b in xrange(num_batches):
        model.train_on_batch(x_train[b*batch_size:b*batch_size+batch_size, :], y_train[b*batch_size:b*batch_size+batch_size, :])
    print(i, model.evaluate(x_test, y_test, batch_size=64, verbose=0))
    print(clock()-t)



print("trained the model")

print(model.evaluate(x_test, y_test, batch_size=64, verbose=0))

print("evaluated the model")
print("Total Time: {}".format(clock()))

exp.end_date_time = str(datetime.datetime.now())
exp.total_time = clock()


##
# Write the Experiment object to file
# TODO: write to file every 5-10 epochs (when writing the model
# to a .json file) in case job gets killed
##

experiment_file = open(experiment_file_name, "wb")
experiment_file.write(exp.SerializeToString())
experiment_file.close()

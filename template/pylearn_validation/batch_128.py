from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import sys
sys.stdout = open('batch_128.log', 'w')

from load_data import load_all

x_train, y_train, x_test, y_test = load_all()
print("loaded data")

model = Sequential()

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

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

print("compiled the model")

model.fit(x_train, y_train,
          nb_epoch=200,
          batch_size=128,
          verbose=2)

print("trained the model")

print(model.evaluate(x_test, y_test, batch_size=128, verbose=0))

print("evaluated the model")

# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')

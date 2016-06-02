"""
This file is used to write a configuration file for future use in training.
"""

from __future__ import print_function
import os
from os import path
from deep_learning.utils.dataset import get_data_dir_path, get_path_to_dataset
import json

def set_configurations():
    print("What would you like to name this experiment setup?")
    print("*Note: this is used for naming the result files")
    name = raw_input(">> ")

    print("Which dataset would you like to run on?")
    ## Collect the names of the directories here
    # Selects dataset
    data_dir = get_data_dir_path()
    print(' '.join(filter(lambda x: path.isdir(path.join(data_dir, x)), os.listdir(data_dir))))
    dataset = raw_input(">> ")
    print()

    data_dir = get_path_to_dataset(dataset)
    with open(path.join(data_dir, dataset+".json")) as fp:
        j = json.load(fp)
    print("Which coordinates would you like to use?")
    print(' '.join(j.keys()))
    coords = raw_input(">> ")
    print()

    # Collect data files from that dataset and repeat question

    batch_size = raw_input("What will be your batch size?\n>> ")
    print()
    learning_rate = raw_input("What will be your learning rate?\n>> ")
    print()

    layers = raw_input("How many layers do you want?\n>> ")
    print()

    nodes = raw_input("How many nodes per layer do you want?\n>> ")
    print()

    print("How often do you want to save the model?")
    print("(Number of epochs)")
    sf = raw_input(">> ")
    print()

    print("How would you like to determine the end of the program:")
    print("1. After a certain number of epochs")
    print("2. After a certain amount of time")
    print("3. When the accuracy plateaus")
    print("*Note: You may select multiple, just separate them by spaces.")
    ending = filter(None, raw_input("[1/2/3] >> ").split())
    print()

    terms = {"epochs": None, "timeout": None, "plateau": {"m": None, "w": None}}
    if '1' in ending:
        terms['epochs'] = int(raw_input("How many epochs do you want to run?\n>> "))
        print()
    if '2' in ending:
        print("After how long do you want the program to be killed?")
        print("(Put units after the number, i.e. s/m/h/d/w/y)")
        time = raw_input(">> ")
        print()
        unit = time[-1]
        time = int(time[:-1])
        if unit == 'm':
            time *= 60
        elif unit == 'h':
            time *= 3600
        elif unit == 'd':
            time *= 3600*24
        elif unit == 'w':
            time *= 3600*24*7
        elif unit == 'y':
            time *= 3600*24*7*52
        terms['timeout'] = time
    if '3' in ending:
        print("For determining plateau:")
        x = raw_input("Over what interval would you like to measure the accuracy change?\n>> ")
        print()
        y = raw_input("What is the minimal increase in percentile you can accept over this interval?\n>> ")
        print()
        terms['plateau'] = dict(x=float(x), y=int(y))

    d = dict(save_name=name,
             dataset=dataset,
             coords=coords,
             batch_size=int(batch_size),
             learning_rate=float(learning_rate),
             layers=int(layers),
             nodes=int(nodes),
             save_freq=int(sf),
             terms=terms)
    return d

def write_configurations(d):

    with open(d['name']+".cfg", 'w') as fp:
        json.dump(d, fp)

    print("Finished! Remember, you can always manually overwrite these values from the command line later.")

if __name__ == '__main__':
    d = set_configurations()
    write_configurations(d)

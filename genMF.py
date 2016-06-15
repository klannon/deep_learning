from __future__ import print_function
import argparse, os
from deep_learning import trainNN
import deep_learning.utils.dataset as ds

template = '{0}{7}{1}{7}cfg.json {0}{7}{1}{7}{1}.exp {0}{7}{1}{7}weights.npy : {4} {0}{7}{3}.npz\n'+\
           '    python {4} {5} {2}/{3} {6}'

def write_Makeflow(datafile, **funcs):
    # var to update, num of jobs, func(var, i)

    sys_parser = argparse.ArgumentParser()
    sys_parser.add_argument("jobs", metavar="numJobs", type=int)
    sys_parser.add_argument("-f", "--flags", default='')
    sys_parser.add_argument("-o", "--out", default="jobs.mf")
    sys_args = vars(sys_parser.parse_args())

    # Will need to update dataset and name at minimum

    switch_map = {"b": "batch_size",
                  "r": "learning_rate",
                  "e": "max_epochs",
                  "n": "nodes",
                  "l": "layers",
                  "t": "timeout",
                  "m": "monitor_fraction",
                  "f": "save_freq",
                  "x": "run",
                  "y": "rise",
                  "d": "dataset",
                  "s": "save_name"}
    for key in funcs:
        funcs[switch_map.get(key, key)] = funcs.pop(key)

    with open(sys_args["out"], 'w') as f:
        for i in xrange(sys_args["jobs"]):
            ups = ' '.join(["--{}{}".format(k, v(i)) for k,v in funcs.items()])
            name = funcs["save_name"](i)
            dataset, format = datafile(i).split('/')
            datadir = ds.get_path_to_dataset(dataset)
            print(template.format(datadir, name, dataset, format, trainNN.__file__[:-1],
                                  sys_args["flags"], ups, os.sep), file=f)

if __name__ == '__main__':
    #Dict of funcs that update a parameter, i.e.:
    funcs = {"s": lambda i: "job{}".format(i), }      # Updates the save name (required!)
    #         "l": lambda i: (i+1)*10}               # Updates the num_layers
    write_Makeflow(lambda i: "ttHLep/Sorted", **funcs)
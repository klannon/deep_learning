#!/auto/igb-libs/linux/centos/6.x/x86_64/pkgs/python/2.7.4/bin/python
#import argparse
import sys
import os
import theano
import pylearn2
import pylearn2.datasets.physics
import pylearn2.training_algorithms.sgd
import pylearn2.termination_criteria
import pylearn2.costs.mlp.dropout 
#import pylearn2.space
import pylearn2.models.mlp as mlp
import pylearn2.train

###########
# CHANGES
#  Changed where the datasets are defined because physics.PHYSICS was changed.
#
#  Manually implemented a dataset_train_monitor set with the new code.
###########


def init_train():
    # Initialize train object.
    idpath = os.path.splitext(os.path.abspath(__file__))[0] # ID for output files.
    save_path = idpath + '.pkl'

    # Dataset
    path = os.environ['PYLEARN2_DATA_PATH']+os.sep+'SUSY.csv'
    derived_feat, nvis = False, 8
    dataset_train, dataset_valid, dataset_test = pylearn2.datasets.physics.PHYSICS(path, 0.6, 0.2, derived_feat)
    dataset_train_monitor = pylearn2.datasets.physics._PHYSICS({'data': dataset_train.X[:100000, :], 'labels': dataset_train.y[:100000].reshape(100000, 1)}, 'train', dataset_train.args['benchmark'])
    
    # Parameters
    momentum_saturate = 200
    
    # Model
    model = pylearn2.models.mlp.MLP(layers=[mlp.Tanh(
                                                layer_name='h0',
                                                dim=300,
                                                istdev=.1),
                                            mlp.Tanh(
                                                layer_name='h1',
                                                dim=300,
                                                istdev=.05),
                                            mlp.Tanh(
                                                layer_name='h2',
                                                dim=300,
                                                istdev=.05),
                                            mlp.Tanh(
                                                layer_name='h3',
                                                dim=300,
                                                istdev=.05),
                                            mlp.Sigmoid(
                                                layer_name='y',
                                                dim=1,
                                                istdev=.001)
                                           ],
                                    nvis=nvis
                                    )

    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
                    batch_size=100,   # If changed, change learning rate!
                    learning_rate=.05, # In dropout paper=10 for gradient averaged over batch. Depends on batchsize.
#                    init_momentum=.9, 
                    monitoring_dataset = {'train':dataset_train_monitor,
                                          'valid':dataset_valid,
                                          'test':dataset_test
                                          },
                    termination_criterion=pylearn2.termination_criteria.Or(criteria=[
                                            pylearn2.termination_criteria.MonitorBased(
                                                channel_name="valid_objective",
                                                prop_decrease=0.00001,
                                                N=40),
                                            pylearn2.termination_criteria.EpochCounter(
                                                max_epochs=momentum_saturate)
                                            ]),
                    #cost=pylearn2.costs.cost.SumOfCosts(
                    #    costs=[pylearn2.costs.mlp.Default()
                    #           ]
                    #),
                    cost = pylearn2.costs.mlp.dropout.Dropout(
                        input_include_probs={'h0':1., 'h1':1., 'h2':1., 'h3':1., 'y':0.5},
                        input_scales={ 'h0': 1., 'h1':1., 'h2':1., 'h3':1., 'y':2.}),

                    update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                        decay_factor=1.0000003, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
                                        min_lr=.000001
                                        )
                )
    # Extensions 
    # extensions=[ 
    #     #pylearn2.train_extensions.best_params.MonitorBasedSaveBest(channel_name='train_y_misclass',save_path=save_path)
    #     pylearn2.training_algorithms.sgd.MomentumAdjustor(
    #         start=0,
    #         saturate=momentum_saturate,
    #         final_momentum=.99  # Dropout=.5->.99 over 500 epochs.
    #         )
    #     ]
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 # extensions=extensions,
                                 save_path=save_path,
                                 save_freq=100)
    return train
    
def train(mytrain):
    # Execute training loop.
    debug = False
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print 'Using=%s' % theano.config.device # Can use gpus. 
    print 'Writing to %s' % logfile
    print 'Writing to %s' % mytrain.save_path
    sys.stdout = open(logfile, 'w')
    print "opened log file"
    mytrain.main_loop()


if __name__=='__main__':
    # Initialize and train.
    mytrain = init_train()
    train(mytrain)


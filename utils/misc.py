from __future__ import division
import numpy as np
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr

def splitter(dataset, ratios):
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)
    total_background = int(y_train[:,0].sum()+y_test[:,0].sum())
    total_signal = int(y_train[:,1].sum()+y_test[:,1].sum())

    UPPER_LIMIT = int(1.5*total_background) if total_background < total_signal else int(1.5*total_signal)

    datasets = []

    for i in xrange(ratios):
        denom = sum(ratios[i], ratios[-i-1])
        back_ix = np.random.choice(total_background, total_background-(int((ratios[i] / denom) * UPPER_LIMIT)), replace=False)
        sig_ix = np.random.choice(total_signal, total_signal-(int((ratios[-i-1] / denom) * UPPER_LIMIT)), replace=False)

        all_x_signal = np.concatenate((x_train[y_train[:,1]==1], x_test[y_test[:,1]==1]))
        all_y_signal = np.concatenate((y_train[y_train[:,1]==1], y_test[y_test[:,1]==1]))
        all_x_background = np.concatenate((x_train[y_train[:,0]==1], x_test[y_test[:,0]==1]))
        all_y_background = np.concatenate((y_train[y_train[:,0]==1], y_test[y_test[:,0]==1]))

        small_x_signal = np.delete(all_x_signal, sig_ix, axis=0)
        small_y_signal = np.delete(all_y_signal, sig_ix, axis=0)
        small_x_background = np.delete(all_x_background, back_ix, axis=0)
        small_y_background = np.delete(all_y_background, back_ix, axis=0)

        all_x = np.concatenate((small_x_background, small_x_signal))
        all_y = np.concatenate((small_y_background, small_y_signal))

        tr.shuffle_in_unison(all_x, all_y)

        cutoff = int(all_x.shape[0] * 0.85)  # 80% training 20% testing
        train_x = all_x[:cutoff]
        train_y = all_y[:cutoff]
        test_x = all_x[cutoff:]
        test_y = all_y[cutoff:]

        datasets.append((train_x, train_y, test_x, test_y))

    return datasets
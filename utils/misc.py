from __future__ import division
import numpy as np
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr

def splitter(dataset, ratios, separate=False):
    data, format = dataset.split('/')
    x_train, y_train, x_test, y_test = ds.load_dataset(data, format)
    total_background = int(y_train[:,0].sum()+y_test[:,0].sum())
    total_signal = int(y_train[:,1].sum()+y_test[:,1].sum())

    if separate:
        bkg_test = int(y_test[:,0].sum())
        bkg_train = int(y_train[:,0].sum())
        sig_test = int(y_test[:,1].sum())
        sig_train = int(y_train[:,1].sum())

        TEST_UPPER_LIMIT = int(1.5*bkg_test) if bkg_test < sig_test else int(1.5*sig_test)
        TRAIN_UPPER_LIMIT = int(1.5*bkg_train) if bkg_train < sig_train else int(1.5*sig_train)

    else:
        all_x_signal = np.concatenate((x_train[y_train[:, 1] == 1], x_test[y_test[:, 1] == 1]))
        all_y_signal = np.concatenate((y_train[y_train[:, 1] == 1], y_test[y_test[:, 1] == 1]))
        all_x_background = np.concatenate((x_train[y_train[:, 0] == 1], x_test[y_test[:, 0] == 1]))
        all_y_background = np.concatenate((y_train[y_train[:, 0] == 1], y_test[y_test[:, 0] == 1]))

        UPPER_LIMIT = int(1.5*total_background) if total_background < total_signal else int(1.5*total_signal)

    datasets = []

    for i in xrange(len(ratios)):
        denom = sum((ratios[i], ratios[-i-1]))

        if separate:
            test_bkg_ix = np.random.choice(bkg_test, bkg_test - (int((ratios[i] / denom) * TEST_UPPER_LIMIT)),
                                           replace=False)
            test_sig_ix = np.random.choice(sig_test, sig_test - (int((ratios[-i - 1] / denom) * TEST_UPPER_LIMIT)),
                                           replace=False)
            train_bkg_ix = np.random.choice(bkg_train, bkg_train - (int((ratios[i] / denom) * TRAIN_UPPER_LIMIT)),
                                            replace=False)
            train_sig_ix = np.random.choice(sig_train, sig_train - (int((ratios[-i - 1] / denom) * TRAIN_UPPER_LIMIT)),
                                            replace=False)

            test_small_x_sig = np.delete(x_test[y_test[:, 1] == 1], test_sig_ix, axis=0)
            test_small_y_sig = np.delete(y_test[y_test[:, 1] == 1], test_sig_ix, axis=0)
            test_small_x_bkg = np.delete(x_test[y_test[:, 0] == 1], test_bkg_ix, axis=0)
            test_small_y_bkg = np.delete(y_test[y_test[:, 0] == 1], test_bkg_ix, axis=0)
            train_small_x_sig = np.delete(x_train[y_train[:, 1] == 1], train_sig_ix, axis=0)
            train_small_y_sig = np.delete(y_train[y_train[:, 1] == 1], train_sig_ix, axis=0)
            train_small_x_bkg = np.delete(x_train[y_train[:, 0] == 1], train_bkg_ix, axis=0)
            train_small_y_bkg = np.delete(y_train[y_train[:, 0] == 1], train_bkg_ix, axis=0)

            train_x = np.concatenate((train_small_x_bkg, train_small_x_sig))
            train_y = np.concatenate((train_small_y_bkg, train_small_y_sig))
            test_x = np.concatenate((test_small_x_bkg, test_small_x_sig))
            test_y = np.concatenate((test_small_y_bkg, test_small_y_sig))

            tr.shuffle_in_unison(train_x, train_y)
            tr.shuffle_in_unison(test_x, test_y)


        else:
            bkg_ix = np.random.choice(total_background, total_background - (int((ratios[i] / denom) * UPPER_LIMIT)),
                                       replace=False)
            sig_ix = np.random.choice(total_signal, total_signal - (int((ratios[-i - 1] / denom) * UPPER_LIMIT)),
                                      replace=False)

            small_x_signal = np.delete(all_x_signal, sig_ix, axis=0)
            small_y_signal = np.delete(all_y_signal, sig_ix, axis=0)
            small_x_background = np.delete(all_x_background, bkg_ix, axis=0)
            small_y_background = np.delete(all_y_background, bkg_ix, axis=0)

            all_x = np.concatenate((small_x_background, small_x_signal))
            all_y = np.concatenate((small_y_background, small_y_signal))

            tr.shuffle_in_unison(all_x, all_y)

            cutoff = int(all_x.shape[0] * 0.80)  # 80% training 20% testing
            train_x = all_x[:cutoff]
            train_y = all_y[:cutoff]
            test_x = all_x[cutoff:]
            test_y = all_y[cutoff:]

        datasets.append((train_x, train_y, test_x, test_y))

    return datasets
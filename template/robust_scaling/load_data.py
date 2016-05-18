import numpy as np
from transformations import transform


def load_train():
    train = np.genfromtxt("/scratch365/cdablain/dnn/data/train_all_ptEtaPhi_ttbar_wjet.txt", delimiter=',')
    x_train = train[:,1:]
    print x_train[0:2]

    y_train_raw = train[:, 0]
    y_train_list = []
    # [ 1.  2.]
    for row in y_train_raw:
        if row == 1:
            y_train_list.append([0., 1.])
        if row == 2:
            y_train_list.append([1., 0.])
    y_train = np.array(y_train_list)
    print y_train[:3]
    return(x_train, y_train)


def load_test():
    test = np.genfromtxt("/scratch365/cdablain/dnn/data/test_all_ptEtaPhi_ttbar_wjet.txt", delimiter=',')
    x_test = test[:,1:]
    print x_test[0:2]
    y_test_raw = test[:, 0]
    y_test_list = []
    # [ 1.  2.]
    for row in y_test_raw:
        if row == 1:
            y_test_list.append([0., 1.])
        if row == 2:
            y_test_list.append([1., 0.])
    y_test = np.array(y_test_list)
    print y_test[0:2]

    return(x_test, y_test)

def load_all():
    data = np.load("data.npz")
    return(data['x_train'], data['y_train'], data['x_test'], data['y_test'])

def write_data_to_archive():
    (x_train, y_train) = load_train()
    (x_test, y_test) = load_test()
    
    (x_train, x_test) = transform(x_train, x_test)
    print x_train[0:2]
    print x_test[0:2]
    
    np.savez("data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

if __name__ == "__main__":
    write_data_to_archive()

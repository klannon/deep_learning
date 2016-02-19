from sklearn import preprocessing as pp

__author__ = 'Matthew Drnevich'

def standardize(datasets):   return map(pp.scale, datasets)

def get_transform(dataset):   return pp.StandardScaler().fit(dataset)

def transform(train_set, test_set):
    scale = pp.StandardScaler().fit(train_set.X)
    train_set.X, test_set.X = scale.transform(train_set.X), scale.transform(test_set.X)
    return train_set, test_set
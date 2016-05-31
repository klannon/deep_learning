import os

def get_datasets():
    return os.listdir("../data")

def get_files_from_dataset(dataset):
    return os.listdir("../data/%s" % dataset)
    
if __name__ == "__main__":
    print get_datasets()

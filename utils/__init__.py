
def which(myDict):
    rval = ""
    for k, v in myDict.items():
        if v is True:
            rval += k + ', '
    return rval.rstrip(', ') if rval.rstrip(', ') else None

def kParse(string):
    """
    Parse the output from Keras into the numerical values only.

    Parameters
    ----------
    string

    Returns
    -------
    List of numbers output by Keras (time, loss, acc, val_loss, val_acc)
    """
    nums = filter(lambda x: x.isdigit() or x[:-1].isdigit() or len(filter(lambda y: y.isdigit(), x.split('.')))==2,
           string.split())
    if len(nums) >= 3:
        nums[0] = int(nums[0][:-1])
        nums[1:] = [float(x) for x in nums[1:]]
    return nums

class Angler(object):
    """
    Catches the output from Keras and writes it to an Experiment instance.
    """
    def __init__(self, exp):
        self.experiment = exp

    def write(self, string):
        nums = kParse(string)
        if len(nums) == 3:
            a = self.experiment.results.add()
            a.num_seconds, a.train_loss, a.train_accuracy = nums
        elif len(nums) == 5:
            a = self.experiment.results.add()
            a.num_seconds, a.train_loss, a.train_accuracy, a.val_loss, a.val_accuracy = nums
        else:
            pass

    def close(self):
        pass


"""
Create UID (Unique Identifier Code) for naming log + experiment files, much like permisisons.
Utilize hexadecimal labeling with
"""
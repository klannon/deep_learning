from __future__ import division
import matplotlib.pyplot as plt
from deep_learning.protobuf import load_experiment

def s_b(experiment):
    exp = experiment
    s_b_array = [r.s_b for r in exp.results]
    plt.subplot(221)
    plt.plot(xrange(len(s_b_array)), s_b_array)
    plt.title("S/sqrt(B)")
    plt.xlabel("Epoch")
    plt.ylabel("Significance")
    #plt.show()
    #plt.savefig("s_b.png", format="png")

def auc(experiment):
    exp = experiment
    auc_array = [r.auc for r in exp.results]
    plt.subplot(222)
    plt.plot(xrange(len(auc_array)), auc_array)
    plt.title("Area Under the Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Area")
    #plt.show()
    #plt.savefig("auc.png", format="png")

def correct(experiment):
    exp = experiment
    tb = sum(exp.results[0].matrix[0].columns)
    ts = sum(exp.results[0].matrix[1].columns)
    bb = [r.matrix[0].columns[0]/tb for r in exp.results]
    ss = [r.matrix[1].columns[1]/ts for r in exp.results]
    plt.subplot(223)
    plt.plot(xrange(len(bb)), bb, '', xrange(len(ss)), ss)
    plt.title("Correct Confusion Matrix Entries")
    plt.xlabel("Epoch")
    plt.ylabel("Percent Accuracy")
    #plt.show()
    #plt.savefig("correct.png", format="png")

def accuracy(experiment):
    exp = experiment
    acc_array = [r.test_accuracy for r in exp.results]
    plt.subplot(224)
    plt.plot(xrange(len(acc_array)), acc_array)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #plt.show()
    #plt.savefig("accuracy.png", format="png")

if __name__ == "__main__":
    ratios = [2,1,1]
    for i, e in xrange(len(ratios)):
        exp = load_experiment("ttHLep/drop_split_big")
        s_b(e)
        auc(e)
        correct(e)
        accuracy(e)
        plt.tight_layout()
        plt.savefig("DropBig{}to{}.png".format(ratios[i], ratios[-1 - i]), format="png")
        plt.clf()
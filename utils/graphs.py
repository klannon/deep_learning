from __future__ import division
import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
from math import sqrt
import numpy as np
from deep_learning.protobuf import load_experiment
from math import ceil
import deep_learning.utils as ut
import deep_learning.utils.dataset as ds
import deep_learning.utils.transformations as tr
import deep_learning.utils.stats as st

def s_b(experiment, label=None, subplot=221):
    exp = experiment
    label = label if label else exp.description
    s_b_array = [r.s_b for r in exp.results]
    if subplot:
        plt.subplot(subplot)
    plt.plot(xrange(len(s_b_array)), s_b_array, label=label)
    plt.title("S/sqrt(B)")
    plt.xlabel("Epoch")
    plt.ylabel("Significance")
    plt.legend(loc=0, fontsize="x-small")
    #plt.show()
    #plt.savefig("s_b.png", format="png")

def auc(experiment, label=None, subplot=222):
    exp = experiment
    label = label if label else exp.description
    auc_array = [r.auc for r in exp.results]
    if subplot:
        plt.subplot(subplot)
    plt.plot(xrange(len(auc_array)), auc_array, label=label)
    plt.title("Area Under the Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Area")
    plt.legend(loc=0, fontsize="x-small")
    #plt.show()
    #plt.savefig("auc.png", format="png")

def correct(experiment, labels=None, subplot=223):
    exp = experiment
    labels = labels if labels else [exp.description+" BB", exp.description+" SS"]
    tb = sum(exp.results[0].matrix[0].columns)
    ts = sum(exp.results[0].matrix[1].columns)
    bb = [r.matrix[0].columns[0]/tb for r in exp.results]
    ss = [r.matrix[1].columns[1]/ts for r in exp.results]
    if subplot:
        plt.subplot(subplot)
    plt.plot(xrange(len(bb)), bb, '', label=labels[0])
    plt.plot(xrange(len(ss)), ss, '', label=labels[1])
    plt.title("Correct Confusion Matrix Entries")
    plt.xlabel("Epoch")
    plt.ylabel("Percent Accuracy")
    plt.legend(loc=0, fontsize="x-small")
    #plt.show()
    #plt.savefig("correct.png", format="png")

def accuracy(experiment, label=None, subplot=224):
    exp = experiment
    label = label if label else exp.description
    acc_array = [r.test_accuracy for r in exp.results]
    if subplot:
        plt.subplot(subplot)
    plt.plot(xrange(len(acc_array)), acc_array, label=label)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc=0, fontsize="x-small")
    #plt.show()
    #plt.savefig("accuracy.png", format="png")

def permute_accuracy(model, data):
    pass

def roc_curve(experiment, label=None, subplot=None):
    exp = experiment
    label = label if label else exp.description
    curve = exp.results[-1].curve
    x_points = [point.background for point in curve]
    y_points = [point.signal for point in curve]
    if subplot:
        plt.subplot(subplot)
    plt.plot(x_points, y_points, label=label)
    plt.title("ROC Curve")
    plt.xlabel("Background Inefficiency")
    plt.ylabel("Signal Efficiency")
    plt.legend(loc=0, fontsize="x-small")

def _new_roc_curve(model, data, datapoints=20, label=None, subplot=None):
    datapoints += 1
    e_b = np.zeros(datapoints)
    e_s = np.zeros(datapoints)
    for i in xrange(datapoints):
        cutoff = 1 - i * (1 / datapoints)
        e_b[i], e_s[i] = st.efficiencies(model, data, cutoff)[:, 1]
    if subplot:
        plt.subplot(subplot)
    plt.plot(e_b, e_s, label=label)
    plt.title("Efficiency Curve")
    plt.ylabel("Signal Efficiency")
    plt.xlabel("Background Inefficiency")
    plt.legend(loc=0, fontsize="x-small")

def output_distro(model, data, batch_size=64, labels=None, subplot=None):
    x_train, y_train, x_test, y_test = data
    labels = labels if labels else ["Background", "Signal"]
    output = np.zeros(y_test.shape)
    for i in xrange(int(ceil(x_test.shape[0]/batch_size))):
        output[i*batch_size:(i+1)*batch_size] = model.predict(x_test[i*batch_size:(i+1)*batch_size])

    background = output[(y_test[:] == 1)[:, 0]][:, 1]
    signal = output[(y_test[:] == 1)[:, 1]][:, 1]
    if subplot:
        plt.subplot(subplot)
    plt.hist(background, 50, alpha=0.5, label=labels[0])
    plt.hist(signal, 50, alpha=0.5, label=labels[1])
    plt.title("Model Output")
    plt.xlabel("Percent")
    plt.ylabel("Number of Events")
    plt.legend(loc=0, fontsize="x-small")

def overlay_distro(model1, model2, data, category="signal", batch_size=64, labels=None, subplot=None):
    x_train, y_train, x_test, y_test = data
    labels = labels if labels else ["Model1", "Model2"]
    output1 = np.zeros(y_test.shape)
    for i in xrange(int(ceil(x_test.shape[0] / batch_size))):
        output1[i * batch_size:(i + 1) * batch_size] = model1.predict(x_test[i * batch_size:(i + 1) * batch_size])
    output2 = np.zeros(y_test.shape)
    for i in xrange(int(ceil(x_test.shape[0] / batch_size))):
        output2[i * batch_size:(i + 1) * batch_size] = model2.predict(x_test[i * batch_size:(i + 1) * batch_size])
    category = category.lower()
    col = 1 if category == "signal" else 0 if category == "background" else None
    category = category.capitalize()
    signal1 = output1[(y_test[:] == 1)[:, col]][:, 1]
    signal2 = output2[(y_test[:] == 1)[:, col]][:, 1]
    if subplot:
        plt.subplot(subplot)
    plt.hist(signal1, 50, alpha=0.5, label=labels[0])
    plt.hist(signal2, 50, alpha=0.5, label=labels[1])
    plt.title("Model {} Output".format(category))
    plt.xlabel("Percent")
    plt.ylabel("Number of Events")
    plt.legend(loc=0, fontsize="x-small")

def scatterplot(model1, model2, data, category="signal", labels=None, subplot=None):
    x_train, y_train, x_test, y_test = data
    labels = labels if labels else ["Model1", "Model2"]
    output1 = model1.predict(x_test)
    output2 = model2.predict(x_test)
    category = category.lower()
    col = 1 if category == "signal" else 0 if category == "background" else None
    category = category.capitalize()
    signal1 = output1[(y_test == 1)[:, 1]][:, col]
    signal2 = output2[(y_test == 1)[:, 1]][:, col]
    if subplot:
        plt.subplot(subplot)
    #graph_data = pd.DataFrame({labels[0]: signal1, labels[1]: signal2})
    #sns.regplot(labels[0], labels[1], graph_data, ci=68, line_kws={"linestyle": "--", "color": "seagreen"})
    plt.scatter(signal1, signal2, label=category)
    plt.title("Model {} Output".format(category))
    plt.xlabel("{} Model".format(labels[0]))
    plt.ylabel("{} Model".format(labels[1]))
    plt.legend(loc=0, fontsize="x-small")


if __name__ == "__main__":
    exp = load_experiment("ttHLep/S_1to1")
    exp2 = load_experiment("ttHLep/SL_1to1")
    """label = "Default"
    label2 = "4j"
    s_b(exp, label)
    s_b(exp2, label2)
    auc(exp, label)
    auc(exp2, label2)
    correct(exp, [label+" TTBar", label+" TTHiggs"])
    correct(exp2, [label2+" TTBar", label2+" TTHiggs"])
    accuracy(exp, label)
    accuracy(exp2, label2)"""
    #roc_curve(exp, label="1", subplot=211)
    #roc_curve(exp2, label="2", subplot=212)
    from deep_learning.trainNN import load_model
    model1 = load_model("ttHLep/U_1to1_l1")
    model2 = load_model("ttHLep/S_1to1_l1")
    from keras.optimizers import Adam
    model1.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    h_file, (x_train, y_train, x_test, y_test) = ds.load_dataset("ttHLep", "Unsorted/1to1/transformed")
    data = (x_train, y_train, x_test, y_test)
    _h1_file, (_1, y1, _2, y2) = ds.load_dataset("ttHLep", "Sorted")
    _data = (_1, y1, _2, y2)
    _h2_file, (_1, y3, _2, y4) = ds.load_dataset("ttHLep", "Sorted/1to1")
    TOTAL_BKG, TOTAL_SIG = map(sum, zip(ut.sum_cols(y1), ut.sum_cols(y2)))
    THIS_BKG, THIS_SIG = map(sum, zip(ut.sum_cols(y3), ut.sum_cols(y4)))
    weight = (THIS_SIG/TOTAL_SIG)/sqrt(THIS_BKG/TOTAL_BKG)
    print weight
    override = (THIS_SIG/TOTAL_SIG, THIS_BKG/TOTAL_BKG)
    """one, two = [0,1]
    if one:
        output_distro(model1, data, subplot=221, labels=["Default Bkg", "Default Sig"])
        output_distro(model2, data, subplot=222, labels=["Large DS Bkg", "Large DS Sig"])
        _new_roc_curve(model1, data, subplot=223, label="Default")
        _new_roc_curve(model2, data, subplot=224, label="Large DS")
    elif two:
        overlay_distro(model1, model2, data, category="background", labels=["Default", "Large DS"], subplot=121)
        overlay_distro(model1, model2, data, labels=["Default", "Large DS"], subplot=122)
        plt.gcf().set_size_inches(10, 5)"""
    output_distro(model1, data, subplot=111, labels=["Supernet Bkg", "Supernet Sig"])
    #output_distro(model2, data, subplot=222, labels=["Supernet Bkg", "Supernet Sig"])
    #overlay_distro(model1, model2, data, category="background", labels=["Default", "Supernet"], subplot=223)
    #overlay_distro(model1, model2, data, labels=["Default", "Supernet"], subplot=224)
    plt.tight_layout()
    #plt.show()
    plt.savefig("temp.png")
    print weight*st.significance(model1, data), weight*st.significance(model2, data), st.significance('','',override=override)
    #print model1.evaluate(x_test, y_test)[1], exp2.results[-1].test_accuracy
    print st.confusion_matrix(model1, data), st.confusion_matrix(model2, data)
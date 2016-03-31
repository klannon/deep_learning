import ROOT
from argparse import ArgumentParser

colors = (ROOT.kBlack, ROOT.kRed, ROOT.kGray, ROOT.kBlue, ROOT.kGreen,
          ROOT.kYellow, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange, ROOT.kSpring,
          ROOT.kTeal, ROOT.kAzure, ROOT.kViolet, ROOT.kPink)
names = ("black", "red", "gray", "blue", "green", "yellow", "magenta", "cyan",
         "orange", "spring", "teal", "azure", "violet", "pink")

dColor = dict(zip(names, colors))

parser = ArgumentParser(
    description="Plot graphs of data from .log files in ROOT for up to {} different variables.".format(len(colors)))
parser.add_argument("files", help="List of files to analyze", nargs="+")
parser.add_argument("-c", "--colors", help="A list of color choices in sequence with your file choices. Two colors"+
                    "per file (train and test).", nargs="*", type=str.lower)
args = parser.parse_args()

c = ROOT.TCanvas("c")
mg = ROOT.TMultiGraph()

for nF, filename in enumerate(args.files):
    tr = ROOT.TGraph()
    te = ROOT.TGraph()
    tr.SetPoint(1, 0, 0.0)
    te.SetPoint(1, 0, 0.0)
    tr.SetTitle("Train {}".format(nF+1))
    te.SetTitle("Test {}".format(nF+1))
    try:
        tr.SetLineColor(dColor[args.colors[nF*2]])
        te.SetLineColor(dColor[args.colors[nF*2+1]])
    except (IndexError, TypeError):
        tr.SetLineColor(colors[nF*2])
        te.SetLineColor(colors[nF*2+1])

    with open(filename) as f:
        epochs = 0
        state = [False]*2
        for line in f:
            accs = (line.find("train_y_misclass") >= 0, line.find("test_y_misclass") >= 0)
            for ix, b in enumerate(accs):
                if b:
                    state[ix] = True
            channel_str = None
            if any(accs):
                try:
                    channel_str = line.split(":")[1]
                except:
                    continue
            if channel_str is not None:
                channel_str = channel_str.strip()
                accuracy = 1 - float(channel_str)
                if accs[0] is True:
                    # append to train graph
                    tr.SetPoint(epochs+1, epochs, accuracy)
                elif accs[1] is True:
                    # append to test graph
                    te.SetPoint(epochs+1, epochs, accuracy)
                if all(state):
                    epochs += 1
                    state = [False]*2
    mg.Add(tr)
    mg.Add(te)

mg.Draw("AC")
title = raw_input("Enter the title: ").rstrip()
x_axis = raw_input("Enter the name for the x-axis: ").rstrip()
y_axis = raw_input("Enter the name for the y-axis: ").rstrip()
save = raw_input("Enter the name for the save file: ").rstrip()
c.BuildLegend(0.6, 0.15, 0.9, 0.35)
c.Update()
c.SaveAs("{}.png".format(save))
# the 3v files, the whole shebang(s)
# Whole file: 
    # Train: 1000000
    # Test: 100000
    # Train Fraction: 1 (train_)
    # Train Fraction: 0 (test_)
# train file
python csv.py --pathToData=/scratch365/cdablain/dnn/data/train_highLevel_ttHbb_ttjets_1M.txt --trainFraction=1 --numLabels=2

# test file
python csv.py --pathToData=/scratch365/cdablain/dnn/data/test_highLevel_ttHbb_ttjets_100k.txt --trainFraction=0 --numLabels=2

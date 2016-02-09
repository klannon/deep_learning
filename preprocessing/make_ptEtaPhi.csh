# the 3v files, the whole shebang(s)
# Whole file: 
    # Train: 1700000
    # Test: 90000
    # Train Fraction: 1 (train_all)
    # Train Fraction: 0 (test_all)
# train file
python csv.py --pathToData=/scratch365/cdablain/dnn/data/train_all_ptEtaPhi_ttbar_wjet.txt --trainFraction=1 --numLabels=2

# test file
python csv.py --pathToData=/scratch365/cdablain/dnn/data/test_all_ptEtaPhi_ttbar_wjet.txt --trainFraction=0 --numLabels=2

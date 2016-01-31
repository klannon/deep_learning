# In the HIGGS file, the high-level features are in rows 22-28
# Whole file: 11,000,000
    # Train: 10,890,000
    # Test: 110,000
    # Train Fraction: .99
python csv.py --pathToData=/scratch365/cdablain/dnn/data/HIGGS.csv --trainFraction=.99 --xcolmin=22 --xcolmax=29 --savePath=HIGGS_high_level

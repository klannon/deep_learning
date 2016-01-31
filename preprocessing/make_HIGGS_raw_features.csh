# In the HIGGS file, the raw (low-level) features are in rows 1-21
# Whole file: 11,000,000
    # Train: 10,890,000
    # Test: 110,000
    # Train Fraction: .99        
python csv.py --pathToData=/scratch365/cdablain/dnn/data/HIGGS.csv --trainFraction=.99 --xcolmin=1 --xcolmax=22 --savePath=HIGGS_raw_features

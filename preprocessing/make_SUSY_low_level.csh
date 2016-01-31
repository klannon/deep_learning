# In the SUSY file, the low-level features are in rows 1-8
# Whole file: 5,000,000
    # Train: 4,900,000
    # Test: 100,000
    # Train Fraction: .98    
python csv.py --pathToData=/scratch365/cdablain/dnn/data/SUSY.csv --trainFraction=.98 --xcolmin=1 --xcolmax=9 --savePath=SUSY_low_level

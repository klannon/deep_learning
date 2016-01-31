# In the SUSY file, the high-level features are in rows 9-18
# Whole file: 5,000,000
    # Train: 4,900,000
    # Test: 100,000
    # Train Fraction: .98
python csv.py --pathToData=/scratch365/cdablain/dnn/data/SUSY.csv --trainFraction=0.98 --xcolmin=9 --xcolmax=19 --savePath=SUSY_high_level

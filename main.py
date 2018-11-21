#!/usr/bin/env python
# coding: utf-8

from preprocessor import Data_preprocessor
from BPR import BPR
import pandas as pd

__author__ = "Bo-Syun Cheng"
__email__ = "k12s35h813g@gmail.com"

if __name__ == "__main__":
    data = pd.read_csv('ratings_small.csv')
    dp = Data_preprocessor(data)
    processed_data = dp.preprocess()
    
    bpr = BPR(processed_data)
    bpr.fit()

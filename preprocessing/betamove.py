import numpy as np
import os
import copy
import pickle
import heapq
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing_helper import *

cwd = os.getcwd()
parent_wd = cwd.replace('/preprocessing', '')
raw_data_path = parent_wd + '/preprocessing/processed_data_xy_mode'  
url_data_path = parent_wd + '/raw_data/moonGen_scrape_2016_cp'
left_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016LeftHand.csv'
right_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016RightHand.csv'

LeftHandfeatures = pd.read_csv(left_hold_feature_path, dtype=str)
RightHandfeatures = pd.read_csv(right_hold_feature_path, dtype=str)

# convert features from pd dataframe to dictionary of left and right hand
RightHandfeature_dict = {}
LeftHandfeature_dict = {}
for index in RightHandfeatures.index:
    LeftHandfeature_item = LeftHandfeatures.loc[index]
    LeftHandfeature_dict[(int(LeftHandfeature_item['X_coord']), int(LeftHandfeature_item['Y_coord']))] = np.array(
        list(LeftHandfeature_item['Difficulties'])).astype(int)
    RightHandfeature_item = RightHandfeatures.loc[index]
    RightHandfeature_dict[(int(RightHandfeature_item['X_coord']), int(RightHandfeature_item['Y_coord']))] = np.array(
        list(RightHandfeature_item['Difficulties'])).astype(int)
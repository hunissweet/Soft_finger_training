import torch.nn.functional as F
import copy
import os
import torch
from torchinfo import summary
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from matplotlib.colors import LogNorm
import sklearn.preprocessing
#from directory_tree import display_tree
# Customed Library
import engine ,model_builder,utils
import numpy as np
#import torchinfo
from timeit import default_timer as timer 
import matplotlib.pyplot as plt
import random
import joblib
#display_tree('./')
import pickle
from datetime import datetime




def calibration_function(tar:pd.DataFrame,Cal_list_col:list,Sensor_Cal_list_col:list  ):
    for i in range(len(Cal_list_col)):
        for j in range(len(Cal_list_col[i])):

            col_name=Cal_list_col[i][j]

            tar[col_name]=tar[col_name]-tar[col_name][0]

    # Sensor signal
    Sensor_Cal_list_col=[right_finger_sensor,left_finger_sensor]

    for i in range(len(Sensor_Cal_list_col)):
        for j in range(len(Sensor_Cal_list_col[i])):

            col_name=Sensor_Cal_list_col[i][j]

            tar[col_name]=tar[col_name]-tar[col_name][0] +0*j
    return tar
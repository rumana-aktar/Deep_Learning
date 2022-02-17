# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 01/27/2022
#   
#   Problem: get the 15-digits/chars from an image
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import os; 
#clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()


import pickle, matplotlib.pyplot as plt, numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix , classification_report
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report


## actual digit index in python:  28 x 14
## x1 = [ 82   101   140   160   198   218   257   277   315   335   374   394   121   237   296   305];
## x2 = [ 96   115   154   174   212   232   271   291   329   349   388   408   135   251   310   319]
## y1 = 40; y2 = 68

def readMasks(maskDir, maskFiles):

    MASKS = []
    for i in range(len(maskFiles)):
        digit = cv2.imread(maskDir+maskFiles[i], cv2.IMREAD_GRAYSCALE) 
        digit = np.array(digit);   ones = digit == 255;   digit[ones] = 1; # converts all 255 values to 1
        MASKS.append(digit)
    MASKS = np.array(MASKS)

    return MASKS
    
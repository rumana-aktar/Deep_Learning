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

#maskFiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (dash), 11 (dot), 12 (dot)]
#maskFiles=['0G.png', '1G.png', '2G.png', '3G.png', '4G.png', '5G.png', '6G.png', '7G.png', '8G.png', '9G.png', 'dashG.png', 'dotG.png', 'dot2.png']


def segmentDigits(I, x1, x2, y1, y2, MASK, prediction):

    # print([MASK.shape, prediction.shape])
    # print([x1, x2])
    
    for i in range(len(x1)):
        mask = MASK[prediction[i]] if i < 12 else MASK[i-2]
        block = I[y1:y2, x1[i]:x2[i]]
        pos = mask == 1; block[pos] = 0; #reset the position to 0 where mask==1
        I[y1:y2, x1[i]:x2[i]] = block


    return I
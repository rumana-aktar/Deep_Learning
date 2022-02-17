# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 01/27/2022
#   
#   Problem: Tries to load the saved model for digit classification and check if it can classify unknown data
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import imp
import os; 
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()


import tensorflow as tf, pickle, matplotlib.pyplot as plt, numpy as np, glob
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sn, cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

from GetDigitClassification import classifyDigits
from ReadDigitMasks import readMasks
from SegmentDigits import segmentDigits



# #--------------------- DIGIT IMAGE (for classification) COORDINATES --------------------------------------------
x1 = [ 77,  97, 135, 155, 193, 213, 252, 272, 310, 330, 369, 389, 116, 232, 291];
x2 = [ 99, 119, 157, 177, 215, 235, 274, 294, 332, 352, 391, 411, 138, 254, 313]
y1 = 35; y2 = 72
width, height = 22, 37


# #--------------------- MODEL LOAD --------------------------------------------
filename = 'digitClassification.sav'
cnn = pickle.load(open(filename, 'rb'))


# #--------------------- READ MASK DATASET --------------------------------------------
maskDir = '/Volumes/E/DNCC/raw_data/burn-in dataset/MaskingData/masks/refined2/'
maskFiles=['0G.png', '1G.png', '2G.png', '3G.png', '4G.png', '5G.png', '6G.png', '7G.png', '8G.png', '9G.png', 'dashG.png', 'dotG.png', 'dotG.png', 'dotG.png']
MASKS = readMasks(maskDir, maskFiles)
print(MASKS.shape)

# #--------------------- MASK IMAGE (for masking) COORDINATES --------------------------------------------
mask_x1 = [ 80,    99,   138,   158,   196,   216,   255,   275,   313,   333,   372,   392,   119,   235,   294,   353];
mask_x2 = [ 96,   115,   154,   174,   212,   232,   271,   291,   329,   349,   388,   408,   135,   251,   310,   369]
mask_y1 = 39; mask_y2 = 69; 


# #--------------------- READ IMAGE DATASET --------------------------------------------
no_of_fixed_digits = 7
imgDir = '/Volumes/E/DNCC/raw_data/seq3/'
frameStarts = 'Fr'
imageType = 'png'
files = sorted(glob.glob(imgDir + frameStarts + "*." + imageType)); no_total_frames=len(files)

outDir = '/Volumes/E/DNCC/raw_data/seq3_MaskedOutput2/'
if not os.path.isdir(outDir):
    os.mkdir(outDir) 


# #--------------------- MAKE THE PREDICTION --------------------------------------------
predictionList = []
for i in range(no_total_frames):
    I = cv2.imread(files[i] ,cv2.IMREAD_GRAYSCALE) 
    prediction = classifyDigits(I[:y2, :], x1, x2, y1, y2, width, height, cnn)
    predictionList.append(prediction)

# #--------------------- REFINE THE PREDICTION --------------------------------------------
# take median for first 'no_of_fixed_digits', as they are fixed for each video sequences
predictionList = np.array(predictionList)
med = np.median(predictionList,axis=0)
for i in range(len(predictionList)):
    predictionList[i][0:no_of_fixed_digits] = med[0:no_of_fixed_digits]

# #--------------------- SEE FEW PREDICTION --------------------------------------------
# for i in range(50):
#     print([(i+728), predictionList[i]])

# # #--------------------- SEGMENT THE DEGITS WITH MASKS AND SAVE THE OUTPUT--------------------------------------------
for i in range(no_total_frames):
    I = cv2.imread(files[i] ,cv2.IMREAD_GRAYSCALE)  
    I = segmentDigits(I, mask_x1, mask_x2, mask_y1, mask_y2, MASKS, predictionList[i])
    fname=outDir+'Frame_000'+str(i+728)+'.'+imageType
    cv2.imwrite(fname, I)


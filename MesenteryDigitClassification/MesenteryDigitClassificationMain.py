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

import os; 
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()


import tensorflow as tf, pickle, matplotlib.pyplot as plt, numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sn, cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report


# #--------------------- Read the dataset and reshape y_train --------------------------------------------

img = "/Volumes/E/DNCC/raw_data/burn-in dataset/Data_Digits/1/Frames_000011.png"
new_XX = cv2.imread(img ,cv2.IMREAD_GRAYSCALE)  # convert to array
new_X = cv2.resize(new_XX, (22, 37))

new_XXX = []
new_XXX.append(new_X)   
new_XXX.append(new_X)   
new_XXX = np.array(new_XXX)


new_X= new_XXX.reshape(-1, new_XXX[0].shape[0], new_XXX[0].shape[1], 1)

new_X = new_X / 255.0

# #--------------------- Save and load the model --------------------------------------------
filename = 'digitClassification.sav'
cnn = pickle.load(open(filename, 'rb'))

new_y_predicted = cnn.predict(new_X)
new_y_predicted_labels = [np.argmax(i) for i in new_y_predicted]
print(new_y_predicted_labels)

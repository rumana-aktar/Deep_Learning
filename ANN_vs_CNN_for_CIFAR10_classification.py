# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 01/22/2022
#   
#   Problem: Classify objects (of CIFAR10 DATASET) into 10 classes using Artificial Neural Network
#               and Convolutional Neural Network
#          : ANN accuray 46%; CNN accuracy: 70%   
#          : followed the tutorial: #https://www.youtube.com/watch?v=7HPwo4wnJeA
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


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np

# #--------------------- Read the dataset and reshape y_train --------------------------------------------
(X_train, y_train), (X_test, y_test)= datasets.cifar10.load_data()
print([X_train.shape, X_test.shape])
#print(X_train[0]);#print(y_train.shape);#print(y_train[:5])
y_train = y_train.reshape(-1, ) # make 1d instead of 2d; #print(y_train[:5])
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# normalize the pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0




# #--------------------- artificial neural network --------------------------------------------
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report for Artificial Neural Network: \n", classification_report(y_test, y_pred_classes))


# #--------------------- result --------------------------------------------
# # Classification Report for Artificial Neural Network:  
# #                precision    recall  f1-score   support

# #            0       0.68      0.34      0.45      1000
# #            1       0.54      0.70      0.61      1000
# #            2       0.29      0.58      0.39      1000
# #            3       0.42      0.14      0.22      1000
# #            4       0.49      0.23      0.32      1000
# #            5       0.30      0.62      0.41      1000
# #            6       0.60      0.40      0.48      1000
# #            7       0.58      0.51      0.54      1000
# #            8       0.55      0.70      0.62      1000
# #            9       0.66      0.35      0.46      1000

# #     accuracy                           0.46     10000
# #    macro avg       0.51      0.46      0.45     10000
# # weighted avg       0.51      0.46      0.45     10000


# #--------------------- cnn network--------------------------------------------
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') #soft max is normalizing, 
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)
y_pred = cnn.predict(X_test)
y_pred[:5] 
y_classes = [np.argmax(element) for element in y_pred] #make it 0-9
# print("check first 5 prediction and test values: "); # print(y_classes[:5]); # print(y_test[:5])


y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report Convolutional Neural Network: \n", classification_report(y_test, y_pred_classes))


# #--------------------- result --------------------------------------------
# Classification Report: 
#                precision    recall  f1-score   support

#            0       0.81      0.69      0.75      1000
#            1       0.87      0.78      0.82      1000
#            2       0.69      0.49      0.57      1000
#            3       0.49      0.58      0.53      1000
#            4       0.57      0.75      0.65      1000
#            5       0.62      0.61      0.61      1000
#            6       0.84      0.69      0.76      1000
#            7       0.72      0.78      0.75      1000
#            8       0.81      0.80      0.81      1000
#            9       0.72      0.83      0.77      1000

#     accuracy                           0.70     10000
#    macro avg       0.71      0.70      0.70     10000
# weighted avg       0.71      0.70      0.70     10000
















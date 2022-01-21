# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 01/22/2022
#   
#   Problem: Classify handwritten latters (of MNIST DATASET) into 10 classes using Artificial Neural Network
#               and Convolutional Neural Network
#          : ANN accuray 92%-97%; CNN accuracy: 98%   
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
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sn

# #--------------------- Read the dataset and reshape y_train --------------------------------------------
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
print([X_train.shape, X_test.shape, X_train[0].shape, y_train.shape, y_train[0].shape])

# # #--------------------- normalize the pixel values and flatten X data
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
print([X_train_flattened.shape, X_test_flattened.shape, X_train_flattened[0].shape, y_train.shape, y_train[0].shape])


# #--------------------- artificial neural network --------------------------------------------
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid') ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
# #--------------------- evaluate the model --------------------------------------------
print(model.evaluate(X_test_flattened, y_test))
# #--------------------- prediction --------------------------------------------
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print([y_test[:5], y_predicted_labels[:5]])
# #--------------------- confusion matrix --------------------------------------------
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig("ConfusionMatrix_ANN_without_Hidden_layers.png")
plt.close()

# # # #--------------------- result: ann without hidden layers --------------------------------------------
# # accuracy = 0.925000011920929 ;  # confusion_matrix=
# # [[ 963    0    2    2    0    5    5    2    1    0]
# #  [   0 1118    3    2    0    1    4    2    5    0]
# #  [   8   10  915   20    8    4   12    9   42    4]
# #  [   4    0   17  924    0   26    2   11   18    8]
# #  [   1    1    4    1  915    0   12    3    9   36]
# #  [  10    3    2   32    9  783   14    4   29    6]
# #  [  14    3    5    1    7   15  910    2    1    0]
# #  [   2    7   23    7    8    1    0  936    2   42]
# #  [   7   11    5   25    9   29    8    9  862    9]
# #  [  11    7    1   10   31    7    0   13    5  924]]


# #--------------------- artificial neural network using HIDDEN layers--------------------------------------------
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

# #--------------------- evaluate the model --------------------------------------------
print(model.evaluate(X_test_flattened,y_test))
# #--------------------- prediction --------------------------------------------
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print([y_test[:5], y_predicted_labels[:5]])
# #--------------------- confusion matrix --------------------------------------------
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig("ConfusionMatrix_ANN_Hidden_layers.png")
plt.close()


# # # #--------------------- result: ann with hidden layers --------------------------------------------
# # accuracy = 0.973800003528595;  # confusion_matrix=
# # [[ 971    0    1    4    0    2    0    1    1    0]
# #  [   0 1123    4    1    0    0    2    1    4    0]
# #  [   2    2 1004    9    1    0    1    3    9    1]
# #  [   0    0    3  992    0    4    0    3    3    5]
# #  [   0    0    4    2  940    3    3    2    2   26]
# #  [   2    0    0    9    0  871    4    1    4    1]
# #  [   7    2    3    1    5    7  930    1    2    0]
# #  [   2    4   11    5    2    0    0  989    7    8]
# #  [   2    0    4    5    3    7    4    2  944    3]
# #  [   3    5    0    9    5    4    1    4    4  974]]




# #--------------------- artificial neural network using hidden layers with flatten network--------------------------------------------
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

# #--------------------- evaluate the model --------------------------------------------
print(model.evaluate(X_test,y_test))
# #--------------------- prediction --------------------------------------------
y_predicted = model.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print([y_test[:5], y_predicted_labels[:5]])
# #--------------------- confusion matrix --------------------------------------------
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig("ConfusionMatrix_ANN_Hidden_Flatten_layers.png")
plt.close()


# # # #--------------------- result: ann with hidden layers --------------------------------------------
# # accuracy = 0.9775999784469604;  # confusion_matrix=
# # [[ 975    0    1    1    0    0    1    1    1    0]
# #  [   0 1127    2    0    0    2    2    1    1    0]
# #  [   5    5 1001    5    0    1    1    7    6    1]
# #  [   0    0    1  983    0    9    1    6    6    4]
# #  [   0    1    3    1  962    0    3    3    1    8]
# #  [   2    0    0    5    0  874    4    3    3    1]
# #  [   3    2    1    1    2    6  940    0    3    0]
# #  [   1    6    7    3    0    0    0 1004    3    4]
# #  [   6    0    4    8    4    8    1    5  931    7]
# #  [   2    4    0    6    8    2    0    7    1  979]]


# #--------------------- cnn network--------------------------------------------
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
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
y_predicted = cnn.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print([y_test[:5], y_predicted_labels[:5]])
# #--------------------- confusion matrix --------------------------------------------
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig("ConfusionMatrix_CNN.png")
plt.close()

# # # #--------------------- result: Classification Report Convolutional Neural Network:  --------------------------------------------
# # accuracy = 0.9888 ; # confusion_matrix=
# [[ 974    1    0    0    1    0    0    4    0    0]
#  [   0 1127    0    6    0    0    0    1    1    0]
#  [   3    0 1009    0    3    0    0   15    2    0]
#  [   1    0    0 1003    0    1    0    4    1    0]
#  [   0    0    0    0  976    0    1    2    0    3]
#  [   1    0    0   11    0  874    1    2    1    2]
#  [   5    2    1    0    3    4  941    0    2    0]
#  [   0    1    0    0    0    0    0 1027    0    0]
#  [   3    0    1    0    0    0    0    3  965    2]
#  [   1    0    0    0    6    3    0    6    1  992]]

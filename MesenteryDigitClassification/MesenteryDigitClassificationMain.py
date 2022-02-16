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


import tensorflow as tf, pickle
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sn, cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report


# #--------------------- Read the dataset and reshape y_train --------------------------------------------
# # try loading lik mesentery blur

DATADIR = "/Volumes/E/DNCC/raw_data/burn-in dataset/Data_Digits"
CATEGORIES = ["0", "1", "2",  "3", "4", "5", "6", "7", "8", "9"]

img = "/Volumes/E/DNCC/raw_data/burn-in dataset/Data_Digits/0/Frames_000011.png"
new_XX = cv2.imread(img ,cv2.IMREAD_GRAYSCALE)  # convert to array
new_X = cv2.resize(new_XX, (22, 37))

# DATADIR = "/Volumes/E/DNCC/raw_data/seq5/Data_Digits/smallset"
# CATEGORIES = ["0", "1", "2"]


training_data = []
for category in CATEGORIES:  # do dogs and cats

    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

    #print(path)

    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            #print([path, class_num])
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (22, 37))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            print(e)
            pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

# #--------------------- Shuffle the dataset --------------------------------------------
import random
random.shuffle(training_data)

# #--------------------- Generate X and y--------------------------------------------
X, y = [],[]
for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X)   
y = np.array(y) 
new_XXX = []
new_XXX.append(new_X)   
new_XXX.append(new_X)   
new_XXX = np.array(new_XXX)


print([X.shape, X[0].shape])
X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
new_X= new_XXX.reshape(-1, new_XXX[0].shape[0], new_XXX[0].shape[1], 1)

print([X.shape, X[0].shape])


# #--------------------- Generate train and test and y--------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print("\n\n[X_test.shape, X_test.shape]:  ", end ="")
print([X_test.shape, X_test[0].shape, new_X.shape])

# # #--------------------- normalize the pixel values and flatten X data
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train_flattened = X_train.reshape(len(X_train), 37*22)
X_test_flattened = X_test.reshape(len(X_test), 37*22)

new_X = new_X / 255.0
new_X_flattened = new_X.reshape(len(new_X), 37*22)

#print([X_train_flattened.shape, X_test_flattened.shape, X_train_flattened[0].shape, y_train.shape, y_train[0].shape])

print("\n[X_test_flattened.shape, X_test_flattened[0].shape]:  ", end ="")
print([X_test_flattened.shape, X_test_flattened[0].shape, X_test_flattened])
print("\n\n")



# # #--------------------- cnn network--------------------------------------------
# cnn = models.Sequential([
#     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(37, 22, 1)), 
#     layers.MaxPooling2D((2, 2)),
    
#     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax') #soft max is normalizing, 
# ])

# cnn.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# cnn.fit(X_train, y_train, epochs=5)


# # #--------------------- Save and load the model --------------------------------------------
# filename = 'digitClassification.sav'
# pickle.dump(cnn, open(filename, 'wb'))

# #--------------------- Save and load the model --------------------------------------------
filename = 'digitClassification.sav'
cnn = pickle.load(open(filename, 'rb'))



cnn.evaluate(X_test, y_test)
y_predicted = cnn.predict(X_test)

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print([y_test[:5], y_predicted_labels[:5]])

new_y_predicted = cnn.predict(new_X)
new_y_predicted_labels = [np.argmax(i) for i in new_y_predicted]
print(new_y_predicted_labels)



# # #--------------------- confusion matrix --------------------------------------------
# cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
# print(cm)

# plt.figure(figsize = (10,7))
# sn.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.savefig("ConfusionMatrix_Mesentery_Digits.png")
# plt.close()

# print(classification_report(y_test, y_predicted_labels))


# # # #--------------------- result: Classification Report Convolutional Neural Network:  --------------------------------------------
# Epoch 1/5
# 113/113 [==============================] - 2s 11ms/step - loss: 0.8083 - accuracy: 0.7774
# Epoch 2/5
# 113/113 [==============================] - 1s 11ms/step - loss: 0.0534 - accuracy: 0.9858
# Epoch 3/5
# 113/113 [==============================] - 1s 11ms/step - loss: 0.0320 - accuracy: 0.9925
# Epoch 4/5
# 113/113 [==============================] - 1s 11ms/step - loss: 0.0182 - accuracy: 0.9950
# Epoch 5/5
# 113/113 [==============================] - 1s 12ms/step - loss: 0.0129 - accuracy: 0.9967
# 56/56 [==============================] - 0s 5ms/step - loss: 0.0090 - accuracy: 0.9977
# [array([1, 0, 2, 7, 1]), [1, 0, 2, 7, 1]]
# tf.Tensor(
# [[168   0   0   0   0   0   0   0   2   0]
#  [  0 167   0   0   0   0   0   0   0   0]
#  [  0   0 200   0   0   0   0   0   0   0]
#  [  0   0   0 174   0   0   0   0   0   0]
#  [  0   0   0   0 179   0   0   0   0   0]
#  [  0   0   0   0   0 168   0   0   0   0]
#  [  0   0   0   0   0   0 195   0   0   0]
#  [  0   0   0   0   0   0   0 161   0   2]
#  [  0   0   0   0   0   0   0   0 171   0]
#  [  0   0   0   0   0   0   0   0   0 182]], shape=(10, 10), dtype=int32)
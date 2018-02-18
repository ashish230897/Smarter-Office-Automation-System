'''
This code is used to classify the products in an office inventory using their images
'''
import numpy as np
import glob
import os
import cv2
import json
import math
import sys
import h5py
import sklearn
from sklearn.model_selection import train_test_split
import random
from random import shuffle
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,BatchNormalization
from keras.layers.convolutional import Convolution2D,ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import callbacks 
import matplotlib.pyplot as plt

##Loading .npy files of training data##
x = np.load("")
y = np.load("")

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.05,random_state = 0)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.05,random_state = 0)
X_train = X_train/255
X_test = X_test/255
X_valid = X_valid/255

def CancerModeling():
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
    model.add(Convolution2D(32, 3, 3,activation= 'relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3,activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6,activation='softmax'))
              
    model.compile(loss='categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])
    
    return model

model = CancerModeling()
model.fit(X_train, y_train, validation_data=(X_valid,y_valid), nb_epoch=4, batch_size=32, verbose=1)
model.summary()
model.save('inventory_management.h5')
scores = model.evaluate(X_test,y_test,verbose=0)
print("Accuracy ",(scores[1]))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

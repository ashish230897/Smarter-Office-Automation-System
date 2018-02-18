import tensorflow as tf
import glob
import os
import matplotlib
import cv2
import math
import json
import sys
import h5py
import matplotlib.pyplot as plt
import random
from random import shuffle
import numpy as np
from keras import layers
from keras import losses
from keras import metrics
from keras.layers import Input, Dropout, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.models import model_from_json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import savefig


import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("loaded model from disk")

img = cv2.imread('img5.jpg')
img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.expand_dims(img, axis = 0)
img = np.expand_dims(img, axis = 3)


Y_pred = model.predict(img)

i = 0

plt.imshow(img[i, :, :, 0])
plt.show()
plt.imshow(Y_pred[i, :, :, 0], cmap = 'hot')
savefig("out.png")

oldrange = np.amax(Y_pred[i, :, :, 0]) - np.amin(Y_pred[i, :, :, 0])
if oldrange == 0:
	oldrange = 1
newrange = 0.035
Y_pred[i, :, :, 0] = ( (Y_pred[i, :, :, 0] - np.amin(Y_pred[i, :, :, 0]) )*newrange)/oldrange

print(np.sum(Y_pred[i, :, :, 0]))


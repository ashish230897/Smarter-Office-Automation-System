import tensorflow as tf
import glob
import os
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
from keras.layers import Input, Dropout, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Concatenate, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.models import model_from_json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def crowdnet(input_shape = (128, 128, 1)):
	
	X_input = Input(input_shape)

	X = ZeroPadding2D((1, 1))(X_input)
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1_1')(X)
	X = Activation('relu', name = 'relu1_1')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1_2')(X)	
	X = Activation('relu', name = 'relu1_2')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1_3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1_3')(X)	
	X = Activation('relu', name = 'relu1_3')(X)
	X = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool1')(X)
	
	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv2_1')(X)	
	X = Activation('relu', name = 'relu2_1')(X)
	X = ZeroPadding2D((1, 1))(X)	
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv2_2')(X)		
	X = Activation('relu', name = 'relu2_2')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2_3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv2_3')(X)		
	X = Activation('relu', name = 'relu2_3')(X)
	X = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool2')(X)
	
	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv3_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv3_1')(X)	
	X = Activation('relu', name = 'relu3_1')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv3_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv3_2')(X)	
	X = Activation('relu', name = 'relu3_2')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv3_3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv3_3')(X)
	X = Activation('relu', name = 'relu3_3')(X)
	X = ZeroPadding2D((1, 1))(X)	
	X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv3_4', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv3_4')(X)
	X = Activation('relu', name = 'relu3_4')(X)
	
	
	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(81, (3, 3), strides = (1, 1), name = 'conv4_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv4_1')(X)	
	X = Activation('relu', name = 'relu4_1')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(81, (3, 3), strides = (1, 1), name = 'conv4_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv4_2')(X)	
	X = Activation('relu', name = 'relu4_2')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(81, (3, 3), strides = (1, 1), name = 'conv4_3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv4_3')(X)
	X = Activation('relu', name = 'relu4_3')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(81, (3, 3), strides = (1, 1), name = 'conv4_4', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv4_4')(X)
	X = Activation('relu', name = 'relu4_4')(X)
	
	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv5_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv5_1')(X)	
	X = Activation('relu', name = 'relu5_1')(X)	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv5_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv5_2')(X)	
	X = Activation('relu', name = 'relu5_2')(X)
	

	X = UpSampling2D(size = (2, 2))(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv6_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv6_1')(X)	
	X = Activation('relu', name = 'relu6_1')(X)	
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv6_2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv6_2')(X)	
	X = Activation('relu', name = 'relu6_2')(X)
	

	X = UpSampling2D(size = (2, 2))(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv7_1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv7_1')(X)	
	X = Activation('relu', name = 'relu7_1')(X)
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(1, (3, 3), strides = (1, 1), name = 'conv7_2', kernel_initializer = glorot_uniform(seed=0))(X)
	
	# Create model
	model = Model(inputs = X_input, outputs = X, name= 'crowdnet')

	return model

def get_list():
	train_file1 = '/home/ashish/deeplearning/heat_maps/new_model1/train_0.h5'
	train_file2 = '/home/ashish/deeplearning/heat_maps/new_model1/train_7000.h5'
	train_file3 = '/home/ashish/deeplearning/heat_maps/new_model1/train_14000.h5'

	contents = [train_file1, train_file2, train_file3]
	return contents

contents = get_list()

model = crowdnet(input_shape = (128, 128, 1))
model.load_weights("model.h5")


learning_rate = 0.00000000001
epoch = 4
decay_put = learning_rate/epoch
optimizer_put = Adam(lr = learning_rate, decay = decay_put)

model.compile(optimizer = optimizer_put, loss = losses.mean_squared_error, metrics = [metrics.mse, metrics.mae])

##The model is saved everytime the validation accuracy improves to prevent failure after long intervals of training##
filepath = 'model.h5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

with h5py.File('/home/ashish/deeplearning/heat_maps/new_model1/valid_0.h5', 'r') as hf:
	X_valid = np.array(hf.get('data'))
	Y_valid = np.array(hf.get('label'))

for k in range(Y_valid.shape[0]):
	oldrange = np.amax(Y_valid[k]) - np.amin(Y_valid[k])
	if oldrange == 0:
		oldrange = 1	
	newrange = 255.0
	Y_valid[k] = ( (Y_valid[k] - np.amin(Y_valid[k]) )*newrange)/oldrange


X_valid = np.expand_dims(X_valid, axis = 3)
Y_valid = np.expand_dims(Y_valid, axis = 3)


for num in range(epoch):
	print("ONGOING EPOCH IS " + str(num))
	for i in range(3):
		with h5py.File(contents[i], 'r') as hf:
			X_train = np.array(hf.get('data'))
			Y_train = np.array(hf.get('label'))

			##Scaling the output pixel values of ground truth between 0 to 255##
			for j in range(Y_train.shape[0]):
				oldrange = np.amax(Y_train[j]) - np.amin(Y_train[j])
				if oldrange == 0:
					oldrange = 1
				newrange = 255.0
				Y_train[j] = ( (Y_train[j] - np.amin(Y_train[j]) )*newrange)/oldrange	

			X_train = np.expand_dims(X_train, axis = 3)
			Y_train = np.expand_dims(Y_train, axis = 3)
			model.fit(X_train, Y_train, batch_size=8, epochs = 1,  validation_data = (X_valid, Y_valid), callbacks = callbacks_list)
			
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


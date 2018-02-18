'''
The following code loads the trained weights. It classifeies the image received from the database
into one of the 6 classes mentioned eg raspberry pi, beagle bone etc.
'''
import numpy as np
import glob
import os
import cv2
import json
import math
import sys
import h5py
from imutils.video import FPS
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
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

storage_client = storage.Client.from_service_account_json('nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json')
cred = credentials.Certificate('nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json')
firebase_admin.initialize_app(cred,{ 'databaseURL':'https://nvidiahack-d0e6d.firebaseio.com'})
bucket = storage_client.get_bucket('nvidiahack-d0e6d.appspot.com')

def on_connect(client,userdata,flags,rc):
	print("Connected With Result"+str(rc))

client = mqtt.Client()
client.connect('13.126.45.185')
client.subscribe("/newInventaryRequests")


def inventory_management():
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

model = amazon_go()
model.load_weights("inventory_management.h5")

items = ["Arduino", "BeagleBone", "HackRFOne", "HdmiDisplay", "RaspberryPi3", "WeemosD1"]

def on_message(client,userdata,msg):
	msg_pay = msg.payload.decode("utf-8")
	print(msg.topic + str(msg_pay))
	blob= bucket.blob(str(msg_pay) + '.png')
	blob.download_to_filename(str(msg_pay) +'.png')
	frame = cv2.imread(str(msg_pay) +'.png')
	img = cv2.resize(frame, (64, 64), interpolation = cv2.INTER_AREA)
	img = np.expand_dims(img, axis = 0)	
	Y_pred = model.predict(img)
	index = np.argmax(Y_pred, axis = 1)[0]	
	product = items[index]
	ref = db.reference('/4z9Wl4iadoWQ2P8DG7Vg67TuVL72/Object')
	body= json.loads(ref.get())
	body.count = 1 + body.count
	ref.update({ str(body.count) : product,"count": body.count})

	
client.on_message = on_message
client.loop_forever()


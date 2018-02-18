from __future__ import absolute_import
import imutils
from imutils.video import FPS
import time
import os
import cv2
import sys
import binascii
import marshal
import urllib.request
import line_profiler
import glob
import math
import json
import h5py
import random
from random import shuffle
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from keras.utils.generic_utils import custom_object_scope
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import Progbar
from PIL import Image, ImageDraw, ImageFont
from random import shuffle
from keras.models import model_from_json
from matplotlib.pyplot import savefig
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import requests as req

cred = credentials.Certificate('nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json')
firebase_admin.initialize_app(cred,{ 'databaseURL':'https://nvidiahack-d0e6d.firebaseio.com/'})
client = storage.Client.from_service_account_json('nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json')
bucket = client.get_bucket('nvidiahack-d0e6d.appspot.com')
url='http://192.168.1.9:8080/shot.jpg?rnd=850498'
image_name = 0

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .3):
	"""Filters YOLO boxes by thresholding on object and class confidence.

	Arguments:
	box_confidence -- tensor of shape (19, 19, 5, 1)
	boxes -- tensor of shape (19, 19, 5, 4)
	box_class_probs -- tensor of shape (19, 19, 5, 80)
	threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

	Returns:
	scores -- tensor of shape (None,), containing the class probability score for selected boxes
	boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
	classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

	Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
	For example, the actual output size of scores would be (10,) if there are 10 boxes.
	"""

	##Compute box scores##
	box_scores = box_confidence*box_class_probs

	##Find the box_classes thanks to the max box_scores, keep track of the corresponding score##
	box_classes = tf.argmax(box_scores, axis = -1)
	box_class_scores = tf.reduce_max(box_scores, axis = -1)

	##Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the##
	##same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)##
	filtering_mask = box_class_scores >= threshold

	##Apply the mask to scores, boxes and classes##
	scores = tf.boolean_mask(box_class_scores, filtering_mask)
	boxes = tf.boolean_mask(boxes, filtering_mask)
	classes = tf.boolean_mask(box_classes, filtering_mask)

	return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
	"""
	Applies Non-max suppression (NMS) to set of boxes

	Arguments:
	scores -- tensor of shape (None,), output of yolo_filter_boxes()
	boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
	classes -- tensor of shape (None,), output of yolo_filter_boxes()
	max_boxes -- integer, maximum number of predicted boxes you'd like
	iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

	Returns:
	scores -- tensor of shape (, None), predicted score for each box
	boxes -- tensor of shape (4, None), predicted box coordinates
	classes -- tensor of shape (, None), predicted class for each box

	Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
	function will transpose the shapes of scores, boxes, classes. This is made for convenience.
	"""

	max_boxes_tensor = K.variable(max_boxes, dtype='int32')     ##tensor to be used in tf.image.non_max_suppression()##
	K.get_session().run(tf.variables_initializer([max_boxes_tensor])) ##initialize variable max_boxes_tensor##

	##Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep##
	nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

	##Use K.gather() to select only nms_indices from scores, boxes and classes##
	scores = K.gather(scores, nms_indices)
	boxes = K.gather(boxes, nms_indices)
	classes = K.gather(classes, nms_indices)

	return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.5, iou_threshold=.2): #iou_thresh : 0.5 score_thresh : 0.6
	"""
	Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

	Arguments:
	yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
		    box_confidence: tensor of shape (None, 19, 19, 5, 1)
		    box_xy: tensor of shape (None, 19, 19, 5, 2)
		    box_wh: tensor of shape (None, 19, 19, 5, 2)
		    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
	image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
	max_boxes -- integer, maximum number of predicted boxes you'd like
	score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
	iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

	Returns:
	scores -- tensor of shape (None, ), predicted score for each box
	boxes -- tensor of shape (None, 4), predicted box coordinates
	classes -- tensor of shape (None,), predicted class for each box
	"""

	##Retrieve outputs of the YOLO model##
	box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

	##Convert boxes to be ready for filtering functions##
	boxes = yolo_boxes_to_corners(box_xy, box_wh)

	##Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold##
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)

	##Scale boxes back to original image shape.##
	boxes = scale_boxes(boxes, image_shape)

	##Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold##
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = iou_threshold)

	return scores, boxes, classes

def main():
	global image_name
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")
	print("loaded heta_map model from disk")

	sess = K.get_session()

	class_names = read_classes("model_data/pascal_classes.txt")
	anchors = read_anchors("model_data/yolo_anchors.txt")

	yolo_model = load_model("model_data/yolo.h5")

	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


	time.sleep(1.0)
	fps = FPS().start()

	#cap = cv2.VideoCapture(1)
	imgResp = urllib.request.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	frame = cv2.imdecode(imgNp,-1)
	#ret,frame = cap.read()

	image_shape = (float(frame.shape[0]), float(frame.shape[1]))

	scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
	count = np.zeros(10, dtype = int)
	cnt = -1
	heat_cnt = 0
	while True:
		if heat_cnt == 1080:
			image_name += 1
			time.sleep(3.0)
			
			img = cv2.resize(frame, (128, 128), interpolation = cv2.INTER_AREA)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = np.expand_dims(img, axis = 0)
			img = np.expand_dims(img, axis = 3)

			Y_pred = model.predict(img)

			plt.imshow(Y_pred[0, :, :, 0], cmap = 'hot')
			savefig("out.png")
			##Below is the heatmap that is to be updated on firebase##
			blob = bucket.blob('main'+str(image_name)+'.png')
			file_to_upload = open('out.png', 'rb')
			blob.upload_from_file(file_to_upload)
			file_to_upload.close()
			
			oldrange = np.amax(Y_pred[0, :, :, 0]) - np.amin(Y_pred[0, :, :, 0])
			if oldrange == 0:
				oldrange = 1
			newrange = 0.035
			Y_pred[0, :, :, 0] = ( (Y_pred[0, :, :, 0] - np.amin(Y_pred[0, :, :, 0]) )*newrange)/oldrange
			##Below count is the count of people in the room##
					
			person_count = np.sum(Y_pred[0, :, :, 0])
			db.reference('/Heatmap').update({ 'numberOfPeople': str(person_count)})			
			print(person_count)
			heat_cnt = 0
		
		else:
			if cnt == 9:
				counts = np.bincount(count)
				print('Number of persons are ' + str(np.argmax(counts)))
				if(np.argmax(counts)):
					r=req.post('http://192.168.1.2:443/lightNumber',data='{"ac":true}',verify=False)
					print(r.text)
				else:
					r=req.post('http://192.168.1.2:443/lightNumber',data='{"ac":false}',verify=False)
					print(r.text)
				cnt = 0
			else : cnt += 1

			#ret, frame = cap.read()
			imgResp = urllib.request.urlopen(url)
			imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
			frame = cv2.imdecode(imgNp,-1)


			image, image_data = preprocess_image(frame, model_image_size = (416, 416))

			out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input : image_data,  K.learning_phase(): 0})

			colors = generate_colors(class_names)

			count[cnt] = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
			cv2.imshow('frame', np.array(image)[:, :, ::-1])

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			fps.update()
			heat_cnt += 1

	fps.stop()
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	cv2.destroyAllWindows()

main()


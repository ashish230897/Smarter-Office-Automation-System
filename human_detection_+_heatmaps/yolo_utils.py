import colorsys
import imghdr
import os
import random
from keras import backend as K
import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_classes(classes_path):
    ##f acts as the file object and f.readlines() puts all the lines as a list in class_names##
    with open(classes_path) as f:
        class_names = f.readlines()
    ##strip() function of string is used to remove trailing and leading whitespaces##
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        ##The string read in anchors is splitted whenever a , occurs and is stored as a list##
        anchors = [float(x) for x in anchors.split(',')]
        ##The list is converted  to array of any number of rows and two columns##
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    ##colorsys is used to convert between different color systems##
    ##lambda is an anonymous function that takes as aruguments next to the lambda keyword and inputs the arguments to the expression provided alongside##
    ##map maps all the arguments of the provided list to the function##
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    ##stack the the list along the row, returns a tensor##
    image_dims = K.stack([height, width, height, width])
    ##combine the four rows into 1 row with four columns, returns a tensor##
    image_dims = K.reshape(image_dims, [1, 4])
    ##WHY DO WE MULTIPLE BY IMAGE DIMENSIONS ?##
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img, model_image_size):
    #image_type = imghdr.what(img_path)
    #image = Image.open(img_path)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ##image.resize is PIL function that resizes image dimensions with original dimensions provided in reverse order##
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    ##Normalizing##
    image_data /= 255.
    ##Add the first dimension for number of examples##
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    ##Below line decides what the font is and what it's size is.##
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300  ##does floor division##
    cnt = 0
    heat = np.zeros((image.size[1], image.size[0]), dtype = 'float32')
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        if predicted_class == 'person':
            cnt += 1
            label = '{} {:.2f}'.format(predicted_class, score)

            ##ImageDraw is a module to do 2d graphics on Image objects, it creates an object that can be used to draw on given image##
            draw = ImageDraw.Draw(image)
            ##Returns the size of text string label in pixels##
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom), sep = ' ', end = '\n', flush = True)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
    #print("Nmuber of persons are " + str(cnt), flush = True)
    return cnt

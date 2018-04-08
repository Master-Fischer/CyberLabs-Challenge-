# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:49:31 2018

@author: Lucca
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="caminho para a imagem")
args = vars(ap.parse_args())

model = load_model("BeersNSodas2.model")

image = load_img(args['image'],target_size=(100,100))
image = img_to_array(image)
image = np.array(image,dtype="float")/255.

image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])

pred = model.predict(image)

(Soda, Beer) = pred[0]

if np.argmax(pred[0]) == 0:
    print("Soda!")
elif np.argmax(pred[0]) == 1:
    print("Beer!")

   

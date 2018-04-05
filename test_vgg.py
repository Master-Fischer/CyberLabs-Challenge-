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


vgg16 = VGG16(weights=None)

vgg_weights = vgg16.get_weights()

model = Sequential()
for layer in vgg16.layers:
    model.add(layer)
model.layers.pop()

model.add(Dense(2,activation='softmax'))


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

image = cv2.imread(args['image'])
image = cv2.resize(image, (224,224))  #Melhores resultados do que (28,28)
image = np.array(image, dtype='float')/255.0 #<----- NORMALIZE!!
image = image.reshape(1, image.shape[0],image.shape[1], image.shape[2])

w = np.load('weights_array.npy')
model.set_weights(w)

pred = model.predict(image)

(Soda, Beer) = pred[0]

if np.argmax(pred[0]) == 0:
    print("Refrigerante!")
else:
    print("Cerveja!")
   

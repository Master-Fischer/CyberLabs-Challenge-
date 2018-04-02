# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:33:06 2018

@author: Lucca
"""
import tables
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam,RMSprop
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_shift
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_process_img(directory, data, labels):
    folders = os.listdir(directory)
    
    for folder in folders:
        images = os.listdir(directory+"\\"+folder)
        for img in images:
            image = load_img(directory+"\\"+folder+"\\"+img,target_size=(224,224))
            image = img_to_array(image)
            #image = preprocess_input(image)
            data.append(image)
            if folder == "Beers":
                labels.append(1)
            else:
                labels.append(0)

            


data = []
labels = []

load_process_img("Dataset", data, labels)           

labels = np.array(labels)
data = np.array(data, dtype='float')/255.0 #<----- NORMALIZE!!

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)
trainY = to_categorical(trainY,num_classes=2)
testY = to_categorical(testY, num_classes=2)

bs = 10

vgg16 = VGG16(weights='imagenet')
print("vgg: ")
vgg_weights = vgg16.get_weights()

model = Sequential()	
for layer in vgg16.layers:
    model.add(layer)


model.layers.pop()
    

for layer in model.layers[6:]:
    layer.trainable = False

	
model.add(Dense(2,activation='softmax'))

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

aug = ImageDataGenerator(rotation_range=30,height_shift_range=0.3,width_shift_range=0.2, zoom_range=0.5, shear_range=0.4)

train_batches = aug.flow(trainX,trainY,batch_size=bs,shuffle=True)
test_batches = aug.flow(testX,testY,batch_size=bs,shuffle=True)

model.fit_generator(train_batches,epochs = 200,validation_data=test_batches ,steps_per_epoch=100)	

print(model.evaluate(testX,testY))

w = model.get_weights()
np.save("weights_array2",w)

print(model.predict(image))






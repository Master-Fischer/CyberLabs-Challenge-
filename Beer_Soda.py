# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:44:32 2018

@author: Lucca
"""

import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_shift
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

#Cria CNN usando arquitetura LeNet
def LeNet(width,height,depth):

    shape = (width,height, depth)
    
    model = Sequential()
    
    model.add(Conv2D(20,(5,5),input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(50,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    
    return model

#Pega as imagens redimensiona elas 
def load_process_img(directory, data, labels):
    folders = os.listdir(directory)
    
    for folder in folders:
        images = os.listdir(directory+"\\"+folder)
        for img in images:
            image = load_img(directory+"\\"+folder+"\\"+img,target_size=(100,100))
            image = img_to_array(image)
            data.append(image)
            if folder == "Beers":
                labels.append(1)
            else:
                labels.append(0)

 

data = []
labels = []

load_process_img("All", data, labels)           

labels = np.array(labels)
data = np.array(data, dtype='float')/255.0 #<----- NORMALIZAR!!

(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.25)

bs = 10
epochs = 400

#One-hot encoding dos labels []
train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

#Inicializa Modelo
model = LeNet(100,100,3)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Inicializa o Image augmentor do keras para dar mais variedade ao dataset
aug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.2, zoom_range=0.4,horizontal_flip=True, fill_mode="nearest")

#Prepara o dataset auumentado as imagens
train_images = aug.flow(train_data,train_labels, batch_size=bs)
test_images = aug.flow(test_data,test_labels, batch_size=bs)

#Treina o modelo
model.fit_generator(train_images,validation_data =test_images,steps_per_epoch=100, epochs=epochs)

score = model.evaluate_generator(test_images)
#Mostra accuracy final
print(score[1])
model.save("BeersNSodas2.model")

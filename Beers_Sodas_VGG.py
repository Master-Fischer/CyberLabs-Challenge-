# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:33:06 2018

@author: Lucca
"""
import random
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_shift
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

#Funcao que carrega as fotos e dimensiona elas
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

            

bs = 10
epochs = 200

#Carrega todas as fotos,normaliza e separa em train e test set
data = []
labels = []

load_process_img("Dataset", data, labels)           

labels = np.array(labels)
data = np.array(data, dtype='float')/255.0 #<----- NORMALIZE!!

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

#One-hot encoding
trainY = to_categorical(trainY,num_classes=2)
testY = to_categorical(testY, num_classes=2)

#Carrega Modelo VGG16 do keras com weights treinados da imagenet
vgg16 = VGG16(weights='imagenet')
print("vgg: ")
vgg_weights = vgg16.get_weights()

#Transforma modelo em Sequential()
model = Sequential()	
for layer in vgg16.layers:
    model.add(layer)

#Remove ultima camada (VGG original foi treinado para clasificar 1000 clases)
model.layers.pop()

#Define quais camadas seram treinadas
for layer in model.layers[6:]:
    layer.trainable = False

#Adciona ultima camada dense com 2 outputs (Cervejas e refrigerantes)
model.add(Dense(2,activation='softmax'))

#compila modelo
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Inicializa o aumentador de imagens
aug = ImageDataGenerator(rotation_range=30,height_shift_range=0.3,width_shift_range=0.2, zoom_range=0.5, shear_range=0.4)
train_batches = aug.flow(trainX,trainY,batch_size=bs,shuffle=True)
test_batches = aug.flow(testX,testY,batch_size=bs,shuffle=True)

#Treina o modelo
model.fit_generator(train_batches,epochs = epochs,validation_data=test_batches ,steps_per_epoch=100)	

#Mostra Score do modelo
print(model.evaluate(testX,testY))

#Salva os weights do modelo
w = model.get_weights()
np.save("weights_array",w)




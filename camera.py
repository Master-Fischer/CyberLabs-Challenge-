
import numpy as np
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from keras.models import Sequential
from keras.models import load_model
import time
import imutils
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam,RMSprop
from keras.preprocessing.image import load_img, img_to_array


def show_webcam(model):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        
        frame = vs.read()
        #frame = imutils.resize(frame,width=224,height=224)
        img = cv2.resize(frame,(224,224))
        img = img_to_array(img)
		
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)
	        	
        
        pred = model.predict(img)
        #(Background ,Beer, Soda) 
        result = pred[0]
        if np.argmax(result) == 0:
            clas = "Beer"
        elif np.argmax(result) == 1:
            clas = "Soda"
        else:
            clas = "Nada"
        proba = str(result[np.argmax(result)])
        print(clas)
        
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27: 
            vs.stop()
            break  # esc to quit
    cv2.destroyAllWindows()



w = np.load("camera_weights.npy")

vgg16 = VGG16(weights=None)

model = Sequential()	
for layer in vgg16.layers:
    model.add(layer)

model.layers.pop()
model.add(Dense(3,activation='softmax'))

model.set_weights(w)

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
show_webcam(model)




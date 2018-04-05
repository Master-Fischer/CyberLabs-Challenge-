
import numpy as np
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from keras.models import Sequential
from keras.models import load_model
import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam,RMSprop
from keras.preprocessing.image import load_img, img_to_array


def show_webcam(model):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        
        frame = vs.read()
        imgProcess = cv2.resize(frame, (224,224))
        imgProcess = img_to_array(imgProcess)
        imgProcess = np.array(imgProcess, dtype='float')/255.0 #<----- NORMALIZE!!
        #imgProcess = preprocess_input(imgProcess)		
        imgProcess = imgProcess.reshape(1,imgProcess.shape[0],imgProcess.shape[1],imgProcess.shape[2])
        
        pred = model.predict(imgProcess)
        (Soda, Beer) = pred[0]
        clas = "Beer" if Beer > Soda else "Soda"
        proba = Beer if Beer > Soda else Soda
        print(clas + " " + str(proba))
        
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27: 
            vs.stop()
            break  # esc to quit
    cv2.destroyAllWindows()



w = np.load("weights_array.npy")

vgg16 = VGG16(weights=None)

model = Sequential()	
for layer in vgg16.layers:
    model.add(layer)

model.layers.pop()
model.add(Dense(2,activation='softmax'))

model.set_weights(w)

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
show_webcam(model)




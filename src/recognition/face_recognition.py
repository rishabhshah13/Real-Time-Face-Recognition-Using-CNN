import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Activation, Flatten
from PIL import Image
from Model import model
import time

sys.path.append('../..')
from config import *

def getImagesAndLabels():
    imagePaths = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')
        except:
            continue
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples, ids

_, ids = getImagesAndLabels()
model = model(IMAGE_SIZE + (1,), len(set(ids)))
model.load_weights(MODEL_PATH)
model.summary()

# Open the file in read mode
with open(os.path.join(DATASET_PATH, 'labels.txt'), 'r') as file:
    labels = file.read().split('\n')



import sys
sys.path.append('../..')
from config import *

faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX
def start():

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps variable


    cap = cv2.VideoCapture(0)
    print('here')
    ret = True

    clip = []
    while ret:
        #read frame by frame
        ret, frame = cap.read()
        nframe = frame
        faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE)


        # Calculate FPS and display it on the image
        frame_count += 1
        if frame_count >= 1:  # Calculate FPS every 30 frames (adjust as needed)
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(frame_count, ' /q ' , elapsed_time)
            frame_count = 0
            start_time = time.time()

        #Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        try:
            (x,y,w,h) = faces[0]
        except:
            continue
        frame = frame[y:y+h,x:x+w]
        frame = cv2.resize(frame, (32,32))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result small' , frame)
        c= cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
        
        #gray = gray[np.newaxis,:,:,np.newaxis]
        gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
        print(gray.shape)
        prediction = model.predict(gray)
        print("prediction:" + str(prediction))

        
        print("\n\n\n\n")
        print("----------------------------------------------")
        # labels = ['Rishabh']
        prediction = prediction.tolist()
        
        listv = prediction[0]
        n = listv.index(max(listv))
        print("\n")
        print("----------------------------------------------")
        #print( "Highest Probability: " + labels[n] + "==>" + str(prediction[0][n]) )
        print( "Highest Probability: " + "User " + str(n) + "==>" + str(prediction[0][n]) )
        
        print("----------------------------------------------")
        print("\n")
        for (x, y, w, h) in faces:
            try:
                cv2.rectangle(nframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(nframe, str(labels[n]), (x+5,y-5), font, 1, (255,255,255), 2)
                # cv2.putText(nframe, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            except:
                la = 2 
        prediction = np.argmax(model.predict(gray), 1)
        print(prediction)
        cv2.imshow('result', nframe)

        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
start()

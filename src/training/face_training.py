import cv2
import numpy as np
from PIL import Image
import os
import h5py
import dlib
from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from Model import model
from keras import callbacks

sys.path.append('../..')
from config import *

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    return np.array(img)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(downsample_image(img_numpy))
            ids.append(id)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
            continue

    return np.array(faceSamples), np.array(ids)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(FACE_CASCADE_PATH);


print ("\n [INFO] Training faces now.")
faces,ids = getImagesAndLabels(path)

K.clear_session()
n_faces = len(set(ids))
model = model((32,32,1),n_faces)
# faces = np.asarray(faces)
faces = np.array([downsample_image(ab) for ab in faces])
ids = np.asarray(ids)
faces = faces[:,:,:,np.newaxis]
print("Shape of Data: " + str(faces.shape))
print("Number of unique faces : " + str(n_faces))


ids = to_categorical(ids)

faces = faces.astype('float32')
faces /= 255.

x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.2, random_state = 0)

checkpoint = callbacks.ModelCheckpoint(MODEL_PATH,
                                           save_best_only=True, save_weights_only=True, verbose=1)
                                    
model.fit(x_train, y_train,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             validation_data=(x_test, y_test),
             shuffle=True,callbacks=[checkpoint])
             

# Print the numer of faces trained and end program
print("\n [INFO] " + str(n_faces) + " faces trained. Exiting Program")

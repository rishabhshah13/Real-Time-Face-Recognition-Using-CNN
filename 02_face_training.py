import cv2
import numpy as np
from PIL import Image
import os
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
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from Model import model


# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.ANTIALIAS)
    return np.array(img)



# function to get the images and label data
def getImagesAndLabels(path):
    
    path = 'dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
K.clear_session()


model = model((32,32,1),len(set(ids)))
faces = np.asarray(faces)
faces = np.array([downsample_image(ab) for ab in faces])
ids = np.asarray(ids)
faces = faces[:,:,:,np.newaxis]
print(str(faces.shape))
print(str(len(set(ids))))
print(str(len((ids))))

ids = to_categorical(ids.astype('float32'))

faces = faces.astype('float32')
faces /= 255

x_train, x_test, y_train, y_test = train_test_split(faces,ids, test_size = 0.2, random_state = 0)

model.fit(x_train, y_train,
             batch_size=32,
             epochs=10,
             validation_data=(x_test, y_test),
             shuffle=True)
#recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
model.save("trained_model.hdf5")

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

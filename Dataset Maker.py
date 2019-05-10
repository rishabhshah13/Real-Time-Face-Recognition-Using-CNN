import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.datasets import fetch_lfw_people


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.ANTIALIAS)
    return np.array(img)



img_list = os.listdir()

x_data = []
y_data = []



def get_face_data():
    people = fetch_lfw_people(color=False, min_faces_per_person=300)
    X_faces = people.images
    X_faces = np.array([downsample_image(ab) for ab in X_faces])
    Y_faces = people.target
    names = people.target_names
    return X_faces,Y_faces,names
    
    
X_faces,Y_faces,names = get_face_data()


for img in img_list:    
    image = cv2.imread(img,0)
    image = downsample_image(image)
    print(str(image.shape))
    x_data.append(image)
    y_data.append(int(1))


x_data = np.asarray(x_data)
#x_data = x_data[:,:,:,np.newaxis]
y_data = np.asarray(y_data)


np.save('x_data.npy',x_data)
np.save('y_data.npy',y_data)


x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')


a = np.concatenate([X_faces,x_data])
b = np.concatenate([Y_faces,y_data])


np.save('x_data.npy',a)
np.save('y_data.npy',b)

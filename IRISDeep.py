# -*- coding: utf-8 -*-

# -- Sheet --

import cv2
import os 
import glob
from keras.callbacks import History

# Getting the current path
pth = os.getcwd()
Db_pth = pth + "/Database"

# This function is for getting the parameters from dataset and takes the iris coordinations
def get_parameters(pth):
    cordinates = []
    names = []

    with open(pth + "/Parameters.txt") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if (i >= 2):
            line = lines[i].split(",")
            element = [int(line[1]), int(line[2]), int(line[3].strip())]
            cordinates.append(element)
            names.append(line[0])

    return cordinates, names


iris_cordinates, names = get_parameters(Db_pth)

images = []
path_list = []
index = 0
# Getting images from Dataset file
for path in names:
    p = Db_pth + '/' + path
    path_list.append(p)
"""
for image_path in glob.glob(Db_pth+'/'+names[index]):
    image = cv2.imread(image_path)
    images.append(image)
    index+=1
"""
for image_path in path_list:
    image = cv2.imread(image_path)
    images.append(image)
    index += 1

import numpy as np
import matplotlib.pyplot as plt

index = 0
print(len(images))
#plt.imshow(images[0])
#plt.show()
changed_image_list = []

# Taking coordinates and cropping given images
for image in images:
    new_img = np.asarray(image)
    
    
    #new_img = np.array(image)
    x,y,r=iris_cordinates[index]
    new_img=new_img[x-r:x+r,y-r:y+r]
    changed_image_list.append(new_img)

    index += 1

# Resizing the cropped images
new_image_list=[]
for image in changed_image_list:
    resized_image = cv2.resize(image, (128, 128),interpolation = cv2.INTER_NEAREST)
    new_image_list.append(resized_image)

changed_image_list=new_image_list



from numpy import linalg as LA
from IPython.display import Image 
from sklearn.preprocessing import normalize

normalized_images = []

index=1
x=[]
y=[]
x_train=[]
y_train=[]

x_validation=[]
y_validation=[]

x_test=[]
y_test=[]
index=0

# Populating the train, validation and test sets
for i in range(len(changed_image_list)):
    if(i%4!=0):
        y.append(index)        
    if(i%4==0):
        index=index+1
        y.append(index)

#print(y)
index = 0
changed_image_list.reverse()
y.reverse()
for i in range (len(changed_image_list)):
    if(index<2):
        a=changed_image_list.pop()
        x_train.append(a)
        i=y.pop()
        y_train.append(i)
        index=index+1
    elif(index<3):
        a=changed_image_list.pop()
        x_validation.append(a)
        i=y.pop()
        y_validation.append(i)
        index=index+1
    elif(index<4):
        a=changed_image_list.pop()
        x_test.append(a)
        i=y.pop()
        y_test.append(i)
        index=0



#print(y_train)
#print(y_validation)
#print(y_test)

# Converting lists to numpy arrays
x_train=np.array(x_train).astype(np.float32)
x_validation=np.array(x_validation).astype(np.float32)
x_test=np.array(x_test).astype(np.float32)

y_train=np.array(y_train)
y_validation=np.array(y_validation)
y_test=np.array(y_test)

#print(x_train.shape)
#print(y_train.shape)

# Appliyin oneHot Encode to labels for catagorical crossentrophy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = np.array(y_train)
y1 = y_train.reshape(-1,1)
y_train = ohe.fit_transform(y1).toarray()


y_validation = np.array(y_validation)
y1 = y_validation.reshape(-1,1)
y_validation = ohe.fit_transform(y1).toarray()

y_test = np.array(y_test)
y1 = y_test.reshape(-1,1)
y_test = ohe.fit_transform(y1).toarray()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import *
#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

#x_train = tf.math.l2_normalize(x_train, axis=-1)
#x_validation = tf.math.l2_normalize(x_validation, axis=-1)
#x_test = tf.math.l2_normalize(x_test, axis=-1)


# Appliying Normalisation
from tensorflow.keras.layers.experimental import preprocessing

layer = preprocessing.Normalization()
layer.adapt(x_train)
x_train = layer(x_train)
layer.adapt(x_validation)
x_validation = layer(x_validation)
layer.adapt(x_test)
x_test = layer(x_test)

#print(x_train)

# Building CNN

#from tensorflow.keras.regularizers import l2
CNN_model = Sequential()
CNN_model.add(layers.Conv2D(8,(5,5),activation='relu', padding='same'))
CNN_model.add(layers.MaxPooling2D((2,2),strides=2, padding='valid'))
CNN_model.add(layers.Conv2D(16,(5,5),activation='relu', padding='same'))
CNN_model.add(layers.MaxPooling2D((2,2),strides=2, padding='valid'))
CNN_model.add(layers.Flatten())
#CNN_model.add(Dense(500, activation='relu'))
#CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(400, activation='softmax'))

CNN_model.compile(loss="categorical_crossentropy", optimizer='RMSprop', metrics=['accuracy'])
from keras import backend as K

"""
CALLBACKS = [
      tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3),
      tf.keras.callbacks.ModelCheckpoint("CNN_model.hdf5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1),
      tf.keras.callbacks.TensorBoard(log_dir='./logs'),
      #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
]
"""
#Training Our Model

K.set_value(CNN_model.optimizer.learning_rate, 0.001)
history = CNN_model.fit(x_train,y_train, epochs=25 ,batch_size=64,validation_data=(x_validation, y_validation))
print(CNN_model.summary())
#h = model2.fit(x_new,y_new, epochs=50, batch_size=64,sample_weight=class_weight)


# Creating Charts for Model Loss and Accuracy
import keras
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#print(len(x_train))
#print(len(y_train))

# Predicting the results

ynew = CNN_model.predict_classes(x_validation)

print(ynew)

x_test = np.asarray(x_test).astype(np.float)
y_pred = CNN_model.predict(x_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_validation[i]))

predic=CNN_model.evaluate(x_test,y_test,verbose=0)
print(f"Results :{predic}")


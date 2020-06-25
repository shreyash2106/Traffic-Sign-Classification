import numpy as np 
import pandas as pd 
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D

# Define a Callback class that stops training once accuracy reaches 97%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

data = []
labels = []
class_size = 43

for file in range(class_size) :
    path = "./Dataset/Train/{0}/".format(file)
    Class = os.listdir(path)
    for c in Class:
        try:
            image = cv2.imread(path + c)
            image_from_array = Image.fromarray(image, 'RGB')
            resized_image = image_from_array.resize((30, 30))
            resized_images = np.array(resized_image)
            data.append(resized_images)
            labels.append(file)
        except AttributeError:
            print("Not able to load image data. ")

Images=np.array(data)
labels=np.array(labels)

# Shuffling the data 
value = np.arange(Images.shape[0])
np.random.seed(class_size)
np.random.shuffle(value)
Images = Images[value]
labels = labels[value]
image_data = len(Images)
labels_data = len(labels)

# Spliting the images into train and validation sets 
(training_images, validation_images) = Images[(int)(0.2 * labels_data):], Images[:(int)(0.2 * labels_data)]
training_images = training_images.astype('float32') / 255 
validation_images = validation_images.astype('float32') / 255
(training_labels, validation_labels) = labels[(int)(0.2 * labels_data):], labels[:(int)(0.2 * labels_data)]
training_shape = training_images.shape[1:]
# Using one hot encoding for the labels
from keras.utils import to_categorical
training_labels = to_categorical(training_labels, class_size)
validation_labels = to_categorical(validation_labels, class_size)

# DNN Model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(5,5), padding='same', activation='relu', input_shape = training_shape),
    tf.keras.layers.Conv2D(16, kernel_size=(5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape = training_shape),
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape = training_shape),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(class_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.environ['KMP_DUPLICATE_LIB_OK']='True'

history = model.fit(training_images, 
                    training_labels, 
                    validation_data = (validation_images, validation_labels), 
                    batch_size = 32, 
                    epochs = 20,
                    verbose = 1,
                    callbacks = [callbacks])

model.save("neural_net.h5")
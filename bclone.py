#!/usr/bin/python3
import json
import numpy as np
import pandas
from sklearn.utils import shuffle
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout
import cv2
from tqdm import tqdm
from skimage import exposure
from pathlib import Path
import pdb
from sklearn.model_selection import train_test_split


img_size = (160,80)
batch_size = 64
epochs = 5

def preprocessImage(X):
    
    #X = 0.299 * X[:, :, 0] + 0.587 * X[:, :, 1] + 0.114 * X[:, :, 2] #Y conversion
    X = (X / 255.).astype(np.float32)
      
    # Apply localized histogram localization  
    #for i in tqdm(range(X.shape[0])):
    #    X[i] = exposure.equalize_adapthist(X[i])   
    
    #X = X.reshape(X.shape + (1,))
    return X

def load_log():
  drive_log = pandas.read_csv("driving_log.csv")
  drive_log.columns = ["center","left","right","steering","throttle","brake", "speed"]
  return drive_log


def lameModel(input_shape):
  # Create the Sequential model
  model = Sequential()

  model.add(Convolution2D(32, 3,3,
                          border_mode='valid',
                          input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Activation('relu'))

  model.add(Convolution2D(64, 3,3,
                          border_mode='valid'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Activation('relu'))

  model.add(Convolution2D(128, 3,3,
                          border_mode='valid'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  # 1st Layer - Add a flatten layer
  model.add(Flatten())

  model.add(Dense(2048))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
                    
  # 2nd Layer - Add a fully connected layer
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  # 4th Layer - Add a fully connected layer
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(1))

  # 5th Layer - Add a ReLU activation layer
  #model.add(Activation('softmax'))
  # TODO: Build a Multi-layer feedforward neural network with Keras here.
  # TODO: Compile and train the model
  return model



def Xgen(drive_log, batch_size, imgsize):
  
  while True:
    total = 0
    drive_log = shuffle(drive_log)
    batch_i = 0
    images = np.zeros((batch_size,) + imgsize + (3,))
    steering_data = np.zeros(batch_size)

    for idx, row in drive_log.iterrows():
      images = np.zeros((batch_size,) + imgsize + (3,))
      steering_data = np.zeros(batch_size)
    
      image = cv2.imread(row.center)
      image = cv2.resize(image, imgsize[::-1])
      image = preprocessImage(image)

      images[batch_i] = image
      steering_data[batch_i] = row.steering
      batch_i += 1
      total += 1

      if (batch_i >= batch_size):
        print("returning {} images at idx {} total {} ".format(batch_size, idx, total))
        batch_i = 0
        yield (images, steering_data)
    print("Out of data {}, return images length {}".format(total, len(images)))

    yield (images[:batch_i,:,:,:],steering_data[:batch_i])

if __name__ == '__main__':
  drive_log = load_log()
  inp_shape = img_size + (3,)
  model = lameModel(inp_shape)
  model.compile('adam', 'mse', ['accuracy'])
  print("length is {}".format(len(drive_log)))
  train_log , test_log = train_test_split(drive_log, test_size = 0.1)
  model.fit_generator(Xgen(train_log, batch_size, img_size), nb_epoch=epochs, samples_per_epoch=len(train_log), verbose=2, validation_data=Xgen(test_log, batch_size, img_size), nb_val_samples=len(test_log))
  model.save('awesome_driver.h5')





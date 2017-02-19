#!/usr/bin/python3
import json
import numpy as np
import pandas
import pylab
from sklearn.utils import shuffle
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Input
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
import cv2
from tqdm import tqdm
from skimage import exposure
from pathlib import Path
import pdb
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


img_size = (160,48)
CROP_TOP=60
CROP_BOTTOM=140

batch_size = 64
epochs = 50
steering_angles = []
zero_angles = 0
OFF_CENTER_IMG = 0.3

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

def preprocessImage(image, imgsize):
    
    #X = 0.299 * X[:, :, 0] + 0.587 * X[:, :, 1] + 0.114 * X[:, :, 2] #Y conversion
    image = image[CROP_TOP:CROP_BOTTOM,:,:]
    image = cv2.resize(image, imgsize, interpolation=cv2.INTER_AREA)
    image = (image / 255.).astype(np.float32)
    return image

def load_log():
  drive_log = pandas.read_csv("driving_log.csv")
  drive_log.columns = ["center","left","right","steering","throttle","brake", "speed"]
  return drive_log


def anotherModel(input_shape):
  init_type = "glorot_uniform"

  model = Sequential()
  model.add(Convolution2D(32, 5, 5, input_shape=input_shape, init=init_type, activation='relu', border_mode='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Convolution2D(64, 5, 5, init=init_type, activation='relu', border_mode='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Convolution2D(128, 5, 5, init=init_type, activation='relu', border_mode='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(1164, activation='relu', init=init_type))
  model.add(Dropout(0.2))
  model.add(Dense(100, activation='relu', init=init_type))
  model.add(Dense(60, activation='relu', init=init_type))
  model.add(Dense(10, activation='relu', init=init_type))
  model.add(Dense(1))
  return model

def nvidiaModel(input_shape):
  init_type = "glorot_uniform"
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, input_shape=input_shape, init=init_type, activation='relu', border_mode='same', subsample=(2,2)))
  model.add(Convolution2D(36, 5, 5, init=init_type, activation='relu', border_mode='same', subsample=(2,2)))
  model.add(Convolution2D(48, 5, 5, init=init_type, activation='relu', border_mode='same', subsample=(2,2)))
  model.add(Convolution2D(64, 3, 3, init=init_type, activation='relu', border_mode='same', subsample=(2,2)))
  model.add(Convolution2D(64, 3, 3, init=init_type, activation='relu', border_mode='same', subsample=(2,2)))
  
  model.add(Flatten())
  model.add(Dropout(0.3))
  model.add(Dense(1164, activation='relu', init=init_type))
  model.add(Dropout(0.3))
  model.add(Dense(100, activation='relu', init=init_type))
  model.add(Dropout(0.3))
  model.add(Dense(60, activation='relu', init=init_type))
  model.add(Dropout(0.3))
  model.add(Dense(10, activation='relu', init=init_type))
  model.add(Dense(1))
  return model

def coolModel(input_shape):
  # Create the Sequential model
  init_type = "glorot_uniform"

  model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
  #for l in model_vgg16_conv.layers:
  #  l.trainable = False
  model_vgg16_conv.summary()


  input = Input(shape=input_shape,name = 'image_input')
  output_vgg16_conv = model_vgg16_conv(input)


  #Add the fully-connected layers 
  x = Flatten(name='flatten')(output_vgg16_conv)
  x = Dense(1024, activation='relu', name='fc1', init=init_type)(x)
  x = Dropout(0.5)(x)
  x = Dense(512, activation='relu', name='fc2', init=init_type)(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation='relu', name='fc3', init=init_type)(x)
  x = Dropout(0.5)(x)
  x = Dense(64, activation='relu', name='fc4', init=init_type)(x)
  x = Dropout(0.5)(x)
  x = Dense(32, activation='relu', name='fc5', init=init_type)(x)
  x = Dense(1)(x)

  # 5th Layer - Add a ReLU activation layer
  #model.add(Activation('softmax'))
  # TODO: Build a Multi-layer feedforward neural network with Keras here.
  # TODO: Compile and train the model
  return Model(input=input, output=x)

def Xgen(drive_log, batch_size, imgsize, bias):
  global steering_angles, zero_angles
  while True:

    num_samples = 0
    images = np.zeros((batch_size,) + imgsize[::-1] + (3,))
    steering_data = np.zeros(batch_size)
    while num_samples < batch_size:
      row = drive_log.iloc[np.random.randint(len(drive_log))]
      steering_angle = row.steering
      if (steering_angle == 0. and np.random.uniform() > 0.05):
        continue

      
      img_path = ""
      img_choice = np.random.randint(3)
      if img_choice == 0:
          img_path = row.left
          steering_angle += OFF_CENTER_IMG
      elif img_choice == 1:
          img_path = row.center
      else:
          img_path = row.right
          steering_angle -= OFF_CENTER_IMG


      steering_angle = clamp(steering_angle, -1.0, 1.0)

      image = cv2.imread(img_path.strip())
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = preprocessImage(image, imgsize)
      if (np.random.randint(2) == 0):
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
      images[num_samples] = image
      steering_data[num_samples] = steering_angle
      steering_angles.append(steering_angle)
      num_samples += 1
      if (row.steering == 0):
        zero_angles += 1

    yield (images, steering_data)

if __name__ == '__main__':
  log = load_log()
  inp_shape = img_size[::-1] + (3,)
  model = nvidiaModel(inp_shape)
  model.summary()
  model.compile(loss = 'mse', optimizer=Adam(lr=0.0001))
  split_train_log , test_log = train_test_split(log, test_size = 0.1)
  for i in tqdm(range(1)):
    drive_log = log
    print("length is {}".format(len(drive_log)))
    model.fit_generator(Xgen(drive_log, batch_size, img_size,1), nb_epoch=epochs, samples_per_epoch=10000, verbose=2, validation_data=Xgen(test_log, batch_size, img_size, 1), nb_val_samples=100)
    print('Zero angles: {}'.format(zero_angles))
    zero_angles = 0
    
  model.save('awesome_driver.h5')
  pandas.DataFrame(steering_angles).hist(bins=100)
  pylab.show()
  steering_angles = []





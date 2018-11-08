from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import cifar10
import keras.backend as K
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# load 'CIFAR10' dataset ---------------------------------------------------------
num_train_samples = 50000

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = K.cast_to_floatx(x_train) / 255
x_test = K.cast_to_floatx(x_test) / 255
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# data normalization
# normalize - samplewise for CNN and featurewis for fullyconnected
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

x_train, x_test = normalize(x_train, x_test)
# ---------------------------------------------------------------------------------

# define a model ------------------------------------------------------------------
weight_decay = 0
dropout = 0.3

model = Sequential()
model.add(
  Conv2D(
        15,
        (3, 3),
        padding='same',
        activation='relu',
        input_shape=(32,32,3),
  )
)
model.add((BatchNormalization()))
model.add(
  Conv2D(
        15,
        (3, 3),
        padding='same',
        activation='relu',
  )
)
model.add((BatchNormalization()))
model.add(MaxPool2D())
model.add(
  Conv2D(
        28,
        (3, 3),
        padding='same',
        activation='relu',
  )
)
model.add((BatchNormalization()))
model.add(
  Conv2D(
        28,
        (3, 3),
        padding='same',
        activation='relu',
  )
)
model.add((BatchNormalization()))
model.add(MaxPool2D())
model.add(
  Conv2D(
        42,
        (3, 3),
        padding='same',
        activation='relu',
  )
)
model.add((BatchNormalization()))
model.add(
  Conv2D(
        42,
        (3, 3),
        padding='same',
        activation='relu',
  )
)
model.add((BatchNormalization()))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dropout(dropout))
model.add((BatchNormalization()))
model.add(Dense(10, activation='softmax',
                )
          )

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
filepath = 'cifar10_nofully_mod'
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
checkPoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkPoint]
model.summary()
history = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=64),
                                verbose=2,
                                epochs=400,
                                callbacks=callbacks_list,
                                validation_data=(x_test, y_test))

import os
import numpy as np
from keras.layers import Dense,  Dropout
from keras.layers import Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from numpy.random import shuffle
from keras.callbacks import ModelCheckpoint
import keras
import keras.backend as K
import keras.losses



def load_photo(directory, size, doc):
    # load all features
    files_list = os.listdir(directory)
    files_list = sorted(files_list)
    examples_num = len(files_list)
    x = np.zeros((examples_num, size, size, 1))
    if doc == 1:
        y = np.zeros((examples_num, 1))
    else:
        y = np.ones((examples_num, 1))
    i = 0
    for name in files_list:
        filename = directory + '/' + name
        image = load_img(filename, grayscale=True, target_size=(size, size))
        # convert the image pixels to a numpy array
        image = keras.backend.cast_to_floatx(img_to_array(image)) / 255
        x[i, :, :, :] = image
        i += 1
    return x, y




# def data_generator(train_mat, labels_mat, batch_size):
#     while 1:
#         idx = list(range(train_mat.shape[0]))
#         shuffle(idx)
#         for ind in range(0, len(idx), batch_size):
#             X1, y = list(), list()
#             for i in range(batch_size):
#                 if ind + i < len(idx):
#                     for imSizeIdx in range(3):
#                         cur_images = train_mat[imSizeIdx][idx[ind + i], :, :, :, :]
#                         seq_length = cur_images.shape[0]
#                         for j in range(seq_length):
#                             rand_indexs = list(range(seq_length))
#                             [n_idx, n_flag] = find_neighbour(j, seq_length, orientation)
#                             rand_indexs.remove(n_idx)
#                             rand_indexs.remove(j)
#                             temp_x1 = return_vector(j, n_idx, cur_images, orientation)
#                             X1.append(temp_x1)
#                             y.append(n_flag)
#                             for s in range(int(len(rand_indexs)/4)):
#                                 idx1 = rand_indexs[s]
#                                 temp_x1 = return_vector(j, idx1, cur_images, orientation)
#                                 X1.append(temp_x1)
#                                 y.append(0)
#             yield np.array(X1), np.array(y)

seq_length = [4, 16, 25]
x_train, x_test, y_train, y_test = list(), list(), list(), list()
pic_dim = 64
dir1 = ['/home/control06/etai_eyal/deep_proj/doc_2x2', '/home/control06/etai_eyal/deep_proj/doc_4x4',
       '/home/control06/etai_eyal/deep_proj/doc_5x5']
dir2 = ['/home/control06/etai_eyal/deep_proj/images_2x2', '/home/control06/etai_eyal/deep_proj/images_4x4',
        '/home/control06/etai_eyal/deep_proj/images_5x5']

x = np.array([])
y = np.array([])

for i in range(3):
    x1, y1 = load_photo(dir1[i], pic_dim, doc=1)
    x2, y2 = load_photo(dir2[i], pic_dim, doc=0)
    if i == 0:
        x = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([y1, y2], axis=0)
    else:
        x = np.concatenate((x, x1, x2), axis=0)
        y = np.concatenate([y, y1, y2], axis=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,)


del x, y, x1, x2, y1, y2

def define_model(pic_dim):
    input_shape = (pic_dim, pic_dim, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    return model


batch_size = 128
orientation = 3
example_num = x_train[0].shape[0]
valid_num = x_test[0].shape[0]
epochs = 10
filepath = 'image_classifier.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = define_model(pic_dim)
# model = keras.models.load_model('/home/control06/etai_eyal/deep_proj/down_only_doc_size_128_all_sizes_v2.h5')

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test), callbacks=[checkpoint])

# import matplotlib.pyplot as plt
# his = history.history
# x = list(range(epochs))
# y_1 = his['val_loss']
# y_2 = his['loss']
# plt.plot(x,y_1)
# plt.plot(x,y_2)
# plt.legend(['validation loss','training_loss'])
# plt.savefig('hybrid_training_loss_stam.png')


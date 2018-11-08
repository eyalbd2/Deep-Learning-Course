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



def load_photo(directory,seq_len, size):
    # load all features
    files_list = os.listdir(directory)
    files_list = sorted(files_list)
    examples_num = int(len(files_list) / seq_len)
    x = np.zeros((examples_num, seq_len, size, size, 1))
    y = np.zeros((examples_num, seq_len))
    i = 0
    for name in files_list:
        filename = directory + '/' + name
        image = load_img(filename, grayscale=True, target_size=(size, size))
        # convert the image pixels to a numpy array
        image = keras.backend.cast_to_floatx(img_to_array(image)) / 255
        label = name.split('_')[-1].split('.')[0]
        x[int(i/seq_len), int(label), :, :, :] = image
        y[int(i/seq_len), int(label)] = label
        i += 1
    return x, y


def calc_cross_cor(x ,y , orientation):
    if orientation == 0:  # up
        x_vec = x[0, :]
        y_vec = y[0, :]
    elif orientation == 1:  # down
        x_vec = x[-1, :]
        y_vec = y[-1, :]
    elif orientation == 2:  # left
        x_vec = x[:, 0]
        y_vec = y[:, 0]
    elif orientation == 3:  # right
        x_vec = x[:, -1]
        y_vec = y[:, -1]
    return np.sum((x_vec-y_vec)**2)


def find_neighbour(cur_image_id, seq_length , orientation):
    if orientation == 0: # up
        n_id = int(cur_image_id - np.sqrt(seq_length))
        if n_id < 0:
            return (n_id % seq_length), 0
        else:
            return n_id, 1

    elif orientation == 1: # down
        n_id = int(cur_image_id + np.sqrt(seq_length))
        if n_id >= seq_length:
            return (n_id % seq_length), 0
        else:
            return n_id, 1

    elif orientation == 2: #left
        if cur_image_id % np.sqrt(seq_length) == 0:
            return int((cur_image_id-1)%seq_length), 0
        else:
            return int(cur_image_id-1), 1

    elif orientation == 3: # right
        n_id = cur_image_id + 1
        if n_id % np.sqrt(seq_length) == 0:
            return int((n_id)%seq_length), 0
        else:
            return int(n_id), 1


def return_vector(j, idx, cur_images, orientation):
    if orientation == 0: # up
        x = (np.concatenate((cur_images[idx, :, :, :], cur_images[j,:, :, :]), axis=0))

    elif orientation == 1: # down
        x = (np.concatenate((cur_images[j, :, :, :], cur_images[idx, :, :, :]), axis=0))

    elif orientation == 2:  # left
        x = (np.concatenate((cur_images[idx, :, :, :], cur_images[j, :, :, :]), axis=1))

    elif orientation == 3:  # right
        x = (np.concatenate((cur_images[j, :, :, :], cur_images[idx, :, :, :]), axis=1))

    return x

def data_generator(train_mat, batch_size, orientation=0):
    while 1:
        idx = list(range(train_mat[0].shape[0]))
        shuffle(idx)
        for ind in range(0, len(idx), batch_size):
            X1, y = list(), list()
            for i in range(batch_size):
                if ind + i < len(idx):
                    for imSizeIdx in range(3):
                        cur_images = train_mat[imSizeIdx][idx[ind + i], :, :, :, :]
                        seq_length = cur_images.shape[0]
                        for j in range(seq_length):
                            rand_indexs = list(range(seq_length))
                            [n_idx, n_flag] = find_neighbour(j, seq_length, orientation)
                            rand_indexs.remove(n_idx)
                            rand_indexs.remove(j)
                            temp_x1 = return_vector(j, n_idx, cur_images, orientation)
                            X1.append(temp_x1)
                            y.append(n_flag)
                            for s in range(int(len(rand_indexs)/4)):
                                idx1 = rand_indexs[s]
                                temp_x1 = return_vector(j, idx1, cur_images, orientation)
                                X1.append(temp_x1)
                                y.append(0)
            yield np.array(X1), np.array(y)

seq_length = [4, 16, 25]
x_train, x_test, y_train, y_test = list(), list(), list(), list()
pic_dim = 128
dir1 = ['/home/control06/etai_eyal/deep_proj/doc_2x2', '/home/control06/etai_eyal/deep_proj/doc_4x4',
       '/home/control06/etai_eyal/deep_proj/doc_5x5']
# dir2 = ['/home/control06/etai_eyal/deep_proj/images_2x2', '/home/control06/etai_eyal/deep_proj/images_4x4',
#        '/home/control06/etai_eyal/deep_proj/images_5x5']

for i in range(3):
    x1, y1 = load_photo(dir1[i], seq_length[i], pic_dim)

    # x2, y2 = load_photo(dir2[i], seq_length[i], pic_dim)

    # x = np.concatenate((x1, x2), axis=0)
    # y = np.concatenate((y1, y2), axis=0)

    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x1, y1, test_size=0.2, )
    # x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x, y, test_size=0.2,)
    x_train.append(x_train_temp)
    x_test.append(x_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)

    del x1, y1, x_train_temp, x_test_temp, y_train_temp, y_test_temp
    # del x1, x2, y1, y2, x, y, x_train_temp, x_test_temp, y_train_temp, y_test_temp


def custom_loss(y_true, y_pred):
    return K.sum(-1.5*y_true*K.log(y_pred+0.0001) -1*(1-y_true)*K.log(1-(0.999*y_pred)))

keras.losses.custom_loss = custom_loss


def define_model(pic_dim, orientation):
    if orientation > 1:
        input_shape = (pic_dim, int(2*pic_dim), 1)
    else:
        input_shape = ((2*pic_dim), pic_dim, 1)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    # summarize model
    return model


batch_size = 1
orientation = 3
example_num = x_train[0].shape[0]
valid_num = x_test[0].shape[0]
epochs = 10
filepath = 'right_only_doc_size_128_all_sizes_v3.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = define_model(pic_dim, orientation)
# model = keras.models.load_model('/home/control06/etai_eyal/deep_proj/down_only_doc_size_128_all_sizes_v2.h5')

history = model.fit_generator(data_generator(x_train, batch_size=batch_size, orientation=orientation),
                                             steps_per_epoch=(example_num // batch_size), epochs=epochs, verbose=2,
                                             validation_data=data_generator(x_test, batch_size=batch_size, orientation=orientation),
                                             validation_steps=(valid_num // batch_size), callbacks=[checkpoint])

# import matplotlib.pyplot as plt
# his = history.history
# x = list(range(epochs))
# y_1 = his['val_loss']
# y_2 = his['loss']
# plt.plot(x,y_1)
# plt.plot(x,y_2)
# plt.legend(['validation loss','training_loss'])
# plt.savefig('hybrid_training_loss_stam.png')


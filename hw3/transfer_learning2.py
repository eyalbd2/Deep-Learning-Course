
# imports
from sklearn.model_selection import train_test_split
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.utils import np_utils
from keras.datasets import cifar10, cifar100
import keras.backend as K
import numpy as np
from cifar100vgg import cifar100vgg as vgg
import keras
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm

# load 'CIFAR10' dataset ---------------------------------------------------------

num_train_samples = 50000
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# reshape the data
x_train = K.cast_to_floatx(x_train) / 255
x_test = K.cast_to_floatx(x_test) / 255
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
y_test_1hot = np_utils.to_categorical(y_test, 10)
#%%
# data normalization if using transfer learning normalize based on relevant data set

def normalize(X_train, X_test, cifar10=True):
    if cifar10:
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test
    else:
        (x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
        x_train_100 = K.cast_to_floatx(x_train_100) / 255
        x_test_100 = K.cast_to_floatx(x_test_100) / 255
        x_train_100 = x_train_100.reshape(-1, 32, 32, 3)
        mean = np.mean(x_train_100, axis=(0, 1, 2, 3))
        std = np.std(x_train_100, axis=(0, 1, 2, 3))
        x_train = (X_train - mean) / (std + 1e-7)
        x_test = (X_test - mean) / (std + 1e-7)
        return x_train, x_test


# get the data
X_train, X_test = normalize(x_train, x_test,cifar10=False)

# split the data into small training set
X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train,
train_size=1000, random_state=42, stratify=y_train)

# set variables
image_input = (32, 32, 3)
y_train_small_1hot = np_utils.to_categorical(y_train_small, 10)

#%% fine tunning model
# remove top two layers (fully connected ,activation)
vgg = vgg(train=False)
vgg.model.layers.pop()
vgg.model.layers.pop()

# set remaining layers to not train
for layer in vgg.model.layers:
    layer.trainable = False

# configure last layer as output
fine_tune = vgg.model
last_layer_name = fine_tune.layers[-1].name
last_layer = fine_tune.get_layer(last_layer_name).output

# New output layer
out = Dense(10, activation='softmax', name='fc')(last_layer)
fine_tune = keras.models.Model(input=fine_tune.input, output=out)

# show that only last layer trains
fine_tune.summary()
fine_tune.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#%% fine tunning the model
hist = fine_tune.fit(X_train_small, y_train_small_1hot, batch_size=128, epochs=100, verbose=2)

#%% evaluate fine tuned model
score = fine_tune.evaluate(X_test, y_test_1hot)
print("the fine tune model scored: %f" % (score))

#%% KNN classifier
# remove top two layers
vgg.model.layers.pop()
vgg.model.layers.pop()
last_layer_name = vgg.model.layers[-1].name
last_layer = vgg.model.get_layer(last_layer_name).output
vgg_no_head = keras.models.Model(input=vgg.model.input, output=last_layer)

# get the out put from the last layer
X_pred = vgg_no_head.predict(X_train_small)
knn_class = KNeighborsClassifier(n_neighbors=5)
# train the knn classifier on the output of the NN)
knn_class.fit(X_pred, np.ravel(y_train_small))
# evaluate model
X_test_no_head = vgg_no_head.predict(X_test)
score = knn_class.score(X_test_no_head, np.ravel(y_test))
print("the KNN model scored: %f" % (score))

#%% our clasifier SVM
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_pred, np.ravel(y_train_small))
score = clf.score(X_test_no_head, np.ravel(y_test))
print("the SVM model scored: %f" %(score))

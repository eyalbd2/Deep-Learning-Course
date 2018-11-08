import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
import random
import numpy as np
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import CuDNNLSTM

# load the dataset but only keep the top words, zero the rest. Introduce special tokens.
top_words = 6000
(X_train, y_train), _= imdb.load_data(num_words=top_words, skip_top=0, index_from=4)

X1_train, X1_test, X2_train, X2_test = train_test_split(X_train, y_train,
                                                        test_size=0.1,
                                                        stratify=y_train
                                                        )

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
word_to_id["<EOS>"] = 3
id_to_word = {v: k for k, v in word_to_id.items()}

for x in X_train:
    x.append(3)

print(len(X1_train))
print(len(X1_test))





def data_generator(max_length, review_list, labels, vocab_size, batch_size=128):
    while 1:
        idx = list(range(len(labels)))
        random.shuffle(idx)
        for ind in range(0, len(idx), batch_size):
            X1, X2, y = list(), list(), list()
            for i in range(batch_size):
                if ind+i < len(idx):
                    cur_review = review_list[idx[ind + i]]
                    cur_label = labels[idx[ind + i]]
                    # temp = list()
                    # for word_ind in range(1, len(cur_review)):
                    # in_seq, out_seq = cur_review[:word_ind], cur_review[word_ind]
                    # temp.append(in_seq)
                    X1.append(sequence.pad_sequences([cur_review], maxlen=max_length, truncating='post')[0])
                    X2.append(cur_label)
                    # predict the next word
                    pred = np.roll(X1[-1], -1, axis=-1)
                    pred[-1] = word_to_id["<EOS>"]
                    y.append(np.array(to_categorical(pred, num_classes=vocab_size)))
            yield [array(X1), array(X2)], array(y)

# def data_generator(reviews, max_length, sentiments, vocab_size, batch_size=128):
#
#     while 1:
#         # keys = [key for (key,desc_list) in descriptions.items()]
#         idx = list(range(len(sentiments)))
#         random.shuffle(idx)
#         for ind in range(0, len(idx), batch_size):
#             X1, X2, y = list(), list(), list()
#
#             for i in range(batch_size):
#                 if ind+i < len(idx):
#                     sentiment = sentiments[idx[ind+i]]
#                     review = reviews[idx[ind+i]]
#                     X1.append(sentiment)
#                     X2.append(pad_sequences([review], maxlen=max_length)[0])
#                     y_ = np.roll(X2[-1], -1, axis=-1)  # we want to predict the next character
#                     y.append(np.array(to_categorical(y_, num_classes=vocab_size)))
#             yield [array(X1), array(X2)], array(y)
from keras.layers import TimeDistributed


# fit model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 128, mask_zero=True)(inputs1)
    se2 = Dropout(0.3)(se1)
    # sequence model
    inputs2 = Input(shape=(1,))
    fe1 = Dense(128, activation='relu')(inputs2)
    fe2 = Dropout(0.3)(fe1)
    # merge layer
    merge = add([fe2, se2])
    # language model
    bn = BatchNormalization()(merge)
    lstm1 = LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.1)(bn)
    lstm2 = LSTM(256, return_sequences=True, dropout=0.4,  recurrent_dropout=0.1)(lstm1)
    lstm3 = LSTM(256, return_sequences=True, dropout=0.4,  recurrent_dropout=0.1)(lstm2)

    # fully connected
    bn = BatchNormalization()(lstm3)
    decoder2 = TimeDistributed(Dense(256, activation='relu'))(bn)
    bn = BatchNormalization()(decoder2)
    do = Dropout(0.3)(bn)
    decoder3 = TimeDistributed(Dense(512, activation='relu'))(do)
    bn = BatchNormalization()(decoder3)
    do = Dropout(0.3)(bn)
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(do)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # summarize model
    print(model.summary())
    return model
# define the model
vocab_size = top_words
max_length = 100
model = define_model(vocab_size, max_length)
batch_size = 128
# define checkpoint callback
filepath = 'best_text_gen_3.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit_generator(data_generator(max_length, X1_train, X2_train, vocab_size, batch_size=128),
                    steps_per_epoch=(22500 // batch_size), epochs=100, verbose=2,
                    validation_data=data_generator(max_length, X1_test, X2_test, vocab_size, batch_size=128),
                    validation_steps=(2500 // batch_size), callbacks=[checkpoint])
